#! /usr/bin/env python3

from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
from PIL import Image
import os, sys
import rospy
import matplotlib.pyplot as plt
import std_msgs.msg as std_msgs
from geometry_msgs.msg import Point
from picker.msg import BoxArray, Box, CircleArray, Circle
from picker.srv import PickConfig, PickConfigResponse, PickConfigRequest
from picker.srv import PickPlace, PickPlaceResponse, PickPlaceRequest
from threading import Thread
from sensor_msgs.msg import Image, CameraInfo
from prompt_tools.msg import Prompt
from prompt_tools.srv import ActionsRate, ActionsRateRequest, ActionsRateResponse
from prompt_tools.srv import DoneTask, DoneTaskRequest, DoneTaskResponse
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse

# task = "To make a tower using colored blocks, I should:"
# task = "To pick red block and place it on all other items one by one, I should:"
# task = "Match two blocks with plates of the same color:"
# task = "Make a tower from red, blue and green blocks"
# task = "Put the red block on each plate in turn:"
task = "Build a tower using colored blocks:"
class MainNode:
    rate_srv_name = "/alpaca/prompt/rate"
    add_done_task_srv_name = "/alpaca/prompt/add_done_task"
    reset_done_tasks_srv_name = "/alpaca/prompt/reset_done_tasks"
    pick_place_srv_name = "/alpaca/pick_place"

    boxes_topic_name = "/alpaca/detector/boxes"
    circles_topic_name = "/alpaca/detector/circles"

    def __init__(self, only_plan=False):
        
        self._boxes = None
        self._circles = None
        self._only_plan = only_plan
        wait_services = []
        if self._only_plan:
            wait_services = [self.rate_srv_name, self.add_done_task_srv_name, self.reset_done_tasks_srv_name]
        else:
            wait_services = [self.rate_srv_name, self.add_done_task_srv_name, self.reset_done_tasks_srv_name, self.pick_place_srv_name]
        for service_name in wait_services:
            try:
                rospy.wait_for_service(service_name, timeout=5)
            except rospy.ROSException:
                rospy.logerr("service {} not found".format(service_name))
                sys.exit(1)
        self._boxes_subscriber = rospy.Subscriber(self.boxes_topic_name, BoxArray, callback_args={"type": "boxes"}, queue_size=2, callback=self._item_detected_cb)
        self._circles_subscriber = rospy.Subscriber(self.circles_topic_name, CircleArray, callback_args={"type": "circles"}, queue_size=2, callback=self._item_detected_cb)
        if not self._only_plan:
            self._pick_place_srv = rospy.ServiceProxy(self.pick_place_srv_name, PickPlace)
        self._rate_srv = rospy.ServiceProxy(self.rate_srv_name, ActionsRate)
        self._add_done_task_srv = rospy.ServiceProxy(self.add_done_task_srv_name, DoneTask)
        self._reset_done_tasks_srv = rospy.ServiceProxy(self.reset_done_tasks_srv_name, Empty)
        self._last_items = None

    def _item_detected_cb(self, msg, args):
        if args["type"] == "boxes":
            self._boxes = msg
        elif args["type"] == "circles":
            self._circles = msg
        else:
            raise RuntimeError(f"Unknown type {args['type']}")
        
        if self._boxes is not None and self._circles is not None:
            if self._boxes.header.stamp == self._circles.header.stamp:
                self._last_items = {
                    "boxes": self._boxes,
                    "circles": self._circles,
                    "stamp": self._boxes.header.stamp
                }
                self._boxes = None
                self._circles = None
    
    def _rated_actions_to_dict(self, rated_response):
        prompt_body = rated_response.rated_actions.body
        done_tasks = rated_response.done_tasks
        actions_texts = rated_response.rated_actions.actions
        pick_objs = rated_response.rated_actions.pick_names
        place_objs = rated_response.rated_actions.place_names
        probabilities = rated_response.rated_actions.probabilities
        actions = (dict(zip(['text', 'pick', 'place', 'prob'], action)) for action in zip(actions_texts, pick_objs, place_objs, probabilities))
        actions = sorted(actions, key=lambda x: x['prob'], reverse=True)
        # logging
        # rospy.loginfo(f"prompt body:\n{prompt_body}")
        done_tasks_text = "\n".join(done_tasks)
        rospy.loginfo(f"already done tasks:\n{done_tasks_text}")
        actions_text = "\n".join(f"{i}: {action['text']} [{action['prob']}]" for i, action in enumerate(actions))
        rospy.loginfo(f"rated actions:\n{actions_text}")
        return prompt_body, done_tasks, actions
    
    def _find_pick_place_objs(self, pick_name, place_name):
        if self._last_items is None:
            raise RuntimeError("No items detected")
        boxes = self._last_items["boxes"]
        circles = self._last_items["circles"]
        boxes = boxes.boxes
        circles = circles.circles
        pick_obj = None
        place_obj = None
        all_items = boxes + circles
        for item in all_items:
            if item.name == pick_name:
                pick_obj = item
            elif item.name == place_name:
                place_obj = item
        if pick_obj is None:
            rospy.logwarn(f"pick object not found: {pick_name}")
        if place_obj is None:
            rospy.logwarn(f"place object not found: {place_name}")
        if pick_obj is None or place_obj is None:
            raise RuntimeError(f"pick or place object not found: pick={pick_name}, place={place_name}")
        return pick_obj, place_obj
    
    def _pick_place(self, selected_action, stamp):
        try:
            pick_obj, place_obj = self._find_pick_place_objs(selected_action["pick"], selected_action["place"])
        except RuntimeError as e:
            # if RuntimeError is raised, it means that no items were detected, so we just continue
            rospy.logerr(e)
            return False
        pick_place_request = PickPlaceRequest(
            header=rospy.Header(stamp=stamp),
            pick_object_name=pick_obj.name,
            pick_object_type=type(pick_obj).__name__,
            pick_object_pos=pick_obj.pos,
            place_object_name=place_obj.name,
            place_object_type=type(place_obj).__name__,
            place_object_pos=place_obj.pos
        )
        try:
            resp = self._pick_place_srv(pick_place_request)
        except rospy.ServiceException as e:
            rospy.logerr("failed to call pick_place service")
            rospy.logerr(e)
            return False
        else:
            if resp.success:
                rospy.loginfo("pick_place service succeeded")
                self._add_done_action(selected_action)
                return True
            else:
                rospy.logerr("pick_place service failed")
                rospy.logerr(resp.reason)
                return False
            
    def _add_done_action(self, action):
        add_task_req = DoneTaskRequest()
        add_task_req.task = action["text"]
        resp = self._add_done_task_srv(add_task_req)
        done_tasks_text = "\n".join(f"{i}: {task}" for i, task in enumerate(resp.done_tasks))
        rospy.loginfo(f"done tasks:\n{done_tasks_text}")

    def run(self):
        global task
        rospy.loginfo("main node started")
        # reset done tasks
        self._reset_done_tasks_srv(EmptyRequest())
        first_action_flag = True
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            rate_req = ActionsRateRequest()
            rate_req.task = task
            rate_req.stamp = stamp
            rospy.loginfo(f"requesting actions rate for task: {task}")
            rate_resp = self._rate_srv(rate_req)
            rospy.loginfo("received actions rate")
            _prompt_body, _done_tasks, actions = self._rated_actions_to_dict(rate_resp)
            if len(actions) == 0:
                rospy.loginfo("no actions to do")
                rospy.signal_shutdown("no actions to do")
                break
            selected_action = actions[0]
            rospy.loginfo(f"selected action: {selected_action['text']}")
            # check if "done" substring is in the selected action
            if "done" in selected_action["text"]:
                if first_action_flag and len(actions) > 1:
                    selected_action = actions[1]
                    rospy.logwarn(f"selected new action: {selected_action['text']}")
                else:
                    rospy.loginfo('"done()" action reached')
                    rospy.signal_shutdown('"done()" action reached')
                    break
            if rospy.is_shutdown():
                break
            if not self._only_plan:
                ret = self._pick_place(selected_action, stamp)
                if not ret:
                    continue
            else:
                self._add_done_action(selected_action)
            first_action_flag = False
            

if __name__ == "__main__":
    rospy.init_node("main_node")
    only_plan = rospy.get_param("~only_plan", False)
    main_node = MainNode(only_plan=only_plan)
    main_node.run()
