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
from picker.msg import Prompt, BoxArray, Box, CircleArray, Circle
import re
from openai_wrapper import generate_request
import openai
from picker.srv import PickConfig, PickConfigResponse, PickConfigRequest
from threading import Thread

openai.api_key = 'API key'
openai.api_base = "http://127.0.0.1:8080"
# base_prompt = '''On the table are {available_objects}.
# List the order of stacking the vertical tower from these colored blocks.
# Write each action as: Pick the "some block" and place it on the "another block".
# Separate actions with a new line. At the end of the work, type "done()". 
# '''
base_prompt = '''On the table are {available_objects}.
Write each action as: pick_place(a, b).
Separate actions with a new line. At the end of the work, type "done()". 
To place green and yellow blocks into the matching color plates, I should:
'''
class SceneDescriptionGenerator:
    boxes_topic_name = "/alpaca/detector/boxes"
    circles_topic_name = "/alpaca/detector/circles"
    pick_box_srv_name = "/alpaca/get_pick_config"
    pick_circle_srv_name = "/alpaca/get_pick_config_circle"

    def __init__(self):
        self._pick_thread = None
        self._boxes = None
        self._circles = None
        self._boxes_subscriber = rospy.Subscriber(self.boxes_topic_name, BoxArray, callback_args={"type": "boxes"}, queue_size=2, callback=self._item_detected_cb)
        self._circles_subscriber = rospy.Subscriber(self.circles_topic_name, CircleArray, callback_args={"type": "circles"}, queue_size=2, callback=self._item_detected_cb)
        rospy.wait_for_service(self.pick_box_srv_name, timeout=5)
        self._pick_config_srv = rospy.ServiceProxy(self.pick_box_srv_name, PickConfig)
        rospy.wait_for_service(self.pick_circle_srv_name, timeout=5)
        self._pick_circle_config_srv = rospy.ServiceProxy(self.pick_circle_srv_name, PickConfig)

    def _item_detected_cb(self, msg, args):

        if args['type'] == "boxes":
            self._boxes = msg
        elif args['type'] == "circles":
            self._circles = msg
        else:
            raise RuntimeError(f"Unknown type {args['type']}")
        if self._boxes is not None and self._circles is not None:
            if self._boxes.header.stamp == self._circles.header.stamp:
                stamp = self._boxes.header.stamp
                description = self._generate_available_actions(self._boxes.boxes, self._circles.circles)
                self._boxes = None
                self._circles = None
                self._publish_available_actions(stamp, description)
    
    def _generate_available_actions(self, boxes: List, circles: List):
        '''generates list of strings describing the scene
        :param boxes: list of boxes
        :param circles: list of circles
        :return: list of available actions
        '''
        variants = []
        pick_objs = []
        place_objs = []
        pick_objs.extend(boxes)
        place_objs.extend(boxes)
        place_objs.extend(circles)
        pick_names = list(map(lambda x: x.name, pick_objs))
        place_names = list(map(lambda x: x.name, place_objs))
        for pick_name, pick_obj in zip (pick_names, pick_objs):
            for place_name, place_obj in zip(place_names, place_objs):
                # pass
                if pick_obj.pos[1] < 80:
                    continue
                if place_obj.pos[1] < 80:
                    continue
                if place_name == "fish":
                    continue
                if pick_obj is place_obj:
                    continue
                variants.append({"pick": pick_obj,
                    "place": place_obj,
                    "text": f"pick_place({pick_name}, {place_name})"})
        return variants

    def _publish_available_actions(self, stamp, description):
        global base_prompt
        '''publishes scene description
        :param stamp: stamp of the scene description
        :param description: list of available actions
        '''
        variants = list(map(lambda x: x["text"], description))
        available_objects = list(map(lambda x: x['pick'].name, description)) + list(map(lambda x: x['place'].name, description))
        available_objects = list(set(available_objects))
        print("base_prompt:\n", base_prompt)
        # print("variants:\n", *variants, sep="\n")
        prompt = generate_request(base_prompt, variants)
        prompt = prompt.format(available_objects=", ".join(variants))
        # print("PROMPT:\n", prompt)
        completion = openai.Completion.create(model="llama-7B-4b", prompt=prompt, max_tokens=0, logprobs=True, echo=True)
        logprobs_avgs = [sum(choice.logprobs.token_logprobs[1:]) / len(choice.logprobs.token_logprobs)-1 for choice in completion.choices]
        rated = [{"text": text, "logprobs": logprobs_avg, "pick_obj": pick_obj, "place_obj": place_obj} for text, logprobs_avg, pick_obj, place_obj in zip(variants, logprobs_avgs, map(lambda x: x["pick"], description), map(lambda x: x["place"], description))]
        rated.sort(key=lambda x: x["logprobs"], reverse=True)
        for i, variant in enumerate(rated):
            rospy.loginfo(f"{i}: {variant['text']}, logprobs: {variant['logprobs']}, pick: {variant['pick_obj'].name}, place: {variant['place_obj'].name}")
        rospy.loginfo(f"Selected: {rated[0]['text']}")
        base_prompt += rated[0]["text"] + "\n"
        self._pick_and_place(rated[0]["pick_obj"], rated[0]["place_obj"])

    def _pick_and_place(self, pick_obj, place_obj):
        if self._pick_thread is not None:
            is_alive = self._pick_thread.is_alive()
            if is_alive:
                raise RuntimeError("pick thread is already running")
        self._pick_thread = Thread(target=self._move_gripper_thread, daemon=True, args=(pick_obj.name, pick_obj.pos, place_obj.name, place_obj.pos, type(place_obj) == Circle))
        self._pick_thread.start()
        self._pick_thread.join()

    def _move_gripper_thread(self, item_name: str, item_pos: Tuple[int, int], place_name: str, place_pos: Tuple[int, int], is_circle: bool):
        timer = rospy.Rate(0.5)
        for i in range(5):
            if i > 0:
                rospy.loginfo(f"retrying {i + 1}...")
            pick_config_req = PickConfigRequest(
                header=rospy.Header(),
                object_name=item_name,
                pos=item_pos,
            )
            place_config_req = PickConfigRequest(
                    header=rospy.Header(),
                    object_name=place_name,
                    pos=place_pos,
            )
            try:
                pick_config = self._pick_config_srv(pick_config_req)
                print("pick_config:\n", pick_config)
                if not is_circle:
                    place_config = self._pick_config_srv(place_config_req)
                    place_config.object_position.z -= 0.05
                    print("place_config:\n", place_config)
                else:
                    place_config = self._pick_circle_config_srv(place_config_req)
                    place_config.object_position.z -= 0.08
                    place_config.object_orientation = 1.57
                    print("place_config:\n", place_config)
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                rospy.signal_shutdown(f"Service call failed: {e}")
            if pick_config.success:
                self._move_gripper((pick_config.object_position.x,
                                        pick_config.object_position.y,
                                        pick_config.object_position.z), 
                                pick_config.object_orientation, 
                                pick_config.object_width, place=False)
                self._move_gripper((place_config.object_position.x,
                                        place_config.object_position.y,
                                        place_config.object_position.z), 
                                place_config.object_orientation, 
                                place_config.object_width, place=True)
                break
            else:
                rospy.logwarn(f"picking is not possible, because: {pick_config.reason}")
                timer.sleep()
        
    def _move_gripper(self, object_pos: Tuple[float, float, float], object_orientation: float, object_width: float, place: bool):
        # rospy.loginfo(f"object pos: {object_pos}\n\torientation: {object_orientation}\n\twidth: {object_width}")

        pre_grasp = [
            ac.Point6D(object_pos[0], object_pos[1], object_pos[2] - 0.2, np.pi, 0, object_orientation),
            ac.Point6D(object_pos[0], object_pos[1], object_pos[2] - 0.008, np.pi, 0, object_orientation),
        ]
        grasp = [
            ac.Point6D(object_pos[0], object_pos[1], object_pos[2] - 0.2, np.pi, 0, object_orientation),
            ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)
        ]
        if not place:
            ret = ac.gripper(False)
            if not ret:
                return False
            ret = ac.move_by_camera(pre_grasp)
            if not ret:
                return False
            ret = ac.gripper(True)
            if not ret:
                return False
            ret = ac.move_by_camera(grasp)
            if not ret:
                return False
            ret = ac.gripper(True)
            if not ret:
                return False
            return True
        else:
            ret = ac.move_by_camera(pre_grasp)
            if not ret:
                return False
            ret = ac.gripper(False)
            if not ret:
                return False
            ret = ac.move_by_camera(grasp)
            if not ret:
                return False
            ret = ac.gripper(False)
            if not ret:
                return False
            return True
        

        
if __name__ == "__main__":
    ac.init_node("description_generator")
    ret = ac.move_by_camera([ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)])
    if not ret:
        rospy.logerr("Failed to move to start position")
        exit(1)
    description_generator = SceneDescriptionGenerator()
    rospy.spin()