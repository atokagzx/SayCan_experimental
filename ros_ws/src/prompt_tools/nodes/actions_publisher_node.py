#! /usr/bin/env python3

from typing import Any, List, Literal, Tuple, Dict
import rospy
from picker.msg import Prompt, BoxArray, Box, CircleArray, Circle
from prompt_tools.msg import Prompt

class ActionsPublisherNode:
    boxes_topic_name = "/alpaca/detector/boxes"
    circles_topic_name = "/alpaca/detector/circles"
    prompt_topic_name = "/alpaca/prompt/actions"
    pickable_objects = ["block", "fish"]
    placeable_objects = ["block", "plate"]
    
    def __init__(self, available_items: List[str], base_prompt: str, action_template: str):
        self._boxes = None
        self._circles = None
        self._available_items = available_items
        self._base_prompt = base_prompt
        self._action_template = action_template
        self._boxes_subscriber = rospy.Subscriber(self.boxes_topic_name, BoxArray, callback_args={"type": "boxes"}, queue_size=2, callback=self._item_detected_cb)
        self._circles_subscriber = rospy.Subscriber(self.circles_topic_name, CircleArray, callback_args={"type": "circles"}, queue_size=2, callback=self._item_detected_cb)
        self._prompt_publisher = rospy.Publisher(self.prompt_topic_name, Prompt, queue_size=1)

    def _item_detected_cb(self, msg, args):

        if args['type'] == "boxes":
            self._boxes = msg
        elif args['type'] == "circles":
            self._circles = msg
        else:
            raise RuntimeError(f"Unknown type {args['type']}")
        if self._boxes is not None and self._circles is not None:
            if self._boxes.header.stamp == self._circles.header.stamp:
                description = self._generate_available_actions(self._boxes, self._circles)
                stamp = self._boxes.header.stamp
                self._boxes = None
                self._circles = None
                self._publish_available_actions(stamp, description)

    def _filter_objects(self, objects: List[str], type: Literal["pick", "place"]):
        if type == "pick":
            check_list = self.pickable_objects
        elif type == "place":
            check_list = self.placeable_objects
        else:
            raise ValueError(f"unknown action type: {type}")
        for obj in objects:
            for check_obj in check_list:
                if (check_obj in obj):
                    yield obj
                    break
                    

    def _generate_available_actions(self, boxes: List, circles: List):
        '''generates list of strings describing the scene
        :param boxes: list of boxes
        :param circles: list of circles
        :return: list of available actions
        '''
        box_names = list(map(lambda x: x.name, boxes.boxes))
        circle_names = list(map(lambda x: x.name, circles.circles))
        action_list = []
        pick_objs = []
        place_objs = []
        pick_objs.extend(box_names)
        place_objs.extend(box_names)
        place_objs.extend(circle_names)
        pick_objs = list(set(pick_objs))
        place_objs = list(set(place_objs))
        pick_objs = list(self._filter_objects(pick_objs, "pick"))
        place_objs = list(self._filter_objects(place_objs, "place"))
        for pick_name in pick_objs:
            for place_name in place_objs:
                if pick_name == place_name:
                    continue
                action_list.append({
                    "text": self._action_template.format(pick_name=pick_name, place_name=place_name),
                    "pick_obj": pick_name,
                    "place_obj": place_name
                })
        action_list.append({
            "text": "robot_all_actions_done()",
            "pick_obj": "",
            "place_obj": ""
        })
        return action_list

    def _publish_available_actions(self, stamp, description):
        message = Prompt()
        message.header.stamp = stamp
        message.header.frame_id = "actions_publisher"
        message.body = self._base_prompt.format(available_objects=", ".join(self._available_items))
        message.actions = list(map(lambda x: x["text"], description))
        message.pick_names = list(map(lambda x: x["pick_obj"], description))
        message.place_names = list(map(lambda x: x["place_obj"], description))
        self._prompt_publisher.publish(message)

if __name__ == "__main__":
    rospy.init_node("actions_publisher_node")
    available_items = rospy.get_param("~available_items", "block;plate").split(";")
    base_prompt = rospy.get_param("~base_prompt", "Available objects: {available_objects}\n")
    action_template = rospy.get_param("~action_template", "pick_place({pick_name}, {place_name})")
    description_generator = ActionsPublisherNode(base_prompt=base_prompt, 
                                                 action_template=action_template, 
                                                 available_items=available_items)
    rospy.spin()