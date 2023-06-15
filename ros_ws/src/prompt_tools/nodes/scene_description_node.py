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
from picker.srv import PickPlace, PickPlaceResponse, PickPlaceRequest
from threading import Thread
from sensor_msgs.msg import Image, CameraInfo

openai.api_key = 'API key'
openai.api_base = "http://127.0.0.1:8080"
# base_prompt = '''On the table are {available_objects}.
# List the order of stacking the vertical tower from these colored blocks.
# Write each action as: Pick the "some block" and place it on the "another block".
# Separate actions with a new line. At the end of the work, type "done()". 
# '''
# base_prompt = '''On the table are {available_objects}.
# Write each action as: pick_place(a, b).
# Separate actions with a new line. At the end of the work, type "done()". 
# To place green and yellow blocks into the matching color plates, I should:
# '''
# base_prompt = '''On the table are: {available_objects}.
# Write each action as: pick_place(a, b).
# Separate actions with a new line. At the end of the work, type "done()". 
# To place green, yellow and blue blocks into the matching color plates, I should:
# '''
# base_prompt = '''On the table are: {available_objects}.
# Write each action as: pick_place(a, b).
# Separate actions with a new line. At the end of the work, type "done()". 
# To stack the vertical tower from blocks one by one, I should:
# '''
# base_prompt = '''On the table are: {available_objects}.
# Write each action as: pick_place(a, b).
# Separate actions with a new line. At the end of the work, type "done()". 
# To cook dinner using seafood, I should:
# '''
# base_prompt = '''On the table are: {available_objects}.
# Write each action as: pick_place(a, b).
# For example, to pick blue block and place it on all other items one by one, I should:
# pick_place(blue block, red block)
# pick_place(blue block, green block)
# pick_place(blue block, yellow block)
# Separate actions with a new line. At the end of the work, type "done()". 
# To pick blue block and place it on all other items one by one, I should:
# '''
base_prompt = '''On the table are: {available_objects}.
Write each action as: pick_place(a, b).
For example, to pick blue block and place it on all other items one by one, I should:
pick_place(blue block, red block)
pick_place(blue block, green block)
pick_place(blue block, yellow block)
Do not repeat the same action twice.
Separate actions with a new line. At the end of the work, type "done()". 
To pick red block and place it on all plates one by one, I should:
'''
# base_prompt = '''On the table are: {available_objects}.
# Write each action as: pick_place(a, b).
# For example, to pick blue block and place it on all other items one by one, I should:
# pick_place(blue block, red block)
# pick_place(blue block, green block)
# pick_place(blue block, yellow block)
# Do not repeat the same action twice.
# Separate actions with a new line. At the end of the work, type "done()". 
# Set all blocks to one plate and fish into another plate. Fish should be placed on green plate.
# '''
# base_prompt = '''On the table are: {available_objects}.
# Write each action as: pick_place(a, b).
# For example, to pick blue block and place it on all other items one by one, I should:
# pick_place(blue block, red block)
# pick_place(blue block, green block)
# pick_place(blue block, yellow block)
# For example, to pick fish and place it on all plates one by one, I should:
# pick_place(fish, red plate)
# pick_place(fish, green plate)
# pick_place(fish, yellow plate)
# Do not repeat the same action twice.
# Separate actions with a new line. At the end of the work, type "done()". 
# Set all blocks in one plate.
# '''
class SceneDescriptionGenerator:
    boxes_topic_name = "/alpaca/detector/boxes"
    circles_topic_name = "/alpaca/detector/circles"
    pick_place_srv_name = "/alpaca/pick_place"

    def __init__(self):
        rospy.wait_for_service(self.pick_place_srv_name, timeout=5)
        self._boxes = None
        self._circles = None
        self._boxes_subscriber = rospy.Subscriber(self.boxes_topic_name, BoxArray, callback_args={"type": "boxes"}, queue_size=2, callback=self._item_detected_cb)
        self._circles_subscriber = rospy.Subscriber(self.circles_topic_name, CircleArray, callback_args={"type": "circles"}, queue_size=2, callback=self._item_detected_cb)
        self._pick_place_srv = rospy.ServiceProxy(self.pick_place_srv_name, PickPlace)
        self._prompt_image_publisher = rospy.Publisher("/alpaca/prompt_image", Image, queue_size=1)
        self._generate_prompt_image("", None, [])

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
    
    def _generate_prompt_image(self, prompt: str, selected: str, variants: List[str]):
        image = np.full((600, 1200, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_vars_y = 0
        for i, line in enumerate(prompt.split("\n")):
            cv2.putText(image, line, (10, 20 + 20 * i), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            start_vars_y = 20 + 20 * i
        for i, variant in enumerate(variants):
            if selected is not None:
                if variant == selected:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.putText(image, variant, (10, start_vars_y + 20 + 20 * i), font, 0.6, color, 2, cv2.LINE_AA)
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        msg.step = image.shape[1] * 3
        msg.data = image.tobytes()
        self._prompt_image_publisher.publish(msg)


    def _publish_available_actions(self, stamp, description):
        global base_prompt
        '''publishes scene description
        :param stamp: stamp of the scene description
        :param description: list of available actions
        '''
        variants = list(map(lambda x: x["text"], description))
        available_objects = list(map(lambda x: x['pick'].name, description)) + list(map(lambda x: x['place'].name, description))
        available_objects = list(set(available_objects))
        print(f"PROMPT:\n{base_prompt.format(available_objects=', '.join(available_objects))}")
        # print("variants:\n", *variants, sep="\n")
        prompt = generate_request(base_prompt, variants)
        prompt = prompt.format(available_objects=", ".join(available_objects))
        self._generate_prompt_image(base_prompt.format(available_objects=', '.join(available_objects)), None, variants)
        # print("PROMPT:\n", prompt)
        completion = openai.Completion.create(model="llama-7B-4b", prompt=prompt, max_tokens=0, logprobs=True, echo=True)
        logprobs_avgs = [sum(choice.logprobs.token_logprobs[1:]) / len(choice.logprobs.token_logprobs)-1 for choice in completion.choices]
        rated = [{"text": text, "logprobs": logprobs_avg, "pick_obj": pick_obj, "place_obj": place_obj} for text, logprobs_avg, pick_obj, place_obj in zip(variants, logprobs_avgs, map(lambda x: x["pick"], description), map(lambda x: x["place"], description))]
        rated.sort(key=lambda x: x["logprobs"], reverse=True)
        for i, variant in enumerate(rated):
            rospy.loginfo(f"{i}: {variant['text']}, logprobs: {variant['logprobs']}, pick: {variant['pick_obj'].name}, place: {variant['place_obj'].name}")
        rospy.loginfo(f"selected: {rated[0]['text']}")
        self._generate_prompt_image(base_prompt.format(available_objects=', '.join(available_objects)), rated[0]["text"], variants)
        pick_place_request = PickPlaceRequest(
            header=rospy.Header(stamp=stamp),
            pick_object_name=rated[0]["pick_obj"].name,
            pick_object_type=type(rated[0]["pick_obj"]).__name__,
            pick_object_pos=rated[0]["pick_obj"].pos,
            place_object_name=rated[0]["place_obj"].name,
            place_object_type=type(rated[0]["place_obj"]).__name__,
            place_object_pos=rated[0]["place_obj"].pos
        )
        try:
            resp = self._pick_place_srv(pick_place_request)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call pick_place service")
            rospy.logerr(e)
        else:
            if resp.success:
                rospy.loginfo("pick_place service succeeded")
                base_prompt += rated[0]["text"] + "\n"
            else:
                rospy.logerr("pick_place service failed")
                rospy.logerr(resp.reason)
        sleep(5)

if __name__ == "__main__":
    ac.init_node("description_generator")
    description_generator = SceneDescriptionGenerator()
    rospy.spin()