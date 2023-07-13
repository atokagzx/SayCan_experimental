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
# To place green, yellow and blue blocks into the matching color plates, I should:
# '''
pre_prompt = '''Write each action as: pick_place(a, b).
Separate actions with a new line.
At the end of the work, type "done()". 
For example, to pick blue block and place it on all other items one by one, I should:
pick_place(blue block, red block)
pick_place(blue block, green block)
pick_place(blue block, yellow block)
Do not repeat the same action twice.
Try not to use the same block twice.
For example, to pick fish and place it on all plates one by one, I should:
pick_place(fish, red plate)
pick_place(fish, green plate)
pick_place(fish, yellow plate)
done()
Here is an example of moving cookie on all cups one by one:
pick_place(cookie, cup 1)
pick_place(cookie, cup 2)
pick_place(cookie, cup 3)
done()
Here is an example how to separate bricks and meal between plates:
pick_place(fish, red plate)
pick_place(purple block, green plate)
pick_place(meat, red plate)
pick_place(egg, red plate)
pick_place(orange cube, green plate)
pick_place(bread, red plate)
pick_place(colorful brick, green plate)
pick_place(pink box, green plate)
done()
Here is an example how to separate meal and tools between plates:
pick_place(fish, red plate)
pick_place(meat, red plate)
pick_place(egg, red plate)
pick_place(bread, red plate)
pick_place(fork, green plate)
pick_place(knife, green plate)
pick_place(screwdriver, green plate)
pick_place(spoon, green plate)
done()
Here is an example of building a tower from green, yellow, black and white blocks:
pick_place(green block, yellow block)
pick_place(black block, green block)
pick_place(white block, black block)
done()
Here is an example of placing all blocks into the matching color plates:
pick_place(green block, green plate)
pick_place(yellow block, yellow plate)
pick_place(black block, black plate)
pick_place(white block, white plate)
done()
Here is an example of listing the order of stacking the vertical tower from these colored blocks:
pick_place(yellow block, black block)
pick_place(green block, yellow block)
pick_place(white block, green block)
pick_place(black block, white block)
done()
Here is an example how to make a tower using colored blocks if you have other objects on the table:
pick_place(purple block, pink block)
pick_place(orange block, purple block)
pick_place(blue block, orange block)
pick_place(green block, blue block)
pick_place(yellow block, green block)
done()
As you see, you should set the order of actions in the way that the robot can perform them.
Try to use as few actions as possible.
Now, complete the task:
'''
# task = '''On the table are: {available_objects}.
# To pick red block and place it on all plates one by one, I should:
# '''
task = '''On the table are: {available_objects}.
To make a tower using colored blocks ending with red block, I should:'''
# task = '''On the table are: {available_objects}.
# To separate fish and blocks into different plates, I should:
# '''
base_prompt = pre_prompt + task
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
# base_prompt = '''On the table are: {available_objects}.
# Write each action as: pick_place(a, b).
# For example, to pick blue block and place it on all other items one by one, I should:
# pick_place(blue block, red block)
# pick_place(blue block, green block)
# pick_place(blue block, yellow block)
# Do not repeat the same action twice.
# Separate actions with a new line. At the end of the work, type "done()". 
# To pick red block and place it on all plates one by one, I should:
# '''
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
        self._new_image = False
        self._description = []
        self._image_stamp = None
        self._prev_action = ""
        self._cummulitive_prompt = ""
        self._available_items = []
        

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
                self._description = description
                self._image_stamp = stamp
                self._new_image = True

    def run(self):
        while not rospy.is_shutdown():
            if self._new_image:
                is_done = self._publish_available_actions(self._image_stamp, self._description)
                if is_done:
                    break
                self._new_image = False
                for i in range(2):
                    while not rospy.is_shutdown():
                        if self._new_image:
                            self._new_image = False
                            break
                        rospy.sleep(0.1)
            rospy.sleep(0.1)
            
            
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
        if len(self._available_items) == 0:
            self._available_items = list(set(pick_names + place_names))
        for pick_name, pick_obj in zip (pick_names, pick_objs):
            for place_name, place_obj in zip(place_names, place_objs):
                if place_name == "fish":
                    continue
                if pick_obj is place_obj:
                    continue
                variants.append({"pick": pick_obj,
                    "place": place_obj,
                    "text": f"pick_place({pick_name}, {place_name})"})
        variants.append({"pick": None, "place": None, "text": "done()"})
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
        description = description.copy()
        if len(description) == 0:
            rospy.loginfo("No available actions")
            return True
        variants = list(map(lambda x: x["text"], description))
        if self._cummulitive_prompt == "":
            self._cummulitive_prompt = base_prompt.format(available_objects=', '.join(self._available_items))
        rospy.loginfo(f"prompt:\n{self._cummulitive_prompt}")
        promt_to_send = generate_request(self._cummulitive_prompt, variants)
        # print(f"PROMPT:\n{base_prompt.format(available_objects=', '.join(available_objects))}")
        # print("variants:\n", *variants, sep="\n")
        # prompt = generate_request(base_prompt, variants)
        # prompt = prompt.format(available_objects=", ".join(available_objects))
        self._generate_prompt_image(self._cummulitive_prompt, None, variants)
        # print("PROMPT:\n", prompt)
        completion = openai.Completion.create(model="llama-7B-4b", prompt=promt_to_send, max_tokens=0, logprobs=True, echo=True)
        logprobs_avgs = [sum(choice.logprobs.token_logprobs[1:]) / len(choice.logprobs.token_logprobs)-1 for choice in completion.choices]
        rated = [{"text": text, "logprobs": logprobs_avg, "pick_obj": pick_obj, "place_obj": place_obj} for text, logprobs_avg, pick_obj, place_obj in zip(variants, logprobs_avgs, map(lambda x: x["pick"], description), map(lambda x: x["place"], description))]
        rated.sort(key=lambda x: x["logprobs"], reverse=True)
        selected_variant = rated[0]
        if selected_variant['text'] == self._prev_action:
            selected_variant = rated[1]
        for i, variant in enumerate(rated):
            if variant['pick_obj'] is None or variant['place_obj'] is None:
                rospy.loginfo(f"{i}: {variant['text']}, logprobs: {variant['logprobs']}")
                continue
            rospy.loginfo(f"{i}: {variant['text']}, logprobs: {variant['logprobs']}, pick: {variant['pick_obj'].name}, place: {variant['place_obj'].name}")
        rospy.loginfo(f"selected: {selected_variant['text']}")
        self._generate_prompt_image(self._cummulitive_prompt, selected_variant["text"], variants)
        if selected_variant["text"] == "done()":
            rospy.loginfo("done")
            return True
        pick_place_request = PickPlaceRequest(
            header=rospy.Header(stamp=stamp),
            pick_object_name=selected_variant["pick_obj"].name,
            pick_object_type=type(selected_variant["pick_obj"]).__name__,
            pick_object_pos=selected_variant["pick_obj"].pos,
            place_object_name=selected_variant["place_obj"].name,
            place_object_type=type(selected_variant["place_obj"]).__name__,
            place_object_pos=selected_variant["place_obj"].pos
        )
        try:
            resp = self._pick_place_srv(pick_place_request)
        except rospy.ServiceException as e:
            rospy.logerr("failed to call pick_place service")
            rospy.logerr(e)
        else:
            if resp.success:
                rospy.loginfo("pick_place service succeeded")
                self._cummulitive_prompt += selected_variant["text"] + "\n"
                self._prev_action = selected_variant['text']
            else:
                rospy.logerr("pick_place service failed")
                rospy.logerr(resp.reason)
        return False
        # sleep(5)

if __name__ == "__main__":
    ac.init_node("description_generator")
    description_generator = SceneDescriptionGenerator()
    description_generator.run()