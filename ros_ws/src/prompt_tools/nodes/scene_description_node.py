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

class SceneDescriptionGenerator:
    boxes_topic_name = "/alpaca/detector/boxes"
    circles_topic_name = "/alpaca/detector/circles"

    def __init__(self):
        self._boxes = None
        self._circles = None
        self._boxes_subscriber = rospy.Subscriber(self.boxes_topic_name, BoxArray, callback_args={"type": "boxes"}, queue_size=2, callback=self._item_detected_cb)
        self._circles_subscriber = rospy.Subscriber(self.circles_topic_name, CircleArray, callback_args={"type": "circles"}, queue_size=2, callback=self._item_detected_cb)

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
                description = self._generate_scene_description(self._boxes.boxes, self._circles.circles)
                self._boxes = None
                self._circles = None
                self._publish_scene_description(stamp, description)
    
    def _generate_scene_description(self, boxes: List, circles: List):
        '''generates list of strings describing the scene
        :param boxes: list of boxes
        :param circles: list of circles
        :return: list of strings describing the scene
        '''
        return [
            "box on plate",
            "box on box",
        ]

    def _publish_scene_description(self, stamp, description):
        '''publishes scene description
        :param stamp: stamp of the scene description
        :param description: list of strings describing the scene
        '''
        rospy.loginfo(f"publishing scene description: {description} with stamp: {stamp}")
    
if __name__ == "__main__":
    rospy.init_node("description_generator")
    description_generator = SceneDescriptionGenerator()
    rospy.spin()