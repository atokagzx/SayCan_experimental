#! /usr/bin/env python3

from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
from PIL import Image
import os, sys
from collections import namedtuple
from rospy import loginfo, logerr, logwarn
import matplotlib.pyplot as plt
from modules.gsam_detector import GSAMDetector
from modules.plates_detector import PlateDetector
from modules.utils import CroppedImage, DrawingUtils



class Picker:
    def __init__(self, height_map_path, plain_offset):
        self._height_map = np.load(height_map_path)
        self._plain_offset = plain_offset
        self._gripper_length = 130 # [mm]
        self._gripper_width = 30 # [mm]

    def _get_pick_config(self, item, depth_map):
        mask = item.mask
        points = np.argwhere(mask)[:,:2] # [y, x]
        points = points[:, ::-1] # [x, y]
        points = points.astype(np.int32)
        real_points = np.array(ac.to_real_points(points, depth_map), dtype=np.float32)
        # [[x, y, z], ...] -> [z, ...]
        heights = real_points[:, 2]
        top_height = np.min(heights)
        mean_height = np.mean(heights)
        angle = item.angle
        width = item.width
        length = item.length
        # pick_point [x, y, angle]
        point_offset_1 = length / 2 * np.cos(angle), length / 2 * np.sin(angle)
        pick_points_1 = np.array([item.pos - point_offset_1, item.pos + point_offset_1, angle], dtype=np.int32)
        point_offset_2 = width / 2 * np.cos(angle + np.pi / 2), width / 2 * np.sin(angle + np.pi / 2)
        pick_points_2 = np.array([item.pos - point_offset_2, item.pos + point_offset_2, angle + np.pi / 2], dtype=np.int32)
        return {
            "pick_points_1": pick_points_1,
            "pick_points_2": pick_points_2,
            "mean_height": mean_height,
            "top_height": top_height,
        }

    def _check_pick_points(self, item, pick_points:List, mean_height, top_height, depth_map, is_plate=False):
        max_pick_height_center = self._height_map[item.pos[1], item.pos[0]] - self._plain_offset
        for pick_point in pick_points:
            real_pick_points = np.asarray(ac.to_real_points(pick_point[:, :2], depth_map))
            # distance between pick point
            distance = np.linalg.norm(real_pick_points[0] - real_pick_points[1])
            if distance > 90:
                continue
            upper_point = np.min(real_pick_points[0][2], real_pick_points[1][2])
            max_pick_height = np.min(upper_point, max_pick_height_center)

    def _check_collisions(self, item, height, angle, depthmap, plate=None, gripper_width=130):
        item_mask = item.mask.copy()
        # dilate item mask
        kernel = np.ones((5, 5), np.uint8)
        item_mask = cv2.dilate(item_mask, kernel, iterations=2)
        collision_mask = np.zeros_like(item_mask)
        collision_mask[depthmap > self._height_map + self._plain_offset] = 1
        collision_mask[item_mask] = 0
        # draw gripper projection
        

def mouse_cb(event, x, y, flags, param):
    global items_detected
    global selected_item
    if event == cv2.EVENT_LBUTTONDOWN:
        sorted_items = sorted(items_detected, key=lambda item: np.linalg.norm(item.pos - np.array([x, y])))
        if len(sorted_items) > 0:
            if np.linalg.norm(sorted_items[0].pos - np.array([x, y])) < 100:
                item = sorted_items[0]
                print(f"Item {item.name} selected")
                selected_item = item

height_map_path = "/workspace/config/height_map.npy"
plain_offset = 1 # [mm]

names_to_detect = ["liquid soap", "fish", "cup", "cream", "book", "fork", "spoon", "block", "tape"]
items_detected = []
selected_item = None
def pick_item(ac, item):
    color, depth_map, camera_info = ac.pop_image()
    picker._get_pick_config(item, depth_map)
    # item_real_pos = ac.to_real_points([item.pos], depth_map, camera_info)[0]
    # angle = 2 * np.pi - item.angle
    # print(f"Item {item.name} real pos: {item_real_pos}, angle: {item.angle}")
    # trajectory = [
    #     ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2),
    #     ac.Point6D(item_real_pos[0], item_real_pos[1], 0.7, np.pi, 0, angle),
    # ]
    # ret = ac.move_by_camera(trajectory)
    # loginfo(f"Move ret: {ret}")
    # ac.gripper(False)
    # trajectory = [
    #     ac.Point6D(item_real_pos[0], item_real_pos[1], item_real_pos[2]+0.01, np.pi, 0, angle),
    # ]
    # ret = ac.move_by_camera(trajectory)
    # loginfo(f"Move ret: {ret}")
    # ac.gripper(True)
    # trajectory = [
    #     ac.Point6D(item_real_pos[0], item_real_pos[1], 0.7, np.pi, 0, angle),
    #     ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2),
    # ]
    # ret = ac.move_by_camera(trajectory)
    # loginfo(f"Move ret: {ret}")
    # ac.gripper(False)

if __name__ == "__main__":
    ac.init_node("alpaca_gsam_example")
    ret = ac.move_by_camera([ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)])
    loginfo(f"Move ret: {ret}")
    detector = GSAMDetector()
    picker = Picker(height_map_path, plain_offset=plain_offset)
    cv2.namedWindow("items_detected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("items_detected", 1920, 1080)
    cv2.setMouseCallback("items_detected", mouse_cb)
    while not ac.is_shutdown():
        if ac.is_image_ready():
            if selected_item is not None:
                pick_item(ac, selected_item)
                selected_item = None
            color, depth_map, _camera_info = ac.pop_image()
            crop = CroppedImage(color, [330, 0, 1600, 1080])
            items = detector.get_items(crop(), names_to_detect)
            items = GSAMDetector.filter_same_items(items)
            items = crop.coords_transform(items)
            items_detected = items
            masked_frame = color.copy()
            for item in items:
                masked_frame = DrawingUtils.draw_box_info(masked_frame, item)
            crop.image = masked_frame
            cv2.imshow("items_detected", masked_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            sleep(0.01)