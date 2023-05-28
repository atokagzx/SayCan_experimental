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
from modules.classes import Box, Circle
from modules.utils import DrawingUtils, normalize_depth
import std_msgs.msg as std_msgs
from modules.ros_utils import DataSubscriber

class Picker:
    def __init__(self, height_map_path, plain_offset):
        self._height_map = np.load(height_map_path)
        self._plain_offset = plain_offset
        self._gripper_footprint_length = 250 # [pixels]
        self._gripper_height = 0.06 # [mm]
        self._fingers_offset = 0 # [pixels]
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.items_detected = []

    def _get_pick_config(self, item):
        assert self.color_image is not None and self.depth_image is not None and self.camera_info is not None
        mask = item.mask
        points = np.argwhere(mask)[:,:2] # [y, x]
        points = points[:, ::-1] # [x, y]
        points = points.astype(np.int32)
        real_points = np.array(ac.to_real_points(points, self.depth_image, self.camera_info), dtype=np.float32)
        # [[x, y, z], ...] -> [z, ...]
        heights = real_points[:, 2]
        non_zero_heights = heights[heights != 0]
        if len(non_zero_heights) == 0:
            rospy.logerr(f"no non-zero heights for item {item.name}")
            return  {
                "pick_points": [],
                "mean_height": 0,
                "top_height": 0,
            }
        top_height = np.min(non_zero_heights)
        mean_height = np.mean(heights)
        angle = item.angle
        width = item.width + self._fingers_offset
        length = item.length + self._fingers_offset
        # pick_point [x, y, angle]
        point_offset_1 = [self._get_point_by_angle_and_distance(angle, length / 2, item.pos), 
                          self._get_point_by_angle_and_distance(angle, -length / 2, item.pos)]
        pick_points_1 = {"pos": np.array([point_offset_1[0], point_offset_1[1]], dtype=np.int32), "angle": angle}
        point_offset_2 = [self._get_point_by_angle_and_distance(angle + np.pi / 2, width / 2, item.pos),
                            self._get_point_by_angle_and_distance(angle + np.pi / 2, -width / 2, item.pos)]
        pick_points_2 = {"pos": np.array([point_offset_2[0], point_offset_2[1]], dtype=np.int32), "angle": angle + np.pi / 2}
        return {
            "edge_points": [pick_points_1, pick_points_2],
            "mean_height": mean_height,
            "top_height": top_height,
        }
    def _get_point_by_angle_and_distance(self, angle, distance, center):
        return np.array([center[0] + distance * np.cos(angle), center[1] + distance * np.sin(angle)], dtype=np.int32)
    
    def _draw_pick_config(self, pick_config):
        masked = self.color_image.copy()
        pick_points_1 = pick_config["edge_points"][0]
        pick_points_2 = pick_config["edge_points"][1]
        for pick_point in [pick_points_1, pick_points_2]:
            cv2.line(masked, tuple(pick_point["pos"][0]), tuple(pick_point["pos"][1]), (0, 0, 255), 2)
            cv2.circle(masked, tuple(pick_point["pos"][0]), 5, (0, 0, 255), -1)
            cv2.circle(masked, tuple(pick_point["pos"][1]), 5, (0, 0, 255), -1)
        # cv2.namedWindow("masked", cv2.WINDOW_NORMAL)
        # cv2.imshow("masked", masked)

    def _get_fingers_collision(self, item, angle):
        # deproject gripper footprint on height map
        grip_points_pixels = [self._get_point_by_angle_and_distance(angle, self._gripper_footprint_length/2, item.pos),
                                self._get_point_by_angle_and_distance(angle, -self._gripper_footprint_length/2, item.pos)]
        footprint_mask = np.zeros([self.color_image.shape[0], self.color_image.shape[1]], dtype=np.uint8)
        item_mask = item.mask.copy()
        kernel = np.ones((10,10),np.uint8)
        item_mask = cv2.dilate(item_mask,kernel,iterations = 3)
        footprint_mask = cv2.line(footprint_mask, tuple(grip_points_pixels[0]), tuple(grip_points_pixels[1]), 255, 100)
        footprint_mask = cv2.bitwise_and(footprint_mask, footprint_mask, mask=~item_mask)
        points_to_check_collision = np.argwhere(footprint_mask)
        # swap x and y
        points_to_check_collision = np.asarray([points_to_check_collision[:, 1], points_to_check_collision[:, 0]]).T
        grip_points = np.asarray(ac.to_real_points(points_to_check_collision, self.depth_image, self.camera_info))[:, 2]
        print(grip_points)
        grip_points = grip_points[grip_points != 0]
        lowest_point, lowest_point_index = np.min(grip_points), np.argmin(grip_points)
        # draw on image
        masked_color = self.color_image.copy()
        masked_color[footprint_mask == 255] = 0
        cv2.circle(masked_color, tuple(points_to_check_collision[lowest_point_index]), 10, (0, 0, 255), -1)
        cv2.namedWindow("gripper_footprint", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("gripper_footprint", 1280, 720)
        cv2.imshow("gripper_footprint", masked_color)
        return lowest_point

    def _check_pick_points(self, item, pick_config, on_plate=False):
        max_pick_height_center = self._height_map[item.pos[1], item.pos[0]] - self._plain_offset
        real_center = np.array(ac.to_real_points([item.pos], self.depth_image, self.camera_info)[0])
        body_collision = pick_config['top_height'] + self._gripper_height
        preferred_height = pick_config['mean_height'] + 0.015
        for pick_point in pick_config['edge_points']:
            
            angle = pick_point['angle']
            points = pick_point['pos']
            real_edge_points = np.asarray(ac.to_real_points(points[:, :2], self.depth_image, self.camera_info))
            # distance between pick point
            distance = np.linalg.norm(real_edge_points[0] - real_edge_points[1])
            print(f"Distance: {distance}")
            if distance > 0.10:
                continue
            obstacles_height = self._get_fingers_collision(item, angle)
            max_pick_height = np.min([body_collision, obstacles_height])
            rospy.loginfo(f"height points:\nbody_collision: {body_collision}\nobstacles_height: {obstacles_height}\nmax_pick_height: {max_pick_height}\npreferred_height: {preferred_height}")
            if max_pick_height < preferred_height:
                continue
            rospy.loginfo(f"Pick height: {preferred_height}")
            pick_pose = np.array([real_center[0], real_center[1], max_pick_height]) 
            print(pick_pose)
            ret = self.execute_picking(pick_pose, angle)
            if ret:
                break

    def execute_picking(self, pick_point, angle):
        angle = -angle + np.pi/2
        pre_grasp = [
            ac.Point6D(pick_point[0], pick_point[1], pick_point[2] - 0.2, np.pi, 0, angle),
            ac.Point6D(pick_point[0], pick_point[1], pick_point[2] - 0.008, np.pi, 0, angle),
        ]
        grasp = [
            ac.Point6D(pick_point[0], pick_point[1], pick_point[2] - 0.2, np.pi, 0, angle),
            ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)
        ]
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
        ret = ac.gripper(False)
        if not ret:
            return False
        return True
    def _check_collisions(self, item, height, angle, depthmap, plate=None, gripper_width=130):
        item_mask = item.mask.copy()
        # dilate item mask
        kernel = np.ones((5, 5), np.uint8)
        item_mask = cv2.dilate(item_mask, kernel, iterations=2)
        collision_mask = np.zeros_like(item_mask)
        collision_mask[depthmap > self._height_map + self._plain_offset] = 1
        collision_mask[item_mask] = 0
        # draw gripper projection
    
    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sorted_items = sorted(self.items_detected, key=lambda item: np.linalg.norm(item.pos - np.array([x, y])))
            if len(sorted_items) > 0:
                if np.linalg.norm(sorted_items[0].pos - np.array([x, y])) < 100:
                    item = sorted_items[0]
                    print(f"Item {item.name} selected")
                    pick_config = self._get_pick_config(item)
                    self._draw_pick_config(pick_config)
                    self._check_pick_points(item, pick_config)

if __name__ == "__main__":
    # rospy.init_node("aim_node")
    ac.init_node("aim_node")
    data_subscriber = DataSubscriber()
    picker = Picker("/workspace/config/height_map.npy", 1)
    ac.gripper(False)
    ret = ac.move_by_camera([ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)])
    # ret = ac.move_by_camera([ac.Point6D(0.2, -0.3, 0.7, np.pi, 0, np.pi/2)])
    # ret = ac.move_by_camera([ac.Point6D(-0.2, -0.3, 0.7, np.pi, 0, np.pi/2)])
    # ret = ac.move_by_camera([ac.Point6D(-0.4203, -0.3397, 0.955, 0, 0, 0)])
    # ret = ac.move_by_camera([ac.Point6D(0.4348, -0.3408, 0.952, 0, 0, 0)])
    # ret = ac.move_by_camera([ac.Point6D(0.4253, 0.2379, 0.9630, np.pi, 0, np.pi/2)])
    # ret = ac.move_by_camera([ac.Point6D(-0.3925, 0.2566, 0.9680, np.pi, 0, np.pi/2)])
    # loginfo(f"Move ret: {ret}")
    rospy.loginfo("Ready to pick")
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("detected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detected", 1280, 720)
    cv2.setMouseCallback("detected", picker.mouse_cb)
    while not rospy.is_shutdown():
        color, depth, boxes, circles, prompt, header, _camera_info = data_subscriber.wait_for_msg(waitkey=True)
        picker.color_image = color
        picker.depth_image = depth
        picker.camera_info = _camera_info
        picker.items_detected = boxes
        depth = normalize_depth(depth, 900, 1000)
        masked = color.copy()
        masked = DrawingUtils.draw_plates(masked, circles)
        for box in boxes:
            masked = DrawingUtils.draw_box_info(masked, box)
        for i, name in enumerate(prompt):
            cv2.putText(masked, name, (10, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("depth", depth)
        cv2.imshow("detected", masked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    