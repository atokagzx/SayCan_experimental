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
from picker.srv import PickConfig, PickConfigResponse
from geometry_msgs.msg import Point

gripper_footprint_image = None

class PickerConfig:
    def __init__(self, debug_mode=False):
        # self._height_map = np.load(height_map_path)
        # self._plain_offset = plain_offset
        self._gripper_footprint_length = 200 # [pixels]
        self._gripper_footprint_width = 40 # [pixels]
        self._gripper_height = 0.06 # [mm]
        self._fingers_offset = 0 # [pixels]
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self._image_header = None
        self.boxes = []
        self._debug_mode = debug_mode

    def set_data(self, color_image, depth, header, camera_info, boxes):
        self.color_image = color_image
        self.depth_image = depth
        self._image_header = header
        self.camera_info = camera_info
        self.boxes = boxes

    def _get_pick_config(self, item):
        assert self.color_image is not None and self.depth_image is not None and self.camera_info is not None
        mask = item.mask
        points = np.argwhere(mask)[:,:2] # [y, x]
        points = points[:, ::-1] # [x, y]
        points = points.astype(np.int32)
        try:
            real_points = np.array(ac.to_real_points(points, self.depth_image, self.camera_info), dtype=np.float32)
        except ValueError as e:
            rospy.logerr(f"error in to_real_points: {e}")
            return  {
                "pick_points": [],
                "mean_height": 0,
                "top_height": 0,
            }
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
    
    # def _draw_pick_config(self, pick_config):
    #     masked = self.color_image.copy()
    #     pick_points_1 = pick_config["edge_points"][0]
    #     pick_points_2 = pick_config["edge_points"][1]
    #     for pick_point in [pick_points_1, pick_points_2]:
    #         cv2.line(masked, tuple(pick_point["pos"][0]), tuple(pick_point["pos"][1]), (0, 0, 255), 2)
    #         cv2.circle(masked, tuple(pick_point["pos"][0]), 5, (0, 0, 255), -1)
    #         cv2.circle(masked, tuple(pick_point["pos"][1]), 5, (0, 0, 255), -1)
    #     # cv2.namedWindow("masked", cv2.WINDOW_NORMAL)
    #     # cv2.imshow("masked", masked)

    def _get_fingers_collision(self, item, angle):
        # deproject gripper footprint on height map
        grip_points_pixels = [self._get_point_by_angle_and_distance(angle, self._gripper_footprint_length/2, item.pos),
                                self._get_point_by_angle_and_distance(angle, -self._gripper_footprint_length/2, item.pos)]
        footprint_mask = np.zeros([self.color_image.shape[0], self.color_image.shape[1]], dtype=np.uint8)
        item_mask = item.mask.copy()
        kernel = np.ones((10,10),np.uint8)
        item_mask = cv2.dilate(item_mask,kernel,iterations = 3)
        footprint_mask = cv2.line(footprint_mask, tuple(grip_points_pixels[0]), tuple(grip_points_pixels[1]), 255, self._gripper_footprint_width)
        footprint_mask = cv2.bitwise_and(footprint_mask, footprint_mask, mask=~item_mask)
        points_to_check_collision = np.argwhere(footprint_mask)
        # swap x and y
        points_to_check_collision = np.asarray([points_to_check_collision[:, 1], points_to_check_collision[:, 0]]).T
        try:
            grip_points = np.asarray(ac.to_real_points(points_to_check_collision, self.depth_image, self.camera_info))[:, 2]
        except ValueError as e:
            rospy.logerr(f"error in to_real_points: {e}")
            return None
        grip_points = grip_points[grip_points != 0]
        lowest_point, lowest_point_index = np.min(grip_points), np.argmin(grip_points)
        # draw on image
        masked_color = self.color_image.copy()
        masked_color[footprint_mask == 255] = 0
        if self._debug_mode:
            global gripper_footprint_image
            gripper_footprint_image[footprint_mask == 255] = 0
            cv2.circle(gripper_footprint_image, tuple(points_to_check_collision[lowest_point_index]), 10, (0, 0, 255), -1)
        return lowest_point

    def _check_pick_points(self, item, pick_config, on_plate=False):
        # max_pick_height_center = self._height_map[item.pos[1], item.pos[0]] - self._plain_offset
        try:
            real_center = np.array(ac.to_real_points([item.pos], self.depth_image, self.camera_info)[0])
        except ValueError as e:
            rospy.logerr(f"error in to_real_points: {e}")
            return None
        body_collision = pick_config['top_height'] + self._gripper_height
        preferred_height = pick_config['mean_height'] + 0.012
        if self._debug_mode:
            global gripper_footprint_image
            gripper_footprint_image = self.color_image.copy()
        for pick_point in pick_config['edge_points']:
            angle = pick_point['angle']
            points = pick_point['pos']
            try:
                real_edge_points = np.asarray(ac.to_real_points(points[:, :2], self.depth_image, self.camera_info))
            except ValueError as e:
                rospy.logerr(f"error in to_real_points: {e}")
                continue
            # distance between pick point
            distance = np.linalg.norm(real_edge_points[0] - real_edge_points[1])
            if distance > 0.12:
                continue
            obstacles_height = self._get_fingers_collision(item, angle)
            if obstacles_height is None:
                continue
            max_pick_height = np.min([body_collision, obstacles_height])
            rospy.loginfo(f"height points:\nbody_collision: {body_collision}\nobstacles_height: {obstacles_height}\nmax_pick_height: {max_pick_height}\npreferred_height: {preferred_height}")
            if max_pick_height < preferred_height:
                continue
            rospy.loginfo(f"pick height: {preferred_height}")
            pick_pose = np.array([real_center[0], real_center[1], preferred_height])
            return pick_pose, angle, distance
        else:
            return None
        
    def pick_config_cb(self, req):
        header = req.header
        # wait for new image
        time = rospy.Time.now()
        while True:
            if rospy.Time.now() - time > rospy.Duration(5):
                return PickConfigResponse(
                    success=False,
                    reason=f'timeout waiting for new image: last image time: {self._image_header.stamp}, looking for: {header.stamp}'
                )
            if self._image_header is None:
                sleep(0.1)
            else:
                if self._image_header.stamp >= header.stamp:
                    break
                else:
                    sleep(0.1)
        name = req.object_name
        pos_on_image = req.pos
        filtered_items = (item for item in self.boxes if item.name == name)
        sorted_items = sorted(filtered_items, key=lambda item: np.linalg.norm(item.pos - np.array(pos_on_image)))
        if len(sorted_items) == 0:
            return PickConfigResponse(
                success=False,
                reason=f'item "{name}" not found'
            )
        item = sorted_items[0]
        distance = np.linalg.norm(item.pos - np.array(pos_on_image))
        if distance > 50:
            return PickConfigResponse(
                success=False,
                reason=f'item "{name}" position is too far from the selected point: "{pos_on_image}"/"{item.pos} distance: "{distance}"'
            )
        pick_config = self._get_pick_config(item)
        pick_config = self._check_pick_points(item, pick_config)
        if pick_config is None:
            return PickConfigResponse(
                success=False,
                reason=f'item "{name}" cannot be picked'
            )
        object_position, object_orientation, width = pick_config
        object_orientation = -object_orientation + np.pi/2
        return PickConfigResponse(
            success=True,
            object_position=Point(*object_position),
            object_orientation=object_orientation,
            object_width=width
        )

if __name__ == "__main__":
    rospy.init_node("pick_config_node")
    debug_mode = rospy.get_param("~debug", False)
    data_subscriber = DataSubscriber()
    picker = PickerConfig(debug_mode=debug_mode)
    rospy.Service("/alpaca/get_pick_config", PickConfig, picker.pick_config_cb)
    # if debug_mode:
    #     cv2.namedWindow("detected", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("detected", 1280, 720)
    # cv2.setMouseCallback("detected", picker.mouse_cb)
    while not rospy.is_shutdown():
        color, depth, boxes, circles, prompt, header, _camera_info = data_subscriber.wait_for_msg(waitkey=True)
        picker.set_data(color_image=color, 
                        depth=depth, 
                        header=header,
                        camera_info=_camera_info,
                        boxes=boxes)
        if debug_mode:
            # masked = color.copy()
            # masked = DrawingUtils.draw_plates(masked, circles)
            # for box in boxes:
            #     masked = DrawingUtils.draw_box_info(masked, box)
            # for i, name in enumerate(prompt):
            #     cv2.putText(masked, name, (10, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imshow("detected", masked)
            if not gripper_footprint_image is None:
                cv2.namedWindow("gripper_footprint", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("gripper_footprint", 1280, 720)
                cv2.imshow("gripper_footprint", gripper_footprint_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        