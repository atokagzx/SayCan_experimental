#! /usr/bin/env python3

from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
from PIL import Image
import os, sys
from collections import namedtuple
import rospy
import matplotlib.pyplot as plt
from modules.gsam_detector import GSAMDetector
from modules.plates_detector import PlateDetector
from modules.utils import CroppedImage, DrawingUtils
from threading import Thread
from picker.srv import PickConfig, PickConfigResponse, PickConfigRequest
from modules.ros_utils import DataSubscriber

class ClickProcessor:
    srv_name = "/alpaca/get_pick_config"
    def __init__(self):
        self._pick_thread = None
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self._image_header = None
        self.boxes = []
        rospy.wait_for_service(self.srv_name, timeout=5)
        self._pick_config_srv = rospy.ServiceProxy(self.srv_name, PickConfig)
        
    def set_data(self, color_image, depth, header, camera_info, boxes):
        self.color_image = color_image
        self.depth_image = depth
        self._image_header = header
        self.camera_info = camera_info
        self.boxes = boxes

    def pick(self, item_name: str, item_pos: Tuple[int, int]):
        if self._pick_thread is not None:
            is_alive = self._pick_thread.is_alive()
            if is_alive:
                raise RuntimeError("pick thread is already running")
        self._pick_thread = Thread(target=self._move_gripper_thread, daemon=True, args=(item_name, item_pos))
        self._pick_thread.start()

    def _move_gripper_thread(self, item_name: str, item_pos: Tuple[int, int]):
        timer = rospy.Rate(0.5)
        for i in range(5):
            if i > 0:
                rospy.loginfo(f"retrying {i + 1}...")
            request = PickConfigRequest(
                header=rospy.Header(),
                object_name=item_name,
                pos=item_pos,
            )
            try:
                pick_config = self._pick_config_srv(request)
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                rospy.signal_shutdown(f"Service call failed: {e}")
            if pick_config.success:
                self._move_gripper((pick_config.object_position.x,
                                        pick_config.object_position.y,
                                        pick_config.object_position.z), 
                                pick_config.object_orientation, 
                                pick_config.object_width)
                break
            else:
                rospy.logwarn(f"picking is not possible, because: {pick_config.reason}")
                timer.sleep()
        
    def _move_gripper(self, object_pos: Tuple[float, float, float], object_orientation: float, object_width: float):
        rospy.loginfo(f"object pos: {object_pos}\n\torientation: {object_orientation}\n\twidth: {object_width}")

        pre_grasp = [
            ac.Point6D(object_pos[0], object_pos[1], object_pos[2] - 0.2, np.pi, 0, object_orientation),
            ac.Point6D(object_pos[0], object_pos[1], object_pos[2] - 0.008, np.pi, 0, object_orientation),
        ]
        grasp = [
            ac.Point6D(object_pos[0], object_pos[1], object_pos[2] - 0.2, np.pi, 0, object_orientation),
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

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sorted_items = sorted(self.boxes, key=lambda item: np.linalg.norm(item.pos - np.array([x, y])))
            if len(sorted_items) > 0:
                if np.linalg.norm(sorted_items[0].pos - np.array([x, y])) < 100:
                    item = sorted_items[0]
                    print(f"item {item.name} selected")
                    try:
                        self.pick(item.name, item.pos)
                    except RuntimeError as e:
                        rospy.logwarn(e)

if __name__ == "__main__":
    ac.init_node("alpaca_gsam_example")
    ret = ac.move_by_camera([ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)])
    if not ret:
        rospy.logerr("Failed to move to start position")
        exit(1)
    picker = ClickProcessor()
    data_subscriber = DataSubscriber()
    cv2.namedWindow("click2pick", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("click2pick", 1280, 720)
    cv2.setMouseCallback("click2pick", picker.mouse_cb)
    while not rospy.is_shutdown():
        color, depth, boxes, circles, prompt, header, _camera_info = data_subscriber.wait_for_msg(waitkey=True)
        picker.set_data(color_image=color, 
                        depth=depth, 
                        header=header,
                        camera_info=_camera_info,
                        boxes=boxes)
        masked = color.copy()
        masked = DrawingUtils.draw_plates(masked, circles)
        for box in boxes:
            masked = DrawingUtils.draw_box_info(masked, box)
        for i, name in enumerate(prompt):
            cv2.putText(masked, name, (10, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("click2pick", masked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    