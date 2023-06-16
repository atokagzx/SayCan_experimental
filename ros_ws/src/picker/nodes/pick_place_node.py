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
from picker.srv import PickPlace, PickPlaceResponse, PickPlaceRequest
from modules.ros_utils import DataSubscriber

def move_gripper_wrapper(positions: List[ac.Point6D]):
    ret = ac.move_by_camera(positions)
    if not ret:
        raise ValueError("move_gripper failed")
    
def set_gripper_wrapper(state: bool):
    ret = ac.gripper(state)
    if not ret:
        raise ValueError("set_gripper failed")

class PickPlaceSkill:
    pick_box_srv_name = "/alpaca/get_pick_config_box"
    pick_circle_srv_name = "/alpaca/get_pick_config_circle"
    place_config_srv_name = "/alpaca/get_place_config"
    def __init__(self):
        self._boxes = None
        self._circles = None
        for service in [self.pick_box_srv_name, self.pick_circle_srv_name]:
            rospy.wait_for_service(service, timeout=5)
        self._pick_box_config_srv = rospy.ServiceProxy(self.pick_box_srv_name, PickConfig)
        self._pick_circle_config_srv = rospy.ServiceProxy(self.pick_circle_srv_name, PickConfig)
        self._place_config_srv = rospy.ServiceProxy(self.place_config_srv_name, PickConfig)  
        # advertise service
        self._pick_srv = rospy.Service("/alpaca/pick_place", PickPlace, self._pick_place_cb)
        
    def _pick_place_cb(self, req: PickPlaceRequest) -> PickPlaceResponse:
        
        pick_req = {
            "object_name": req.pick_object_name,
            "object_type": req.pick_object_type,
            "object_pos": req.pick_object_pos,
        }
        place_req = {
            "object_name": req.place_object_name,
            "object_type": req.place_object_type,
            "object_pos": req.place_object_pos,
        }

        success, pick_place_config = self._get_pick_place_config(pick_req, place_req)
        if not success:
            # pick_place_config is actually response message
            return pick_place_config
        
        success, response = self._pick_place(pick_place_config["pick"], pick_place_config["place"])
        if not success:
            return response
        return PickPlaceResponse(success=True, reason="")

    def _pick_place(self, pick_config, place_config):
        pick_position = pick_config.object_position
        pick_position = [pick_position.x, pick_position.y, pick_position.z]
        place_position = place_config.object_position
        place_position = [place_position.x, place_position.y, place_position.z]
        pick_orientation = pick_config.object_orientation
        place_orientation = place_config.object_orientation
        # place_position[2] -= 0.06

        try:
            # open gripper
            set_gripper_wrapper(False)
            # move to start position
            move_gripper_wrapper([ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2),
                                ac.Point6D(pick_position[0], pick_position[1], pick_position[2] - 0.2, np.pi, 0, pick_orientation),
                                ac.Point6D(pick_position[0], pick_position[1], pick_position[2] - 0.0085, np.pi, 0, pick_orientation)])
            # close gripper
            set_gripper_wrapper(True)            
            move_gripper_wrapper([ac.Point6D(pick_position[0], pick_position[1], pick_position[2] - 0.2, np.pi, 0, pick_orientation),
                                ac.Point6D(place_position[0], place_position[1], place_position[2] - 0.25, np.pi, 0, place_orientation),
                                ac.Point6D(place_position[0], place_position[1], place_position[2] - 0.025, np.pi, 0, place_orientation)])
            # open gripper
            set_gripper_wrapper(False)
            # move to start position
            move_gripper_wrapper([ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)])
        except ValueError as e:
            # open gripper
            ac.gripper(False)
            return False, PickPlaceResponse(success=False, reason=f"failed to execute pick_place skill: {e}")
        else:
            return True, PickPlaceResponse(success=True)
            
    def _get_pick_place_config(self, pick_req, place_req):
        config = {"pick": None, "place": None}
        for config_name, req in zip(["pick", "place"], [pick_req, place_req]):
            timer = rospy.Rate(0.5)
            srv = None
            # for pick we use pick_box_config_srv or pick_circle_config_srv, for place we use place_config_srv for both
            if config_name == "place":
                srv = self._place_config_srv
            elif req["object_type"].lower() == "box":
                srv = self._pick_box_config_srv
            elif req["object_type"].lower() == "circle":
                srv = self._pick_circle_config_srv
            else:
                raise Exception(f"Unknown object type {req['object_type']}")
            # create request, passing object name and it's position in pixel coordinates
            request = PickConfigRequest(
                    header=rospy.Header(),
                    object_name=req["object_name"],
                    pos=req["object_pos"],
                )
            # try to get pick/place config 5 times with 0.5s delay between attempts if it fails
            for attempt in range(5):
                if attempt > 0:
                    rospy.loginfo(f"retrying {attempt + 1}...")
                try:
                    response = srv(request)
                except rospy.ServiceException as e:
                    rospy.logerr(f"service call failed: {e}")
                    rospy.signal_shutdown(f"service call failed: {e}")
                else:
                    if response.success:
                        config[config_name] = response
                        break
                    else:
                        rospy.logwarn(f"pick_place is not possible, because: {response.reason}")
                        timer.sleep()
            else:
                rospy.logerr(f"failed to get pick/place config for {req['object_name']}, because: {response.reason}")
                response = PickPlaceResponse(
                    success=False,
                    reason=f"failed to get pick/place config for {req['object_name']}, because: {response.reason}"
                )
                return False, response
        return True, config


                    
                    
    
if __name__ == "__main__":
    ac.init_node("alpaca_pick_place_skill_node")
    ret = ac.move_by_camera([ac.Point6D(0.0, -0.3, 0.7, np.pi, 0, np.pi/2)])
    if not ret:
        rospy.logerr("Failed to move to start position")
        exit(1)
    skill = PickPlaceSkill()
    rospy.spin()