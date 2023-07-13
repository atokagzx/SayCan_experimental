#! /usr/bin/env python3

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np

from modules.classes import Circle
from modules.utils import DrawingUtils
from modules.circles_detector import PlateDetector

if __name__ == "__main__":
    ac.init_node("alpaca_circles_detector")
    plate_detector = PlateDetector()
    while not ac.is_shutdown():
        if ac.is_image_ready():
            color, depth_map, _camera_info = ac.pop_image()
            plates = plate_detector.detect(color)
            color = DrawingUtils.draw_plates(color, plates)
            cv2.imshow("color", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            sleep(0.01)