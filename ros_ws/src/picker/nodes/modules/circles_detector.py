#! /usr/bin/env python3

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np

from modules.classes import Circle
from modules.utils import DrawingUtils

class PlateDetector:
    colors = {
    "yellow": ((0, 100, 0), (30, 255, 255)),
    "blue": ((60, 45, 0), (120, 255, 255)),
    "green": ((30, 60, 0), (60, 255, 255)),
    }
    def __init__(self, min_radius=110,
                    max_radius=180,
                    min_solidity=0.5,
                    erode_iterations=5,
                    dilate_iterations=5
                    ):
        """Initializes the plate detector
        @param min_radius: The minimum radius of a plate
        @param max_radius: The maximum radius of a plate
        @param min_solidity: The minimum solidity of a plate
        @param erode_iterations: The number of iterations to erode the mask
        @param dilate_iterations: The number of iterations to dilate the mask
        """
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._min_solidity = min_solidity
        self._erode_iterations = erode_iterations
        self._dilate_iterations = dilate_iterations

    def _draw_circle_mask(self, image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
        """Draws a mask of the given circle
        @param image: The image to draw the mask on
        @param circle: The circle to draw
        @return: The mask
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
        return mask

    def detect(self, image: np.ndarray) -> List[Circle]:
        """Detects plates in the given image and returns a list of Circle objects
        @param image: The image color BGR image to detect plates in
        @return: A list of Circle objects
        """
        '''
        circle_dict = {
            'mask': mask,
            'pos': (x, y),
            'radius': radius,
            colors: {
                'color': solidity,
                ...
            },
            'max_color': {
                'color': name,
                'solidity': solidity
            }
        '''
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(grayscale, cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=30, param2=25, minRadius=self._min_radius, maxRadius=self._max_radius)
        circles = np.uint16(np.around(circles))
        circles = circles[0, :]
        
        # # draw circles
        # draw_image = image.copy()
        # for i in circles:
        #     cv2.circle(draw_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #     cv2.circle(draw_image, (i[0], i[1]), 2, (0, 0, 255), 3)
        # cv2.imshow("circles", draw_image)
        # cv2.waitKey(1)
        # filter circles
        color_masks = self._get_color_mask(image, self.colors.values())
        circles_dicts_list = [{'mask': self._draw_circle_mask(grayscale, circle), 
                               'pos': np.array((circle[0], circle[1]), dtype=np.int32),
                               'area': None,
                               'radius': circle[2], 
                               'colors': {}, 
                               'max_color': None} for circle in circles]
        for color_name, color_mask in zip(self.colors.keys(), color_masks):
            color_mask = cv2.erode(color_mask, np.ones((3, 3), dtype=np.uint8), iterations=self._erode_iterations)
            color_mask = cv2.dilate(color_mask, np.ones((3, 3), dtype=np.uint8), iterations=self._dilate_iterations)
            circles_dicts = []
            for circle in circles_dicts_list:
                # compute solidity
                intersection = cv2.bitwise_and(circle['mask'], color_mask)
                area =  cv2.countNonZero(circle['mask'])
                circle['colors'][color_name] = cv2.countNonZero(intersection) / area
                circle['area'] = area
        # select max color
        for circle in circles_dicts_list:
            max_color = max(circle['colors'], key=circle['colors'].get)
            solidity = circle['colors'][max_color]
            circle['max_color'] = {
                'color': max_color,
                'solidity': solidity
            }
        # filter circles
        circles_dicts_list = filter(lambda c: c['max_color']['solidity'] > self._min_solidity, circles_dicts_list)
        # create Circle objects
        circles_list = []
        for circle in circles_dicts_list:
            mask = circle['mask']
            mask = np.expand_dims(mask, axis=2)
            circles_list.append(Circle(name = "{} plate".format(circle['max_color']['color']),
                                            mask = mask,
                                            score = circle['max_color']['solidity'],
                                            pos = circle['pos'],
                                            area=circle['area'],
                                            radius = circle['radius'],
                                            additional_names=[]
                                            ))
                
        return circles_list
    
    def _get_color_mask(self, image, color_ranges):
        """Returns a generator of masks for the given image and color ranges
        @param image: The image to get the masks for
        @param color_ranges: A list of color ranges to get the masks for
        @return: A generator of masks
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for color_range in color_ranges:
            mask = cv2.inRange(hsv_image, *color_range)
            yield mask
