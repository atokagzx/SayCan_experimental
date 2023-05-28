#! /usr/bin/env python3

import numpy as np
from typing import List, Tuple, Dict, Iterable
from modules.classes import Circle, Box, Item
import cv2

class CroppedImage:
    def __init__(self, image, crop_box):
        '''
        @param image: the image to crop
        @param crop_box: the crop box, [x1, y1, x2, y2]
        '''
        self.image = image
        self._original_shape = image.shape
        self._crop_box = crop_box
        self._crop()

        # public attributes
        self.shape = self.image.shape
        self.crop_box = crop_box.copy()
    
    def _crop(self):
        x1, y1, x2, y2 = self._crop_box
        self.image = self.image[y1:y2, x1:x2]
    
    def uncropped(self, fill_value=[0, 0, 0]):
        '''
        Return the image with the cropped region filled with fill_value
        @param fill_value: the value to fill the cropped region with
        '''
        image = np.full(self._original_shape, fill_value, dtype=np.uint8)
        x1, y1, x2, y2 = self._crop_box
        image[y1:y2, x1:x2] = self.image
        return image
    
    def copy(self):
        return CroppedImage(self.image.copy(), self.crop_box.copy())
                            
    def __call__(self):
        return self.image
    
    def coords_transform(self, items: Iterable[Item]) -> Iterable[Item]:
        """Transforms the coordinates of the given items to the original image coordinates
        Mutates the items
        @param items: The items to transform
        @return: The transformed items
        """
        x1, y1, x2, y2 = self.crop_box
        for item in items:
            item.pos += np.array([x1, y1])
            new_mask = np.zeros((*self._original_shape[:2], 1), dtype=np.uint8)
            new_mask[y1:y2, x1:x2] = item.mask
            item.mask = new_mask
            if isinstance(item, Box):
                item.bbox += np.array([x1, y1, x1, y1])
                item.rotated_rect += np.array([x1, y1])
        return items


class DrawingUtils:

    @staticmethod
    def draw_plates(image: np.ndarray, plates: List[Circle]) -> np.ndarray:
        """Draws the given plates on the given image
        @param image: The image to draw the plates on, will be modified
        @param plates: The plates to draw
        @return: The image with the plates drawn on it
        """
        for plate in plates:
            # draw the circle and centroid on the image,
            mask = plate.mask
            color = np.random.uniform(0, 1, size=3)
            rgb_mask = np.array([mask.copy() for _ in range(3)])
            colored_mask = np.concatenate(rgb_mask, axis=-1) * color
            colored_mask = np.array(colored_mask, dtype=np.uint8)
            image = cv2.addWeighted(image, 1, colored_mask, 1, 0)
            color = tuple(map(lambda x: int(x * 255), color))
            black = (0, 0, 0)
            cv2.circle(image, (int(plate.pos[0]), int(plate.pos[1])), int(plate.radius),
                        (0, 255, 255), 2)
            cv2.circle(image, (int(plate.pos[0]), int(plate.pos[1])), 5, color=color, thickness=-1)
            cv2.putText(image, plate.name, (int(plate.pos[0]), int(plate.pos[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(image, f"pos: {plate.pos}", (int(plate.pos[0]), int(plate.pos[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
            cv2.putText(image, f"radius: {plate.radius}", (int(plate.pos[0]), int(plate.pos[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
            cv2.putText(image, f"solidity: {plate.score:.2f}", (int(plate.pos[0]), int(plate.pos[1]) + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
            cv2.putText(image, f"area: {plate.area}", (int(plate.pos[0]), int(plate.pos[1]) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)   
        return image
    
    @staticmethod
    def draw_box_info(image: np.ndarray, item: Box, color: Tuple[int, int, int] = (255, 0, 255)):
        image = DrawingUtils.draw_box_mask(image, item.mask)
        black = (0, 0, 0)
        cv2.circle(image, item.pos, 5, color, -1)
        cv2.rectangle(image, item.bbox[:2], item.bbox[2:], color, 2)
        cv2.drawContours(image, [item.rotated_rect], 0, (0, 255, 0), 2)
        cv2.putText(image, item.name, (item.bbox[0], item.bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(image, f"score: {item.score:.2f}", (item.bbox[0], item.bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)
        cv2.putText(image, f"area: {item.area}", (item.bbox[0], item.bbox[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)
        cv2.putText(image, f"rr_area: {item.rotated_rect_area}", (item.bbox[0], item.bbox[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)
        cv2.putText(image, f"h_l: {item.length} x {item.width}", (item.bbox[0], item.bbox[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)
        cv2.putText(image, f"angle: {int(np.rad2deg(item.angle))}", (item.bbox[0], item.bbox[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)
        cv2.putText(image, f"pos: {item.pos}", (item.bbox[0], item.bbox[1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)
        # cv2.putText(image, f"phr: {item.phrase}", (item.bbox[0], item.bbox[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)

        # draw line from center to bbox at angle
        angle = item.angle
        x1 = item.pos[0]
        y1 = item.pos[1]
        x2 = x1 + int(item.width * np.cos(angle))
        y2 = y1 + int(item.width * np.sin(angle))
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        return image
    
    @staticmethod
    def draw_box_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = None):
        if color is None:
            color = np.random.uniform(0, 1, size=3)
        rgb_mask = np.array([mask.copy() for _ in range(3)])
        colored_mask = np.concatenate(rgb_mask, axis=-1) * color
        colored_mask = np.array(colored_mask, dtype=np.uint8)
        image = cv2.addWeighted(image, 1, colored_mask, 1, 0)
        return image

def normalize_depth(depth_map, low, high):
    depth_map = depth_map.astype(float)
    undetectable_mask = depth_map == 0
    depth_map = (depth_map - low) / (high - low)
    # 10, 11 and 12 are magic numbers for clipping
    depth_map[depth_map > 1] = 10
    depth_map[depth_map < 0] = 11
    higher_mask = depth_map == 10
    lower_mask = depth_map == 11
    depth_map = np.clip(depth_map, 0, 1)
    depth_map = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    depth_map[higher_mask] = (255, 255, 255)
    depth_map[lower_mask] = (100, 100, 100)
    depth_map[undetectable_mask] = (0, 0, 0)
    return depth_map
