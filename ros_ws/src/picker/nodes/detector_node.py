#! /usr/bin/env python3

from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.gsam_detector import GSAMDetector
from modules.plates_detector import PlateDetector
from modules.utils import CroppedImage, DrawingUtils
import rospy
from picker.msg import Prompt, BoxArray, Box, CircleArray, Circle
from cv_bridge import CvBridge, CvBridgeError
import std_msgs.msg as std_msgs
from sensor_msgs.msg import Image, CameraInfo

def set_names_to_detect(msg):
    global names_to_detect
    rospy.loginfo(f"names changed from: {names_to_detect.names}\nto: {msg.names}\nheader: {msg.header}")
    names_to_detect = msg

def publish_detected_boxes(items, header, prompt):
    global image_bridge, boxes_publisher
    boxes_list = []
    for item in items:
        ros_image = image_bridge.cv2_to_imgmsg(item.mask, encoding="passthrough")
        boxes_list.append(Box(
            name=item.name,
            mask=ros_image,
            score=item.score,
            pos=item.pos.tolist(),
            area=item.area,
            additional_names=item.additional_names,
            bbox=item.bbox.tolist(),
            bbox_area=item.bbox_area,
            angle=item.angle,
            width=item.width,
            length=item.length,
            rotated_rect=item.rotated_rect.flatten().tolist(),
            rotated_rect_area=item.rotated_rect_area
        ))
    boxes_no_mask_list = []
    for box in boxes_list:
        boxes_no_mask_list.append(Box(
            name=box.name,
            score=box.score,
            pos=box.pos,
            area=box.area,
            additional_names=box.additional_names,
            bbox=box.bbox,
            bbox_area=box.bbox_area,
            angle=box.angle,
            width=box.width,
            length=box.length,
            rotated_rect=box.rotated_rect,
            rotated_rect_area=box.rotated_rect_area
        ))
    boxes_publisher.publish(BoxArray(header=rospy.Header(
                                            stamp=header.stamp,
                                            frame_id="gsam_node",
                                        ), 
                                    prompt=prompt,
                                    boxes=boxes_list))
    boxes_no_mask_publisher.publish(BoxArray(header=rospy.Header(
                                            stamp=header.stamp,
                                            frame_id="gsam_node",
                                        ),
                                    prompt=prompt,
                                    boxes=boxes_no_mask_list))
    

def publish_detected_circles(items, header):
    global image_bridge, circles_publisher
    circles_list = []
    for item in items:
        ros_image = image_bridge.cv2_to_imgmsg(item.mask, encoding="passthrough")
        circles_list.append(Circle(
            name=item.name,
            mask=ros_image,
            score=item.score,
            pos=item.pos.tolist(),
            area=item.area,
            additional_names=item.additional_names,
            radius=item.radius
        ))
    circles_no_mask_list = []
    for circle in circles_list:
        circles_no_mask_list.append(Circle(
            name=circle.name,
            score=circle.score,
            pos=circle.pos,
            area=circle.area,
            additional_names=circle.additional_names,
            radius=circle.radius
        ))
    circles_publisher.publish(CircleArray(header=rospy.Header(
                                            stamp=header.stamp,
                                            frame_id="gsam_node",
                                        ), 
                                    circles=circles_list))
    circles_no_mask_publisher.publish(CircleArray(header=rospy.Header(
                                            stamp=header.stamp, 
                                            frame_id="gsam_node",
                                        ),
                                    circles=circles_no_mask_list))
    
def publish_images(color, depth_map, _camera_info, header):
    global image_bridge, color_publisher, depth_publisher, camera_info_publisher
    image_depth = image_bridge.cv2_to_imgmsg(depth_map, encoding="passthrough")
    image_depth.header = rospy.Header(
        stamp=header.stamp,
        frame_id="gsam_node",
    )
    image_color = image_bridge.cv2_to_imgmsg(color, encoding="passthrough")
    image_color.header = rospy.Header(
        stamp=header.stamp,
        frame_id="gsam_node",
    )
    color_publisher.publish(image_color)
    depth_publisher.publish(image_depth)
    camera_info_publisher.publish(CameraInfo(
        header=rospy.Header(
            stamp=header.stamp,
            frame_id="gsam_node",
        ),
        height=_camera_info.height,
        width=_camera_info.width,
        distortion_model=_camera_info.distortion_model,
        D=_camera_info.D,
        K=_camera_info.K,
        R=_camera_info.R,
        P=_camera_info.P,
        binning_x=_camera_info.binning_x,
        binning_y=_camera_info.binning_y,
        roi=_camera_info.roi
    ))

def filter_outer_boxes(boxes, bounds=((350, 20),
                                        (1600, 20),
                                        (1600, 850),
                                        (350, 850))):
    if len(boxes) == 0:
        return boxes
    mask = np.zeros((boxes[0].mask.shape[0], boxes[0].mask.shape[1]), dtype=np.uint8)
    # draw the outer bounds as trapazoid
    cv2.fillPoly(mask, np.array([bounds], dtype=np.int32), 255)
    mask = cv2.bitwise_not(mask)
    for box in boxes:
        intersection = cv2.bitwise_and(box.mask, box.mask, mask=mask)
        if np.sum(intersection) > 0:
            box.score = 0
    filtered_boxes = list(filter(lambda box: box.score > 0, boxes))
    return filtered_boxes

def filter_out_low_score_fish(boxes, threshold=0.4):
    """Filter out boxes that are below the threshold
    @param boxes: list of boxes
    @param threshold: threshold to filter out boxes
    @return: list of boxes that are above the threshold
    """
    filtered_boxes = list(filter(lambda box: box.score > threshold or box.name != "fish", boxes))
    return filtered_boxes

def filter_plates_from_boxes(boxes, plates):
    """Filter out boxes that are already detected as plates
    @param boxes: list of boxes
    @param plates: list of plates
    @return: list of boxes that are not plates
    """
    for box in boxes:
        for plate in plates:
            if np.linalg.norm(box.pos - plate.pos) < 50:
                if 0.8 < box.area / plate.area < 1.2:
                    box.score = 0
    filtered_boxes = list(filter(lambda box: box.score > 0, boxes))
    return filtered_boxes

def compose_blocks(boxes):
    sinonim_names = {
        "fish": [],
        "block": ["square", "rectangle", "cube", "box", "small block"],
    }
    for box in boxes:
        for original_name, sinonim_list in sinonim_names.items():
            for sinonim in sinonim_list:
                if sinonim in box.name:
                    # replace the name with the original name
                    box.name = box.name.replace(sinonim, original_name)
    return boxes

if __name__ == "__main__":
    ac.init_node("detector_node")
    names_to_detect_list = list(rospy.get_param("/alpaca/names_to_detect", "").split(";"))
    debug_mode = rospy.get_param("~debug", False)
    names_to_detect = Prompt(header=rospy.Header(
                                stamp=rospy.Time.now(),
                                frame_id="gsam_node",
                            ), 
                        names=names_to_detect_list)
    detecting_names = names_to_detect
    image_bridge = CvBridge()
    boxes_detector = GSAMDetector()
    plate_detector = PlateDetector()
    if debug_mode:
        cv2.namedWindow(rospy.get_name(), cv2.WINDOW_NORMAL)
    rospy.Subscriber("/alpaca/names_to_detect", Prompt, set_names_to_detect)
    boxes_publisher = rospy.Publisher("/alpaca/detector/boxes", BoxArray, queue_size=1, latch=False)
    circles_publisher = rospy.Publisher("/alpaca/detector/circles", CircleArray, queue_size=1, latch=False)
    boxes_no_mask_publisher = rospy.Publisher("/alpaca/detector/boxes_no_mask", BoxArray, queue_size=1, latch=False)
    circles_no_mask_publisher = rospy.Publisher("/alpaca/detector/circles_no_mask", CircleArray, queue_size=1, latch=False)
    color_publisher = rospy.Publisher("/alpaca/detector/camera/color", Image, queue_size=1, latch=False)
    depth_publisher = rospy.Publisher("/alpaca/detector/camera/aligned_depth", Image, queue_size=1, latch=False)
    camera_info_publisher = rospy.Publisher("/alpaca/detector/camera/camera_info", CameraInfo, queue_size=1, latch=False)
    masked_frame_publisher = rospy.Publisher("/alpaca/detector/camera/detected", Image, queue_size=1, latch=False)
    while not ac.is_shutdown():
        if ac.is_image_ready():
            detecting_names = names_to_detect
            color, depth_map, _camera_info, header = ac.pop_image(add_header=True)
            if len(detecting_names.names):
                crop = CroppedImage(color, [330, 30, 1600, 880])
                # if debug_mode:
                #     cv2.imshow("crop", crop())
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                boxes = boxes_detector.get_items(crop(), detecting_names.names)
                boxes = crop.coords_transform(boxes)
                circles = plate_detector.detect(color)
                boxes = filter_plates_from_boxes(boxes, circles)
                boxes = GSAMDetector.filter_same_items(boxes)
                boxes = compose_blocks(boxes)
                boxes = filter_out_low_score_fish(boxes)
                boxes = filter_outer_boxes(boxes)
                publish_detected_boxes(boxes, header, detecting_names)
                publish_detected_circles(circles, header)
                publish_images(color, depth_map, _camera_info, header)
                if debug_mode or masked_frame_publisher.get_num_connections():
                    masked_frame = color.copy()
                    masked_frame = DrawingUtils.draw_plates(masked_frame, circles)
                    for item in boxes:
                        masked_frame = DrawingUtils.draw_box_info(masked_frame, item)
                    if debug_mode:
                        cv2.imshow(rospy.get_name(), masked_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    if masked_frame_publisher.get_num_connections() > 0:
                        image_to_publish = image_bridge.cv2_to_imgmsg(masked_frame, encoding="passthrough")
                        image_to_publish.header = rospy.Header(
                            stamp=header.stamp,
                            frame_id="gsam_node",
                        )
                        masked_frame_publisher.publish(image_to_publish)
            else:
                rospy.loginfo_throttle(5, "No names to detect")
                if debug_mode:
                    cv2.imshow(rospy.get_name(), color)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        else:
            sleep(0.01)