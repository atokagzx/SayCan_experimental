from picker.msg import Prompt, BoxArray, Box, CircleArray, Circle
from cv_bridge import CvBridge, CvBridgeError
import std_msgs.msg as std_msgs
from sensor_msgs.msg import Image, CameraInfo
import rospy
import cv2
import numpy as np

class DataSubscriber:
    def __init__(self):
        self._bridge = CvBridge()
        self._camera_info = None
        self._msg_buffer = {
            "timestamp": rospy.Time(0),
            "color": None,
            "depth": None,
            "boxes": None,
            "circles": None
        }
        self._new_msg = False
        self._subscribe()

    def _subscribe(self):
        rospy.Subscriber("/alpaca/detector/camera/color", Image, self._msg_cb, callback_args={"type": "color"}, queue_size=2)
        rospy.Subscriber("/alpaca/detector/camera/aligned_depth", Image, self._msg_cb, callback_args={"type": "depth"}, queue_size=2)
        rospy.Subscriber("/alpaca/detector/camera/camera_info", CameraInfo, self._cam_info_cb)
        rospy.Subscriber("/alpaca/detector/boxes", BoxArray, self._msg_cb, callback_args={"type": "boxes"}, queue_size=2)
        rospy.Subscriber("/alpaca/detector/circles", CircleArray, self._msg_cb, callback_args={"type": "circles"}, queue_size=2)

    def _cam_info_cb(self, msg):
        self._camera_info = msg

    def _msg_cb(self, msg, args):
        timestamp = msg.header.stamp
        if timestamp > self._msg_buffer["timestamp"]:
            for key in self._msg_buffer.keys():
                self._msg_buffer[key] = None
            self._msg_buffer["timestamp"] = timestamp
        self._msg_buffer[args["type"]] = msg
        for key in self._msg_buffer.keys():
            if self._msg_buffer[key] is None:
                self._new_msg = False
                return
        self._new_msg = True
        
    def wait_for_msg(self, waitkey=False):
        while not self._new_msg or self._camera_info is None:
            if rospy.is_shutdown():
                raise rospy.ROSInterruptException("ROS shutdown")
            if waitkey:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt("Keyboard interrupt")
            else:
                rospy.sleep(0.01)
        self._new_msg = False
        color, depth, boxes, circles, prompt, header = self._extract_data()
        return color, depth, boxes, circles, prompt, header, self._camera_info

    def _extract_data(self):
        color = self._msg_buffer["color"]
        depth = self._msg_buffer["depth"]
        boxes = self._msg_buffer["boxes"]
        circles = self._msg_buffer["circles"]
        header = color.header

        color = self._bridge.imgmsg_to_cv2(color, desired_encoding="passthrough")
        depth = self._bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        boxes, prompt = self._extract_boxes(boxes)
        circles = self._extract_circles(circles)
        prompt = prompt.names
        return color, depth, boxes, circles, prompt, header
    
    def _extract_boxes(self, boxes):
        prompt = boxes.prompt
        boxes = boxes.boxes
        boxes_list = []
        for box in boxes:
            mask = self._bridge.imgmsg_to_cv2(box.mask, desired_encoding="passthrough")
            mask = np.expand_dims(mask, axis=2)
            boxes_list.append(Box(name=box.name,
                            mask=mask,
                            score=box.score,
                            pos=np.array(box.pos),
                            area=box.area,
                            additional_names=list(box.additional_names),
                            bbox=np.array(box.bbox),
                            bbox_area=box.bbox_area,
                            angle=box.angle,
                            width=box.width,
                            length=box.length,
                            rotated_rect=np.array(box.rotated_rect).reshape((4, 2)),
                            rotated_rect_area=box.rotated_rect_area))
        return boxes_list, prompt
    
    def _extract_circles(self, circles):
        circles_list = []
        for circle in circles.circles:
            mask = self._bridge.imgmsg_to_cv2(circle.mask, desired_encoding="passthrough")
            mask = np.expand_dims(mask, axis=2)
            circles_list.append(Circle(name=circle.name,
                            mask=mask,
                            score=circle.score,
                            pos=np.array(circle.pos),
                            area=circle.area,
                            additional_names=list(circle.additional_names),
                            radius=circle.radius))
        return circles_list