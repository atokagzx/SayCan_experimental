#! /usr/bin/env python3

import rospy
from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf_conversions

if __name__ == "__main__":
    rospy.init_node("move_robot_node")
    move_group = moveit_commander.MoveGroupCommander("manipulator")
    time = rospy.Time.now()
    path = geometry_msgs.msg.PoseArray()
    path.header.stamp = time
    path.header.frame_id = "base_link"
    path.poses = []
    # path.poses.append(geometry_msgs.msg.Pose(
    #     position=geometry_msgs.msg.Point(x=0.5, y=0.0, z=0.5),
    #     orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    # ))
    print(move_group.get_current_pose())
    print("Angle: ", np.rad2deg(tf_conversions.transformations.euler_from_quaternion([
        move_group.get_current_pose().pose.orientation.x,
        move_group.get_current_pose().pose.orientation.y,
        move_group.get_current_pose().pose.orientation.z,
        move_group.get_current_pose().pose.orientation.w,
    ])))
    # exit()
    path.poses.append(geometry_msgs.msg.Pose(
        position=move_group.get_current_pose().pose.position,
        orientation=move_group.get_current_pose().pose.orientation
    ))
    quat = tf_conversions.transformations.quaternion_from_euler(np.pi, 0, np.pi / 2)
    path.poses.append(geometry_msgs.msg.Pose(
        position=geometry_msgs.msg.Point(x=0.7, y=0.3, z=0.023),
        orientation=geometry_msgs.msg.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    ))
    path.poses.append(geometry_msgs.msg.Pose(
        position=geometry_msgs.msg.Point(x=0.1, y=-0.4, z=0.023),
        orientation=geometry_msgs.msg.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    ))
    path.poses.append(geometry_msgs.msg.Pose(
        position=geometry_msgs.msg.Point(x=0.7, y=-0.4, z=0.023),
        orientation=geometry_msgs.msg.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    ))
    path.poses.append(geometry_msgs.msg.Pose(
        position=geometry_msgs.msg.Point(x=0.1, y=0.3, z=0.023),
        orientation=geometry_msgs.msg.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    ))
    (plan, fraction) = move_group.compute_cartesian_path(path.poses, 0.01, 0.0)
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = move_group.get_current_state()
    display_trajectory.trajectory.append(plan)
    dt = 0.0
    for waypoint in display_trajectory.trajectory[0].joint_trajectory.points:
        waypoint.time_from_start = rospy.Duration(dt)
        dt += 0.06
    # self._display_trajectory_topic.publish(display_trajectory);
    ret = move_group.execute(plan, wait=True)