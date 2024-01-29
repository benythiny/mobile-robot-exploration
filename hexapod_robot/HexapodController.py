#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
 
import numpy as np
 
#import messages
from messages import *
 
DELTA_DISTANCE = 0.12
C_TURNING_SPEED = 5
C_AVOID_SPEED = 10
ORIENTATION_THRESHOLD = math.pi / 6
PI = math.pi
 
 
class HexapodController:
    def __init__(self):
        pass
 
    def goto(self, goal, odometry, collision):
        """
        Method to steer the robot towards the goal position given its current 
        odometry and collision status
        
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
        Returns:
            cmd: Twist steering command
        Notes:
            gx, gy = goal.x, goal.y
            cx, cy = odometry.position.x, odometry.position.y
            dst_to_target = np.linalg.norm(np.array([gx,cx,gy-cy])) #((gx-cx)**2-(gy-cy)**2)**(1/2)
            is_in_goal = dst_to_target <Constants.DELTA_DISTANCE
        """
        cmd_msg = Twist() # zero velocity steering command
        if collision:
            cmd_msg.linear.x = 0
            cmd_msg.angular.z = 0
            return None
        if (goal is not None) and (odometry is not None):
            diff = goal.position-odometry.pose.position
            dst_to_target = (diff).norm()
            is_in_goal = dst_to_target <DELTA_DISTANCE
            if is_in_goal:
                return None
            targ_h = np.arctan2(diff.y,diff.x)
            cur_h = odometry.pose.orientation.to_Euler()[0]
            diff_h = targ_h - cur_h
            diff_h = (diff_h + math.pi) % (2*math.pi) - math.pi
            if abs(diff_h) > np.pi/6:
                cmd_msg.linear.x = 0
                cmd_msg.angular.z = 10*C_TURNING_SPEED*diff_h
            else:
                cmd_msg.linear.x = dst_to_target
                cmd_msg.angular.z = C_TURNING_SPEED*diff_h
        return cmd_msg 
 
    def goto_reactive(self, goal, odometry, collision, laser_scan):
        """Method to steer the robot towards the goal position while avoiding
           contact with the obstacles given its current odometry, collision
           status and laser scan data
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
            laser_scan: LaserScan data perceived by the robot
        Returns:
            cmd: Twist steering command
        """
        # zero velocity steering command
        cmd_msg = Twist()
 
        if odometry is None or goal is None or laser_scan is None or collision is None:
            return cmd_msg
 
        goal_x = goal.position.x
        goal_y = goal.position.y
 
        current_x = odometry.pose.position.x
        current_y = odometry.pose.position.y
        current_ori = odometry.pose.orientation.to_Euler()[0]  # z
 
        if collision:
            return None
 
        dist_to_goal = odometry.pose.dist(goal)
 
        dx = goal_x - current_x
        dy = goal_y - current_y
 
        # goal orientation relative
        goal_ori_rel = math.atan2(dy, dx)
 
        # delta angle for big distance
        dphi = self.get_shortest_difference(goal_ori_rel, current_ori)
 
        repulsive_force = self.filter_scan_msgs(laser_scan)
 
        # calculate angular velocity
        angular_speed_navigation_component = dphi * C_TURNING_SPEED
        angular_speed_avoidance_component = repulsive_force * C_AVOID_SPEED
        angular_speed = angular_speed_navigation_component + angular_speed_avoidance_component
 
        if dist_to_goal < DELTA_DISTANCE:
            # goal orientation absolute
            goal_ori_abs = goal.orientation.to_Euler()[0]
 
            # delta angle for small distance
            dphi_close = self.get_shortest_difference(goal_ori_abs, current_ori)
 
            if abs(dphi_close) >= ORIENTATION_THRESHOLD:
                return None
 
            cmd_msg.linear.x = 0
            cmd_msg.linear.y = 0
            cmd_msg.angular.z = C_TURNING_SPEED * dphi_close
            return cmd_msg
 
        cmd_msg.linear.x = dist_to_goal
        cmd_msg.linear.y = dist_to_goal
        # set angular velocity
        cmd_msg.angular.z = angular_speed
        return cmd_msg
 
 
  
    def filter_scan_msgs(self, scan_msg):
        angle_incr = scan_msg.angle_increment
        angle_max = scan_msg.angle_max
        angle_min = scan_msg.angle_min
        distances = scan_msg.distances
        range_max = scan_msg.range_max
        range_min = scan_msg.range_min
 
        for d in range(len(distances)):
            if distances[d] < range_min:
                distances[d] = range_min
            if distances[d] > range_max:
                distances[d] = range_max
 
        center = math.floor(len(distances)/2)
        left_side = distances[:center]
        right_side = distances[center:]
 
        scan_left = min(left_side)
        scan_right = min(right_side)
 
        rep_force = 1 / scan_left - 1 / scan_right
        return rep_force