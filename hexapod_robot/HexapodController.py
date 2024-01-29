#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
 
import numpy as np
 
#import messages
from messages import *
 
DELTA_DISTANCE = 0.12
C_TURNING_SPEED = 5
C_AVOID_SPEED = 10
ORIENTATION_THRESHOLD = math.pi / 16
PI = math.pi
 
 
class HexapodController:
    def __init__(self):
        pass
 
    def goto(self, goal, odometry, collision):
        """Method to steer the robot towards the goal position given its current
           odometry and collision status
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
        Returns:
            cmd: Twist steering command
        """
        # zero velocity steering command
        cmd_msg = Twist()
 
        if odometry is None or goal is None or collision is None:
            return cmd_msg
 
        goal_x = goal.position.x
        goal_y = goal.position.y
 
        current_x = odometry.pose.position.x
        current_y = odometry.pose.position.y
        current_ori = odometry.pose.orientation.to_Euler()[0]  # z
 
        if collision:
            return None
 
        '''
        if abs(goal_x - current_x) < DELTA_DISTANCE and abs(goal_y - current_y) < DELTA_DISTANCE:
            return None
        '''
 
        dist_to_goal = odometry.pose.dist(goal)
 
        dx = goal_x - current_x
        dy = goal_y - current_y
 
        # goal orientation relative
        goal_ori_rel = math.atan2(dy, dx)
 
        # goal orientation absolute
        goal_ori_abs = goal.orientation.to_Euler()[0]
 
        # delta angle for big distance
        # dphi = self.get_angle_diff(goal_ori_rel, current_ori)
        dphi = self.get_shortest_difference(goal_ori_rel, current_ori)
 
        if dist_to_goal < DELTA_DISTANCE:
            # delta angle for small distance
            # dphi_close = self.get_angle_diff(goal_ori_abs, current_ori)
            dphi_close = self.get_shortest_difference(goal_ori_abs, current_ori)
 
            if abs(dphi_close) >= ORIENTATION_THRESHOLD:
                return None
            cmd_msg.linear.x = 0
            cmd_msg.linear.y = 0
            cmd_msg.angular.z = C_TURNING_SPEED * dphi_close
            return cmd_msg
 
        cmd_msg.linear.x = dist_to_goal
        cmd_msg.linear.y = dist_to_goal
        cmd_msg.angular.z = dphi * C_TURNING_SPEED
        print("Goto message: ", cmd_msg)
        return cmd_msg
    
    def get_shortest_difference(self, th1, th2):
        # Calculate the angle difference between th1 and th2, and use modulo to ensure it's within [0, 2 * pi].
        anglediff = (th1 - th2) % (2 * math.pi)
 
        # If the angle difference is negative, adjust it to be the shortest positive angle.
        if anglediff < 0:
            if abs(anglediff) > (2 * math.pi + anglediff):
                anglediff = 2 * math.pi + anglediff
        else:
            if anglediff > abs(anglediff - 2 * math.pi):
                anglediff = anglediff - 2 * math.pi
 
        return anglediff
 
 
    def get_angle_diff(self, th1, th2):
        delta = ((th2 - th1 + PI) % (2 * PI)) - PI
        return delta
 
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