#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
 
import numpy as np
 
#import messages
from messages import *

#import robot parameters
from HexapodRobotConst import *
 
 
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
        """
        if collision:
            return None
        
        cmd_msg = Twist() # zero velocity steering command
        
        if (goal is None) or (odometry is None):
            return cmd_msg
        
        current_heading = odometry.pose.orientation.to_Euler()[0]
        
        diff_x = goal.position.x - odometry.pose.position.x 
        diff_y = goal.position.y - odometry.pose.position.y
        
        dst_to_target = np.linalg.norm(np.array([diff_x, diff_y]))
        
        target_heading = np.arctan2(diff_y,diff_x)
        
        if dst_to_target <DELTA_DISTANCE:
            return None

        diff_heading = (target_heading - current_heading + PI) % (2*PI) - PI
        
        abs_diff_heading = abs(diff_heading)
        if abs_diff_heading < ORIENTATION_THRESHOLD:
            cmd_msg.angular.z = diff_heading * C_TURNING_SPEED
            cmd_msg.linear.x = dst_to_target
        else:
            cmd_msg.angular.z = diff_heading * C_AVOID_SPEED*C_TURNING_SPEED
            cmd_msg.linear.x = 0
            
        return cmd_msg 
 
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