# -*- coding: utf-8 -*-
import math

"""
Locomotion-related constants
"""
TIME_FRAME = 1 #the length of the simulation step and real robot control frame [ms]

SERVOS_BASE = [	0, 0, -math.pi/6., math.pi/6., math.pi/3., -math.pi/3.,
                0, 0, -math.pi/6., math.pi/6., math.pi/3., -math.pi/3.,
                0, 0, -math.pi/6., math.pi/6., math.pi/3., -math.pi/3.]

#robot construction constants
COXA_SERVOS = [1,13,7,2,14,8]
COXA_OFFSETS = [math.pi/8,0,-math.pi/8,-math.pi/8,0,math.pi/8]
FEMUR_SERVOS = [3,15,9,4,16,10]
FEMUR_OFFSETS = [-math.pi/6,-math.pi/6,-math.pi/6,math.pi/6,math.pi/6,math.pi/6]
TIBIA_SERVOS = [5,17,11,6,18,12]
TIBIA_OFFSETS = [math.pi/3,math.pi/3,math.pi/3,-math.pi/3,-math.pi/3,-math.pi/3]
SIGN_SERVOS = [-1,-1,-1,1,1,1]

FEMUR_MAX = 2.1
TIBIA_MAX = 0.1
COXA_MAX = 0.01

SPEEDUP = 4

"""
Custom constants
"""
ROBOT_SIZE = 0.4
EXTRA_PADDING = 0.13
DELTA_DISTANCE = 0.12
C_TURNING_SPEED = 5
C_AVOID_SPEED = 10
THREAD_SLEEP = 0.4
LASER_MAX_RANGE = 10.0 #laser_scan.range_max
LASER_MIN_RANGE = 0.01 #laser_scan.range_min
