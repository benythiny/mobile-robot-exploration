#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import socket
import json

import sys
import time
import threading as thread
 
import numpy as np
import matplotlib.pyplot as plt
 
sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')
 
#import hexapod robot and explorer
import HexapodRobot 
import HexapodExplorer
 
#import communication messages
from messages import *

from lkh.invoke_LKH import solve_TSP

import argparse

ROBOT_SIZE = 0.4
THREAD_SLEEP = 0.4
FRONTIER_DIST = 0.5

PLANNING_METHOD = 3

SOCKET_PORT = 32000

ex0 = None
ex1 = None

ext_robot_detected = False
 
class Explorer:
    """ Class to represent an exploration agent
    """
    def __init__(self, planning_var=1, robotID = 0):
 
 
        """ VARIABLES
        """ 
        self.id = robotID
        
        #current frontiers
        self.frontiers = None
 
        #current path
        self.path = Path()
 
        #stopping condition
        self.stop = False
        
        # define frontiers
        self.frontiers = None
        self.goal_frontier = None
        
        #prepare the gridmap
        self.gridmap = OccupancyGrid()
        self.gridmap.resolution = 0.1
        self.gridmap.width = 100
        self.gridmap.height = 100
        self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height*self.gridmap.width))
        
        # map for planning with obstacles
        self.gridmap_inflated = OccupancyGrid()
        
        # map for path planning only 
        self.gridmap_planning = OccupancyGrid()
        
        # terminating condition
        self.terminate_counts = 10
        
        # planning methods
        self.planning_method = planning_var
        
        # path that is displayed on the graph
        self.path_to_draw = Path()
        
        self.odometry = None
        self.laser_scan = None
        self.API_navigation_goal = None
        self.current_goal = None
        self.collision_on_path = False
        self.other_robot_pos = None
        
        
        """Connecting the simulator
        """
        
        if self.id == 0:
            #instantiate the robot
            self.robot = HexapodRobot.HexapodRobot(robotID)
        #and the explorer
        self.explor = HexapodExplorer.HexapodExplorer()
                
    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning
        """
        if self.id == 0:
            #turn on the robot 
            self.robot.turn_on()
            print("checkpoint0")
    
            #start navigation thread
            self.robot.start_navigation()
            
            print("checkpoint1")
            
            #start the API update thread
            try:
                update_thread = thread.Thread(target=self.update_from_API)
                update_thread.start() 
            except:
                print("Error: unable to start API update thread")
                sys.exit(1) 
        
        #start the mapping thread
        try:
            mapping_thread = thread.Thread(target=self.mapping)
            mapping_thread.start() 
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)      
 
        #start planning thread
        try:
            planning_thread = thread.Thread(target=self.planning)
            planning_thread.start() 
        except:
            print("Error: unable to start planning thread")
            sys.exit(1)
 
        #start trajectory following
        try:
            traj_follow_thread = thread.Thread(target=self.trajectory_following)
            traj_follow_thread.start() 
        except:
            print("Error: unable to start planning thread")
            sys.exit(1)
 
    def __del__(self):
        if self.id == 0:
            #turn off the robot
            #TODO: check one more time
            self.robot.navigation_goal = None
            time.sleep(0.2)
            self.robot.stop_navigation()
            self.robot.turn_off()
 
    def update_from_API(self):
        print(time.strftime("%H:%M:%S"), "Started API updating thread")
        while not self.stop:
            time.sleep(0.1)
            if self.id == 0:
                self.odometry = self.robot.odometry_
                self.laser_scan = self.robot.laser_scan_
                self.API_navigation_goal = self.robot.navigation_goal
    
    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map
        """
        print(time.strftime("%H:%M:%S"), "Started the mapping thread") 
        while not self.stop:
            time.sleep(THREAD_SLEEP)
            
            odometry = self.odometry
            laser_scan = self.laser_scan
               
            #get the current laser scan and odometry and fuse them to the map
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, laser_scan, odometry)
            
            # add the other robot to the infused map as an obstacle if its position is known
            gridmap_with_other_robot = copy.deepcopy(self.gridmap)
            if self.other_robot_pos is not None:
                x, y = self.other_robot_pos
                [y, x] = ex0.explor.world_to_map(ex0.gridmap, np.array([y, x]))
                gridmap_with_other_robot.data[y * self.gridmap.width + x] = 1
            
            #obstacle growing for the map used for continous collision avoidance
            self.gridmap_inflated = self.explor.grow_obstacles(gridmap_with_other_robot, ROBOT_SIZE)
            
            # obstacle growing for the map used for path planning
            self.gridmap_planning = self.explor.grow_obstacles(gridmap_with_other_robot, ROBOT_SIZE + 0.1)
 
    def goal_is_still_a_frontier(self):
        """
        method to monitor current frontiers. if the goal_frontier is not present
        within the FRONTIER_DIST radius, the goal is removed and a new goal is searched for.
        """ 
        # check if the goal is still a frontier, erase the path otherwise    
        if self.path is None:
            return
        
        if len(self.path.poses) == 0:
            return
        
        if self.goal_frontier is None:
            return
        
        frontier_still_there = False
        for f in self.frontiers:
            # distance from goal_frontier to every new frontier
            dist = self.goal_frontier.dist(f)
            
            # if the new frontier is still close to the original goal, continue
            if dist < FRONTIER_DIST:
                frontier_still_there = True
                continue
            
        if frontier_still_there == False:
            print(time.strftime("%H:%M:%S"), "The old frontier goal is not there anymore. Rerouting to the next closest frontier...")
            self.path = None
            self.robot.navigation_goal = None
          
    def check_termination_condition(self):
        """
        method to monitor the amount of self.terminate_counts left before terminating the script.
        the amount of self.terminate_counts is reduced every time there are no feasible frontiers.
        """ 
        # if there are no more frontiers, return the app 
        if self.frontiers is None:
            self.terminate_counts -= 1
            print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
            
        if self.terminate_counts <= 0:
                print(time.strftime("%H:%M:%S"), "No feasible frontiers detected. Terminating the script.")
                self.stop = True
            
    def check_obstacles_on_path(self):
        """
        method to continuosuly monitor presence of obstacles on the current path using the Bresenham line.
        in case of obstacles detection, the current path is removed. 
        """
        # check if there are no new obstacles on the planned path 
        collision = False
        odometry = self.odometry
            
        if odometry is None or self.path is None:
            return      
        
        map2d = self.gridmap_inflated.data.reshape(self.gridmap_inflated.height, self.gridmap_inflated.width)

        # append current robot position to the current planned path
        trajectory = []
        trajectory.append(odometry.pose)
        trajectory.extend(copy.deepcopy(self.path.poses))
        
        for p in range(len(trajectory) - 1):
            
            # one point from the path
            y, x = trajectory[p].position.y, trajectory[p].position.x 
            b_start = self.explor.world_to_map(self.gridmap_inflated, np.array([y, x]))
            
            # the following point on the path 
            y, x = trajectory[p+1].position.y, trajectory[p+1].position.x 
            b_end = self.explor.world_to_map(self.gridmap_inflated, np.array([y, x]))
                
            # calculate the bresenham line
            b_line = self.explor.bresenham_line(b_start, b_end)
            
            #check for collision
            for (y, x) in b_line: 
                if map2d[y,x] > 0.5: 
                    collision = True
                    break
        
        if collision == True:   # collision detected
            print(time.strftime("%H:%M:%S"),"Collision detected on the planned path. Rerouting...")
            
            self.path = None
            self.collision_on_path = True
            
            if self.id == 0:
                self.robot.navigation_goal = None   # stop the robot
                
        else: 
            self.collision_on_path = False
      
    def check_if_shorter_path_exists(self):
        """
        method to detect if the current path is too much discretized and to find a shorter simplified path.
        """
            
        if self.path is None or self.goal_frontier is None:
            return
        
        orig_path_length = len(self.path.poses)
        if orig_path_length < 12:
            return
        
        odometry = self.odometry
        if odometry is None:
            return
        
        print(time.strftime("%H:%M:%S"), "Path is a bit too long, trying to find a shorter one...")
            
        start = odometry.pose
        alternative_path = self.explor.plan_path(self.gridmap_planning, start, self.goal_frontier)
        alternative_path = self.explor.simplify_path(self.gridmap_planning, alternative_path)
        
        if alternative_path is None:
            return
        
        if len(alternative_path.poses) < (orig_path_length - 1):
            print(time.strftime("%H:%M:%S"), "A shorter path was found. Reformating path.")
            self.path = alternative_path
          
    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path  
        """
        time.sleep(4*THREAD_SLEEP)
        print(time.strftime("%H:%M:%S"), "Planning thread started.")
        
        while not self.stop:
            time.sleep(THREAD_SLEEP)
            odometry = self.odometry 
            if odometry is None:
                continue
            
            # get current robot's position
            start = odometry.pose

            #frontier calculation when there is no planned path
            if self.path is None or len(self.path.poses) == 0:
                
                # implementation of task: f1 + f2 extenstion, preparation for task p1
                if self.planning_method == 1:
                    self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap_inflated)
                    self.frontiers = self.explor.sort_frontiers_by_dist(self.gridmap_inflated, start, self.frontiers)
                
                # implementation of task: f3, preparation for task p2
                elif self.planning_method == 2: 
                    self.frontiers = self.explor.find_inf_frontiers(self.gridmap_inflated)
                    
                # implementation of task: f1 + f2 extenstion, further used with task p3
                elif self.planning_method == 3:
                    self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap_inflated)
                
            # after detecting frontiers, check if there are any left
            self.check_termination_condition()       
            if self.frontiers is None or self.stop:
                continue
            
            # if path is already planned, check path conditions
            if self.path is not None and len(self.path.poses) > 0:
                
                # extend the path that is displayed on the graph with the current robot's position
                self.path_to_draw.poses = []
                self.path_to_draw.poses.append(start)
                self.path_to_draw.poses.extend(self.path.poses)
                
                self.check_if_shorter_path_exists()
                        
                self.check_obstacles_on_path()
                continue
            
            #path planning and goal selection 
            current_path = None
            
            # implementation of task: p3
            if self.planning_method == 3:
                
                current_frontiers = self.frontiers
                if current_frontiers is None:
                    continue
                
                # TSP cannot be solved
                if len(current_frontiers) == 1:
                    self.goal_frontier
                    #self.path = self.explor.plan_path(self.gridmap_planning, start, self.goal_frontier)
                    #continue
                else:   # solve TSP
                    print(time.strftime("%H:%M:%S"), "Started planning the path with TSP.")
                    
                    # 0. add current location to all frontiers
                    all_locations = []
                    all_locations.append(start)
                    all_locations.extend(current_frontiers)
                    num_locations = len(all_locations)
                    
                    # 1. construct a distance matrix
                    dist_matrix = np.zeros((num_locations, num_locations))
                    for row in range(num_locations):
                        for col in range(1, num_locations):
                            current_path = self.explor.plan_path(self.gridmap_planning, all_locations[row], all_locations[col])
                            if current_path is None:
                                value = 10000
                            else:
                                value = len(current_path.poses)
                            dist_matrix[row][col] = value
                            
                    # 2. find the shortest feasible tour by solving the TSP problem
                    TSP_result = solve_TSP(dist_matrix)
                    #print(time.strftime("%H:%M:%S"), TSP_result)
                    if len(TSP_result) < 2:
                        self.terminate_counts -= 1
                        print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
                        print(time.strftime("%H:%M:%S"), "TSP solution was not found.")
                        print(time.strftime("%H:%M:%S"), "TSP result was ", TSP_result)
                        continue
                    
                    # 3. assign the first frontier of the tour as the next exploration goal
                    #TODO: remove this try except
                    try:
                        goal_frontier = all_locations[TSP_result[1]]
                    except:
                        print(time.strftime("%H:%M:%S"), "EXCEPTION OCCURED")
                        print(time.strftime("%H:%M:%S"), "All locations: ", all_locations)
                        print(time.strftime("%H:%M:%S"), TSP_result)
                        self.terminate_counts -= 1
                        print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
                        print(time.strftime("%H:%M:%S"), "TSP solution was not found.")
                        continue
                
                # 4. plan path to this frontier
                current_path = self.explor.plan_path(self.gridmap_planning, start, goal_frontier)
                simple_path  = self.explor.simplify_path(self.gridmap_planning, current_path)
                if current_path is not None:
                    if simple_path is not None:
                        self.path = simple_path
                    else:
                        self.path = current_path
                    self.goal_frontier = goal_frontier
                    print(time.strftime("%H:%M:%S"), "Path was successfully planned.")
                    print(time.strftime("%H:%M:%S"), "Going to frontier at x: ", "{:.2f}".format(self.goal_frontier.position.x), ", y:", "{:.2f}".format(self.goal_frontier.position.y))
                    
                else:
                    print(time.strftime("%H:%M:%S"), "TSP solution was not found, couldn't plan path.")
                    print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
                    self.frontiers = []
                    self.terminate_counts -= 1
            
            else:
                # implementation of tasks: p1 and p2, the frontiers are already sorted in a desired way
                
                print(time.strftime("%H:%M:%S"), "Started planning the path.")
                
                # go through all frontiers and try to plan path to the first one in the list, which is
                # 1) the closest one to the robot in case of p1
                # 2) has the biggest info gain in case of p2
                # if the path couldn't be planned, go to the next one
                for f in self.frontiers:
                    self.goal_frontier = f
                    self.path   = self.explor.plan_path(self.gridmap_planning, start, self.goal_frontier) 
                    simple_path = self.explor.simplify_path(self.gridmap_planning, self.path)
                    
                    if self.path is not None:       
        
                        if simple_path is not None:
                            self.path = simple_path
                            
                        print(time.strftime("%H:%M:%S"), "Path was successfully planned.")
                        print(time.strftime("%H:%M:%S"), "Going to frontier at x:", "{:.2f}".format(self.goal_frontier.position.x), 
                                                                            ", y:", "{:.2f}".format(self.goal_frontier.position.y))
                        break
                        
                    else:
                        print(time.strftime("%H:%M:%S"), "No path was found.")
                        
                # when no path could be planned for any of the frontiers, increase the termination counts
                if self.path is None:
                    
                    self.frontiers = []
                    self.terminate_counts -= 1
                    print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
                                      
    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """ 
        time.sleep(6*THREAD_SLEEP)
        print(time.strftime("%H:%M:%S"), "Trajectory thread started.")
        
        while not self.stop:
            time.sleep(2 * THREAD_SLEEP)
            if self.API_navigation_goal is None:
                if self.path is not None and len(self.path.poses) > 0:
                    #fetch the new navigation goal                    
                    if len(self.path.poses) > 1:
                        nav_goal = self.path.poses[1]
                        self.current_goal = nav_goal
                        #give it to the robot, if the robot is directly connected to API, send command
                        if self.id == 0:
                            self.robot.goto(nav_goal)
                    # remove the previous navigation goal
                    prev_goal = self.path.poses.pop(0)
  
       
class TCP_Server():
    """
    Class to represent server for TCP socket communication
    """
    def __init__(self):
        self.stop = False
        self.client_socket = None
        self.external_robot = None
        
    def send_socket(self ):
        print(time.strftime("%H:%M:%S"), "Socket sending thread started")
        message = "Hello from robot server!"
        self.client_socket.send(message.encode()) 
        while not self.stop:
            time.sleep(0.3)
        
            if ex0 is None or ex0.gridmap is None or ex1 is None:
                continue
                
            # get the odometry data of external robot from API
            if ex1.odometry_ is not None:
                odom_valid = True
                odometry_position = [ex1.odometry_.pose.position.x, ex1.odometry_.pose.position.y,]
                odometry_orientation = [ex1.odometry_.pose.orientation.x, ex1.odometry_.pose.orientation.y, 
                                        ex1.odometry_.pose.orientation.z,
                                        ex1.odometry_.pose.orientation.w]
                
                ex0.other_robot_pos = odometry_position
                
            else:
                odom_valid = False
                odometry_position = [0, 0]
                odometry_orientation = [0, 0, 0, 0]
                
            # get the laser scan data of external robot from API
            if ex1.laser_scan_ is not None:
                scan_valid = True
                scan_msg = [ex1.laser_scan_.angle_increment,
                        ex1.laser_scan_.angle_max, 
                        ex1.laser_scan_.angle_min,
                        ex1.laser_scan_.distances,
                        ex1.laser_scan_.range_max,
                        ex1.laser_scan_.range_min
                ]
            else:
                scan_valid = False
                scan_msg = [0, 0, 0, 0, 0, 0]
                
            # get the navigation goal of ex1 of external robot from API
            
            if ex1.navigation_goal is not None:
                navigation_goal_active = True
            else:
                navigation_goal_active = False
                
            # set current robot's position
            if ex0.odometry is not None:
                other_robot_pos_valid = True
                other_robot_pos = [ex0.odometry.pose.position.x, ex0.odometry.pose.position.y]
            else:
                other_robot_pos_valid = False
                other_robot_pos = [0, 0]
            
            # create a json message
            message = {"map" : ex0.gridmap.data.tolist(),
                        "odometry_valid" : odom_valid,
                        "odometry_position" : odometry_position,
                        "odometry_orientation" : odometry_orientation,
                        "scan_valid" : scan_valid,
                        "scan_msg" : scan_msg,
                        "nav_goal_active" : navigation_goal_active,
                        "other_robot_pos_valid" : other_robot_pos_valid,
                        "other_robot_pos" : other_robot_pos}
            
            message = json.dumps(message)
            
            # send the message
            try:
                self.client_socket.send(message.encode()) 
            except:
                print(time.strftime("%H:%M:%S"), "Couldn't send a message over the socket.")
                
    def receive_socket(self):
        
        print(time.strftime("%H:%M:%S"), "Socket receiving thread started")
        received_data = self.client_socket.recv(4000)
        print(time.strftime("%H:%M:%S"), received_data.decode())
        while not self.stop:
            global ex1
            global ex0
            time.sleep(0.3)
            received_data = self.client_socket.recv(200000)
            #print(time.strftime("%H:%M:%S"), "Received: ", received_data)
            try:
                received_data    = json.loads(received_data)
                ex0.gridmap.data = np.array(received_data['map'])
                
                if ex1 is not None:
                    if received_data['goal_valid']:
                        goal = Pose()
                        goal.position.x = received_data['goal_position'][0]
                        goal.position.y = received_data['goal_position'][1]
                        
                        goal.orientation.x = received_data['goal_orientation'][0]
                        goal.orientation.y = received_data['goal_orientation'][1]
                        goal.orientation.z = received_data['goal_orientation'][2]
                        goal.orientation.w = received_data['goal_orientation'][3]
                        
                        ex1.goto(goal)
                        
                    if received_data['collision_on_path']:
                        ex1.navigation_goal = None
                    
                    ex0.stop = received_data['stop']
                    
            except:
                continue
                    
    def start_socket_exchange(self ):
        # Create a socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to a specific address and port
        server_socket.bind(('localhost', SOCKET_PORT))

        # Listen for incoming connections
        server_socket.listen()

        # Accept a connection
        self.client_socket, client_address = server_socket.accept()
        
        global ext_robot_detected
        ext_robot_detected = True
            
        #start the sending thread
        try:
            socket_sending_thread = thread.Thread(target=self.send_socket)
            socket_sending_thread.start() 
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)
            
        #start the receiving thread
        try:
            socket_receiving_thread = thread.Thread(target=self.receive_socket)
            socket_receiving_thread.start() 
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)
 
                                    
class TCP_Client():
    """
    Class to represent client for TCP socket communication
    """               
    def __init__(self):
        self.stop = False
        self.client_socket = None
        self.external_robot = None
    
    def send_socket(self ):
        print(time.strftime("%H:%M:%S"), "Socket send thread started")
        message = "Hello from robot client!"
        self.client_socket.send(message.encode()) 
        while not self.stop:
            time.sleep(0.3)
        
            if ex0 is None or ex0.gridmap is None:
                continue
                        
            # set robot's current goal to be send to API from server
            if ex0.current_goal is not None:
                goal_valid = True
                goal_position = [ex0.current_goal.position.x, ex0.current_goal.position.y]
                goal_orientation = [ex0.current_goal.orientation.x, ex0.current_goal.orientation.y, 
                                    ex0.current_goal.orientation.z,
                                    ex0.current_goal.orientation.w]
            else:
                goal_valid = False
                goal_position = [0,0]
                goal_orientation = [0,0,0,0]
                
            # set the robot's termination condition
            if ex0.terminate_counts < 1:
                stop = True
            else:
                stop = False
            
            # create a json message
            message = {"stop": stop,
                        "map" : ex0.gridmap.data.tolist(),
                        "goal_valid" : goal_valid,
                        "goal_position" : goal_position,
                        "goal_orientation" : goal_orientation,
                        "collision_on_path" : ex0.collision_on_path}
            
            
            message = json.dumps(message)
            
            # send the message over the socket
            try:
                self.client_socket.send(message.encode()) 
            except:
                print(time.strftime("%H:%M:%S"), "Couldn't send a message over the socket.")
        
    def receive_socket(self):
        
        print(time.strftime("%H:%M:%S"), "Socket receive thread started")
        received_data = self.client_socket.recv(4000)
        print(received_data.decode())
        while not self.stop:
            global ex0
            time.sleep(0.3)   
            received_data = self.client_socket.recv(200000)
               
            if ex0 is None:
                continue
                
            try:
                # try receiveing the message and decoding it from json to dictionary if it was not corrupted
                received_data = json.loads(received_data)
                
                ex0.gridmap.data = np.array(received_data['map'])

                # receive the other robot's position to add it as obstacle to the map
                if received_data['other_robot_pos_valid']:
                    ex0.other_robot_pos = received_data['other_robot_pos']                
                
                # receive current robot's odometry
                if received_data['odometry_valid']:
                    ex0.odometry = Odometry()
                    ex0.odometry.pose.position.x = received_data['odometry_position'][0]
                    ex0.odometry.pose.position.y = received_data['odometry_position'][1]
                    
                    ex0.odometry.pose.orientation.x = received_data['odometry_orientation'][0]
                    ex0.odometry.pose.orientation.y = received_data['odometry_orientation'][1]
                    ex0.odometry.pose.orientation.z = received_data['odometry_orientation'][2]
                    ex0.odometry.pose.orientation.w = received_data['odometry_orientation'][3]
                
                # receive current robot's laser scan data
                if received_data['scan_msg']:
                    ex0.laser_scan = LaserScan()
                    ex0.laser_scan.angle_increment = received_data['scan_msg'][0]
                    ex0.laser_scan.angle_max = received_data['scan_msg'][1]
                    ex0.laser_scan.angle_min = received_data['scan_msg'][2]
                    ex0.laser_scan.distances = received_data['scan_msg'][3]
                    ex0.laser_scan.range_max = received_data['scan_msg'][4]
                    ex0.laser_scan.range_min = received_data['scan_msg'][5]
                    
                    if not isinstance(ex0.laser_scan.distances, list):
                        ex0.laser_scan.distances = []
                    
                # receive the status of the navigation of the current robot
                if received_data['nav_goal_active']:
                    ex0.API_navigation_goal = True
                else:
                    ex0.API_navigation_goal = None   
            except:
                continue        
          
    def start_socket_exchange(self ):
        # Create a socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        self.client_socket.connect(('localhost', SOCKET_PORT))
            
        #start the sending thread
        try:
            socket_sending_thread = thread.Thread(target=self.send_socket)
            socket_sending_thread.start() 
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)
            
        #start the receiving thread
        try:
            socket_receiving_thread = thread.Thread(target=self.receive_socket)
            socket_receiving_thread.start() 
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)
                               
  
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                                    prog='Explorer',
                                    description='performs an autonomous robot exploration',
                                    epilog='Enjoy the mobile robot exploration!')
    parser.add_argument('-p', '--planning', type=int, default=1, choices=[1,2,3],required=False, help="Choose the planning variant.") 
    parser.add_argument('ID', type=int, help="The robot's ID starting from 0.") 
    parser.add_argument('-m', '--multi', action='store_true', help="Use this flag to run the multi robot variant")
    
    args = parser.parse_args()
    
    print(time.strftime("%H:%M:%S"), "Multi robot exploration status is ", args.multi)
    print(time.strftime("%H:%M:%S"), "The chosen planning variant is #",args.planning)
    print(time.strftime("%H:%M:%S"), "Robot's id is #", args.ID)
    
    if args.multi:
        if args.ID == 0:
            tcp = TCP_Server()
        else:
            tcp = TCP_Client()

        tcp.start_socket_exchange()
    
    #instantiate the robot
    ex0 = Explorer(planning_var=args.planning, robotID=args.ID)
    
    # the strcture needed for connecting more than one robot to the API
    if not args.multi or args.ID != 0:
        #start the locomotion
        ex0.start()
    
    #continuously plot the map, targets and plan (once per second)
    fig, (ax, bx) = plt.subplots(nrows=2, ncols=1, figsize=(10,25), num=args.ID)
    plt.ion()
    while not ex0.stop:
        
        if ext_robot_detected:
            ex1 = HexapodRobot.HexapodRobot(1)
            time.sleep(1)
            
            #turn on the current robot 
            ex0.start()
            
            #turn on the external robot 
            ex1.turn_on()
            
            #start navigation thread for extermnal robot
            ex1.start_navigation()
            
            ext_robot_detected = False
        
        plt.cla()
        ax.cla()
        bx.cla()
        
        #plot the map
        ex0.gridmap.plot(ax)
        ex0.gridmap_inflated.plot(bx)
        
        # print all detected frontiers as red dots
        if ex0.frontiers is not None:
            for frontier in ex0.frontiers: 
                ax.scatter(frontier.position.x, frontier.position.y,c='red')
                
        #print simplified path points
        if ex0.path is not None and len(ex0.path.poses) > 0: 
            for pose in ex0.path.poses:
                ax.scatter(pose.position.x, pose.position.y,c='green', s=100, marker='x')
                bx.scatter(pose.position.x, pose.position.y,c='green', s=100, marker='x')
            
        #print the whole path as connected points
        if len(ex0.path_to_draw.poses) > 0:
            ex0.path_to_draw.plot(ax=ax, style = 'point')
            ex0.path_to_draw.plot(ax=bx, style = 'point')
            
        
        # draw current robot's pose
        if ex0.odometry is not None:
            ax.scatter(ex0.odometry.pose.position.x, ex0.odometry.pose.position.y,c='blue', s = 200)
            bx.scatter(ex0.odometry.pose.position.x, ex0.odometry.pose.position.y,c='blue', s = 200)

        
        
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
    
        #to throttle the plotting pause for 1s
        plt.pause(THREAD_SLEEP)
    
    if args.multi:
        tcp.stop = True
        
    try:
        if ex1 is not None:
            ex1.stop_navigation()
            ex1.turn_off()
        ex0.__del__()
    except:
        print(time.strftime("%H:%M:%S"), "The script was terminated")
        
    
    