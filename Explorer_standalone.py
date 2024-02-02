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

ex0 = None
ex1 = None

ext_robot_detected = False

'''gridmap_common = OccupancyGrid()
gridmap_common.resolution = 0.1
gridmap_common.width = 100
gridmap_common.height = 100
gridmap_common.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
gridmap_common.data = 0.5*np.ones((gridmap_common.height*gridmap_common.width))

 '''
 
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
        
        # flag for performing the initialising 360 d rotation
        self.init_rotating = True
        
        # path that is displayed on the graph
        self.path_to_draw = Path()
        
        self.odometry = None
        self.laser_scan = None
        self.API_navigation_goal = None
        
        
        """Connecting the simulator
        """
        
        if self.id == 0:
            #instantiate the robot
            self.robot = HexapodRobot.HexapodRobot(robotID)
        #and the explorer
        self.explor = HexapodExplorer.HexapodExplorer()
        
    ''' def socket_communication(self):
        if self.id == 0: # the robot is server
            # Create a socket
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Bind the socket to a specific address and port
            server_socket.bind(('localhost', 5000))

            # Listen for incoming connections
            server_socket.listen()

            # Accept a connection
            client_socket, client_address = server_socket.accept()
            
            received_data = client_socket.recv(1024)
            print(received_data.decode())
            
            
        else: # the robot is client
            # Create a socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect to the server
            client_socket.connect(('localhost', 5000))
            
            message = "Hello, robot!"
            client_socket.send(message.encode())''' 
                
    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning
        """
        if self.id == 0:
            #turn on the robot 
            self.robot.turn_on()
    
            #start navigation thread
            self.robot.start_navigation()
            
            #start the API update thread
            try:
                update_thread = thread.Thread(target=self.API_update)
                update_thread.start() 
            except:
                print("Error: unable to start API update thread")
                sys.exit(1) 
        
        '''# establish the socket communication
        try:
            socket_thread = thread.Thread(target=self.socket_communication)
            socket_thread.start() 
        except:
            print("Error: unable to start socket thread")
            sys.exit(1) '''
        
        
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
            self.robot.stop_navigation()
            self.robot.turn_off()
 
    def API_update(self):
        while not self.stop:
            time.sleep(0.1)
            if self.id == 0:
                self.odometry = self.robot.odometry_
                self.laser_scan = self.robot.laser_scan_
                self.API_navigation_goal = self.robot.navigation_goal
    
    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map
        """
        
        while not self.stop:
            time.sleep(THREAD_SLEEP)
            
            '''if self.id == 0:
                laser_scan = self.robot.laser_scan_
                odometry   = self.robot.odometry_'''
            odometry = self.odometry
            laser_scan = self.laser_scan
               
            #get the current laser scan and odometry and fuse them to the map
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, laser_scan, odometry)
            
            #obstacle growing for the map used for collision avoidance
            self.gridmap_inflated = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE)
            
            # obstacle growing for the map used for path planning
            self.gridmap_planning = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE + 0.1)
 
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
        # check if there ane no new obstacles on the planned path 
        collision = False
        '''if self.id == 0:
            odometry = self.robot.odometry_'''
        odometry = self.odometry
            
        if odometry is None:
            return     
        
        if self.path is None:
            return   
        
        map2d = self.gridmap_inflated.data.reshape(self.gridmap_inflated.height, self.gridmap_inflated.width)
            
        #map2d[cell[0], cell[1]] = 0

        # append current robot position to the planned path
        planned_path = [odometry.pose]
        planned_path.extend(copy.deepcopy(self.path.poses))
        
        '''for p in planned_path:
            y, x = p.position.y, p.position.x 
            [y, x] = self.explor.world_to_map(self.gridmap_inflated, np.array([y, x]))
            if map2d[y,x] > 0.5: 
                    collision = True
                    break'''
        
        for p in range(len(planned_path) - 1):
            y, x = planned_path[p].position.y, planned_path[p].position.x 
            b_start = self.explor.world_to_map(self.gridmap_inflated, np.array([y, x]))
            
            y, x = planned_path[p+1].position.y, planned_path[p+1].position.x 
            b_end = self.explor.world_to_map(self.gridmap_inflated, np.array([y, x]))
                
            b_line = self.explor.bresenham_line(b_start, b_end)
            
            for (y, x) in b_line: #check for collision
                if map2d[y,x] > 0.5: 
                    collision = True
                    break
        
        if collision == True:
            # collision detected
            print(time.strftime("%H:%M:%S"),"Collision detected on the planned path. Rerouting...")
            self.path = None
            
            if self.id == 0:
                self.robot.navigation_goal = None
      
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
        # self.init_mapping()
        self.init_rotating = False
        print(time.strftime("%H:%M:%S"), "Planning thread started.")
        
        while not self.stop:
            time.sleep(THREAD_SLEEP)
            odometry = self.odometry 
            if odometry is None:
                continue
            
            start = odometry.pose

            if self.path is None or len(self.path.poses) == 0:
                #frontier calculation
                if self.planning_method == 1:
                    self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap_inflated)
                    self.frontiers = self.explor.sort_frontiers_by_dist(self.gridmap_inflated, start, self.frontiers)
                
                elif self.planning_method == 2: 
                    self.frontiers = self.explor.find_inf_frontiers(self.gridmap_inflated)
                    
                elif self.planning_method == 3:
                    self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap_inflated)
                
 
            #path planning and goal selection 
            
            self.check_termination_condition()       
            if self.frontiers is None:
                continue
            
            # if path is already planned, check path conditions
            if self.path is not None and len(self.path.poses) > 0:
                
                # update the path that is displayed on the graph
                self.path_to_draw.poses = []
                self.path_to_draw.poses.append(start)
                self.path_to_draw.poses.extend(self.path.poses)
                
                # self.goal_is_still_a_frontier()
                
                self.check_if_shorter_path_exists()
                        
                self.check_obstacles_on_path()
                continue
            
            #path planning
            
            current_path = None
            
            if self.planning_method == 3:
                current_frontiers = self.frontiers
                if current_frontiers is None:
                    continue
                if len(current_frontiers) == 1:
                    self.goal_frontier
                    self.path = self.explor.plan_path(self.gridmap_planning, start, self.goal_frontier)
                    continue
                
                print(time.strftime("%H:%M:%S"), "Started planning the path with TSP.")
                # TSP formulation
                
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
                print(time.strftime("%H:%M:%S"), TSP_result)
                if len(TSP_result) < 2:
                    self.terminate_counts -= 1
                    print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
                    print(time.strftime("%H:%M:%S"), "TSP solution was not found.")
                    print(time.strftime("%H:%M:%S"), "TSP result was ", TSP_result)
                    continue
                
                # 3. assign the first frontier of the tour as the next exploration goal
                try:
                    goal_frontier = all_locations[TSP_result[1]]
                except:
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
                print(time.strftime("%H:%M:%S"), "Started planning the path.")
                for f in self.frontiers:
                    self.goal_frontier = f
                    '''try:
                        index_by_dist = frontiers_by_dist.index(f)
                        if num_frontier > 3 and index_by_dist > (num_frontier / 2):
                            print(time.strftime("%H:%M:%S"), "Frontier is too far away, its index is ", index_by_dist, " out of ", num_frontier)
                            continue
                        else:
                            print(time.strftime("%H:%M:%S"), "Frontiers index is ", index_by_dist, " out of ", num_frontier)
                    except:
                        print(time.strftime("%H:%M:%S"), "Frontiers index is not in the list")'''
                    
                    self.path = self.explor.plan_path(self.gridmap_planning, start, self.goal_frontier) 
                    simple_path = self.explor.simplify_path(self.gridmap_planning, self.path)
                    
                    if self.path is not None:       
                        
                        if simple_path is not None:
                            self.path = simple_path
                            
                        print(time.strftime("%H:%M:%S"), "Path was successfully planned.")
                        print(time.strftime("%H:%M:%S"), "Going to frontier at x: ", "{:.2f}".format(self.goal_frontier.position.x), ", y:", "{:.2f}".format(self.goal_frontier.position.y))
                        break
                        '''if len(self.path.poses) > 10:
                            print(time.strftime("%H:%M:%S"), "Path is too long. Rerouting")
                            # self.path = None
                        else:
                            print(time.strftime("%H:%M:%S"), "Path was successfully planned.")
                            print(time.strftime("%H:%M:%S"), "Going to frontier at x: ", "{:.2f}".format(self.goal_frontier.position.x), ", y:", "{:.2f}".format(self.goal_frontier.position.y))
                            break'''
                    else:
                        print(time.strftime("%H:%M:%S"), "No path was found.")
                if self.path is None:
                    print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
                    self.frontiers = []
                    self.terminate_counts -= 1
                                      
    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """ 
        print(time.strftime("%H:%M:%S"), "Trajectory thread started.")
        time.sleep(6*THREAD_SLEEP)
        while not self.stop:
            time.sleep(THREAD_SLEEP)
            
            if self.init_rotating:
                continue
            
            if self.API_navigation_goal is None:
                if self.path is not None and len(self.path.poses) > 0:
                    #fetch the new navigation goal                    
                    if len(self.path.poses) > 1:
                        nav_goal = self.path.poses[1]
                        #give it to the robot
                        if self.id == 0:
                            self.robot.goto(nav_goal)
                    # remove the previous navigation goal
                    prev_goal = self.path.poses.pop(0)
 
'''class ExternalExplorer:
    
    def __init__(self, robotID = 1): 
        self.robot = HexapodRobot.HexapodRobot(robotID)
        self.explor = HexapodExplorer.HexapodExplorer()
        
    def start(self):
        #turn on the robot 
        self.robot.turn_on()

        #start navigation thread
        self.robot.start_navigation()
        
    def sendMessages(self):'''
    
    
class Communication():
    
    def __init__(self, role_server=True):
        self.server = role_server
        self.client_socket = None
        self.external_robot = None
    
    def send_socket(self ):
        print(time.strftime("%H:%M:%S"), "Socket send thread started")
        message = "Hello, robot!"
        self.client_socket.send(message.encode()) 
        while True:
            time.sleep(0.5)
            
            if self.server:
                if ex0 is not None and ex0.gridmap is not None:
                    
                    if ex1 is not None and ex1.odometry_ is not None:
                        odom_valid = True
                        odometry_position = [ex1.odometry_.pose.position.x, ex1.odometry_.pose.position.y,]
                        odometry_orientation = [ex1.odometry_.pose.orientation.x, ex1.odometry_.pose.orientation.y, 
                                                ex1.odometry_.pose.orientation.z,
                                                ex1.odometry_.pose.orientation.w]
                    else:
                        odom_valid = False
                        odometry_position = [0, 0]
                        odometry_orientation = [0, 0, 0, 0]
                    
                    message = {"map" : ex0.gridmap.data.tolist(),
                               "odometry_valid" : odom_valid,
                               "odometry_position" : odometry_position,
                               "odometry_orientation" : odometry_orientation}
                    
                    message = json.dumps(message)
                    self.client_socket.send(message.encode()) 
            
        
    def receive_socket(self):
        print(time.strftime("%H:%M:%S"), "Socket receive thread started")
        received_data = self.client_socket.recv(4000)
        print(received_data.decode())
        while True:
            time.sleep(0.5)
            if not self.server:
                received_data = self.client_socket.recv(120000)
                print(time.strftime("%H:%M:%S"), "Received: ", received_data)
                try:
                    received_data = json.loads(received_data)
                    global ex0
                    if ex0 is not None:
                        ex0.gridmap.data = np.array(received_data['map'])
                        #print(time.strftime("%H:%M:%S"), "Received: ", received_data['map'])
                        
                        if received_data['odometry_valid']:
                            ex0.odometry = Odometry()
                            ex0.odometry.pose.position.x = received_data['odometry_position'][0]
                            ex0.odometry.pose.position.y = received_data['odometry_position'][1]
                            
                            ex0.odometry.pose.orientation.x = received_data['odometry_orientation'][0]
                            ex0.odometry.pose.orientation.y = received_data['odometry_orientation'][1]
                            ex0.odometry.pose.orientation.z = received_data['odometry_orientation'][2]
                            ex0.odometry.pose.orientation.w = received_data['odometry_orientation'][3]
                except:
                    print(time.strftime("%H:%M:%S"), "Couldnt decode json")
                    #print(received_data.decode())
            
            
    def start_socket_exchange(self ):
        if self.server:
            # Create a socket
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Bind the socket to a specific address and port
            server_socket.bind(('localhost', 32000))

            # Listen for incoming connections
            server_socket.listen()

            # Accept a connection
            self.client_socket, client_address = server_socket.accept()
            
            global ext_robot_detected
            ext_robot_detected = True
            
            '''#turn on the robot 
            self.external_robot = HexapodRobot.HexapodRobot(1)
            self.external_robot.turn_on()

            #start navigation thread
            self.external_robot.start_navigation()'''
        else:
            # Create a socket
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect to the server
            self.client_socket.connect(('localhost', 32000))
            
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
    parser.add_argument('--variant', type=int, default=1, choices=[1,2,3],required=False, help="Choose the planning variant.") 
    parser.add_argument('ID', type=int, help="The robot's ID starting from 0.") 
    
    args = parser.parse_args()
    planning_variant = args.variant
    print(time.strftime("%H:%M:%S"), "You chose the planning variant #",planning_variant)
    print(time.strftime("%H:%M:%S"), "Robot's id is #", args.ID)
    
    if args.ID == 0:
        role_server = True
    else:
        role_server = False
    cmn = Communication(role_server=role_server)
    cmn.start_socket_exchange()
    
    #instantiate the robot
    ex0 = Explorer(planning_var=planning_variant, robotID=args.ID)
    
    
    if args.ID != 0:
        #start the locomotion
        ex0.start()
    #time.sleep(THREAD_SLEEP)
    
    #continuously plot the map, targets and plan (once per second)
    fig, (ax, bx) = plt.subplots(nrows=2, ncols=1, figsize=(10,25))
    plt.ion()
    while not ex0.stop:
        
        if ext_robot_detected:
            #turn on the robot 
            
            ex1 = HexapodRobot.HexapodRobot(1)
            
            time.sleep(1)
            
            ex0.start()
            time.sleep(1)
            ex1.turn_on()
            time.sleep(1)
            #start navigation thread
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
    ex0.__del__()
    