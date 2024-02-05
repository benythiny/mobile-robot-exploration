#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import socket
import json
import logging

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

ROBOT_SIZE     = 0.4
THREAD_SLEEP   = 0.4
FRONTIER_DIST  = 0.5

SOCKET_PORT    = 32000

GRIDMAP_WIDTH  = 100
GRIDMAP_HEIGHT = 100

ex0 = None
ex1 = None

ext_robot_detected = False
 
class Explorer:
    """ Class to represent an exploration agent
    """
    def __init__(self, robotID = 0):
 
 
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
        self.gridmap.width = GRIDMAP_WIDTH
        self.gridmap.height = GRIDMAP_HEIGHT
        self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height*self.gridmap.width))
        
        # map for planning with obstacles
        self.gridmap_inflated = OccupancyGrid()
        
        # map for path planning only 
        self.gridmap_planning = OccupancyGrid()
        
        # terminating condition
        self.terminate_counts = 20
        
        # path that is displayed on the graph
        self.path_to_draw = Path()
        
        self.odometry = None
        self.laser_scan = None
        self.API_navigation_goal = None
        self.current_goal = None
        self.collision_on_path = False
        self.other_robot_pos = None
        
        #mutex for access to variables exchanged via communication  
        self.lock_gridmap    = thread.Lock()
        self.lock_odometry   = thread.Lock() 
        self.lock_laser_scan = thread.Lock()
          
        
        """Connecting the simulator
        """
        
        #instantiate the robot if the single robot exploration is chosen
        self.robot = HexapodRobot.HexapodRobot(robotID)
        #and the explorer
        self.explor = HexapodExplorer.HexapodExplorer()
                
    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning
        """
        #turn on the robot 
        self.robot.turn_on()

        #start navigation thread
        self.robot.start_navigation()
 
        #start trajectory following
        try:
            traj_follow_thread = thread.Thread(target=self.trajectory_following)
            traj_follow_thread.start() 
        except:
            print("Error: unable to start trajectory thread")
            sys.exit(1)
 
    def __del__(self):
        #turn off the robot
        self.robot.navigation_goal = None
        time.sleep(0.2)
        self.robot.stop_navigation()
        self.robot.turn_off()
       
    def check_termination_condition(self):
        """
        method to monitor the amount of self.terminate_counts left before terminating the script.
        the amount of self.terminate_counts is reduced every time there are no feasible frontiers.
        """   
        if self.terminate_counts <= 0:
                print(time.strftime("%H:%M:%S"), "| id:", self.id, "| ", "No feasible frontiers detected. Terminating the script.")
                self.stop = True
            
    def check_obstacles_on_path(self):
        """
        method to continuosuly monitor presence of obstacles on the current path using the Bresenham line.
        in case of obstacles detection, the current path is removed. 
        """
        # check if there are no new obstacles on the planned path 
        collision = False
         
        odometry = self.robot.odometry_
         
            
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
            print(time.strftime("%H:%M:%S"), "| id:", self.id, "| ", "Collision detected on the planned path. Rerouting...")
            
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
        
         
        odometry = self.robot.odometry_
         
        
        if odometry is None:
            return
        
        print(time.strftime("%H:%M:%S"), "| id:", self.id, "| ", "Path is a bit too long, trying to find a shorter one...")
            
        start = odometry.pose
        alternative_path = self.explor.plan_path(self.gridmap_planning, start, self.goal_frontier)
        alternative_path = self.explor.simplify_path(self.gridmap_planning, alternative_path)
        
        if alternative_path is None:
            return
        
        if len(alternative_path.poses) < (orig_path_length - 1):
            print(time.strftime("%H:%M:%S"), "| id:", self.id, "| ", "A shorter path was found. Reformating path.")
            self.path = alternative_path
                                            
    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """ 
        time.sleep(6*THREAD_SLEEP)
        print(time.strftime("%H:%M:%S"), "| id:", self.id, "| ","Trajectory thread started.")
        
        while not self.stop:
            time.sleep(2 * THREAD_SLEEP)
            
            if self.path is not None and self.gridmap_inflated.data is not None:
                self.check_obstacles_on_path()
                
            if self.path is not None and self.gridmap_inflated.data is not None:
                self.check_if_shorter_path_exists()
            
            if self.robot.navigation_goal is None:
                if self.path is not None and len(self.path.poses) > 0:
                    #fetch the new navigation goal                    
                    if len(self.path.poses) > 1:
                        nav_goal = self.path.poses[1]
                        self.current_goal = nav_goal
                        #give it to the robot, if the robot is directly connected to API, send command
                        self.robot.goto(nav_goal)
                    # remove the previous navigation goal
                    prev_goal = self.path.poses.pop(0)
  

class MultiExplorer:
    """ Class to represent an exploration agent
    """
    def __init__(self):
 
 
        """ VARIABLES
        """ 
        
        #current frontiers
        self.frontiers = None
 
        #stopping condition
        self.stop = False
        
        # define frontiers
        self.frontiers = None
        self.goal_frontier = None
        
        #prepare the gridmap
        self.gridmap = OccupancyGrid()
        self.gridmap.resolution = 0.1
        self.gridmap.width = GRIDMAP_WIDTH
        self.gridmap.height = GRIDMAP_HEIGHT
        self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height*self.gridmap.width))
        
        # map for planning with obstacles
        self.gridmap_inflated = OccupancyGrid()
        
        # map for path planning only 
        self.gridmap_planning = OccupancyGrid()
        
        # terminating condition
        self.terminate_counts = 30        
        
        """Connecting the simulator
        """
        self.explor = HexapodExplorer.HexapodExplorer()
       
    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning
        """
        
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
             
    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map
        """
        print(time.strftime("%H:%M:%S"), "| ", "Started the mapping thread") 
        while not self.stop:
            time.sleep(THREAD_SLEEP)
            global ex0, ex1
            robots = [ex0, ex1]
            #get the current laser scan and odometry and fuse them to the map
            for robot in robots:
                odometry = robot.robot.odometry_
                laser_scan = robot.robot.laser_scan_
                
                if odometry is None or laser_scan is None:
                    continue
            
                self.gridmap = self.explor.fuse_laser_scan(self.gridmap, laser_scan, odometry)
                
                if self.gridmap is None:
                    continue
            
                # add the other robot to the infused map as an obstacle if its position is known
                gridmap_with_other_robot = copy.deepcopy(self.gridmap)
                if robot == ex0:
                    other_robot = ex1
                else:
                    other_robot = ex0
                    
                if other_robot.robot.odometry_ is not None:
                    y, x = other_robot.robot.odometry_.pose.position.y, other_robot.robot.odometry_.pose.position.x
                    [y, x] = self.explor.world_to_map(self.gridmap, np.array([y, x]))
                    gridmap_with_other_robot.data[y * self.gridmap.width + x] = 1
                
                #obstacle growing for the map used for continous collision avoidance
                self.gridmap_inflated = self.explor.grow_obstacles(gridmap_with_other_robot, ROBOT_SIZE)
                
                # obstacle growing for the map used for path planning
                self.gridmap_planning = self.explor.grow_obstacles(gridmap_with_other_robot, ROBOT_SIZE + 0.1)
      
                robot.gridmap = self.gridmap
                robot.gridmap_inflated = self.gridmap_inflated
                robot.gridmap_planning = self.gridmap_planning
                
    def check_termination_condition(self):
        """
        method to monitor the amount of self.terminate_counts left before terminating the script.
        the amount of self.terminate_counts is reduced every time there are no feasible frontiers.
        """ 
        global ex0, ex1
        # if there are no more frontiers, return the app 
        if self.frontiers is None:
            self.terminate_counts -= 1
            print(time.strftime("%H:%M:%S"), "| ", "No feasible frontiers detected. Counts before termination: ", self.terminate_counts)
            
        if self.terminate_counts <= 0 or ex0.stop or ex1.stop:
                print(time.strftime("%H:%M:%S"),  "| ", "No feasible frontiers detected. Terminating the script.")
                self.stop = True
                ex0.stop = True
                ex1.stop = True
                 
    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path  
        """
        time.sleep(4*THREAD_SLEEP)
        print(time.strftime("%H:%M:%S"), "| ","Planning thread started.")
        
        while not self.stop:
            time.sleep(THREAD_SLEEP)
            global ex0, ex1
            robots = [ex0, ex1]
            id = 0
            #get the current laser scan and odometry and fuse them to the map
            for robot in robots:
                odometry = robot.robot.odometry_
                if odometry is None:
                    continue
            
                # get current robot's position
                start = odometry.pose

                #frontier calculation when there is no planned path
                if robot.path is None or len(robot.path.poses) == 0:
                    
                    # implementation of task: f1 + f2 extenstion, preparation for task p1

                    self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap_inflated)
                    self.frontiers = self.explor.sort_frontiers_by_dist(self.gridmap_inflated, start, self.frontiers)
                    robot.frontiers = self.frontiers
                # after detecting frontiers, check if there are any left
                self.check_termination_condition()       
                if self.frontiers is None or self.stop:
                    continue
            
                # if path is already planned, check path conditions
                if robot.path is not None and len(robot.path.poses) > 0:
                    
                    # extend the path that is displayed on the graph with the current robot's position
                    robot.path_to_draw.poses = []
                    robot.path_to_draw.poses.append(start)
                    robot.path_to_draw.poses.extend(robot.path.poses)
                    
                    #robot.check_if_shorter_path_exists()
                            
                    #robot.check_obstacles_on_path()
                    continue
                
                # implementation of tasks: p1 the frontiers are already sorted in a desired way
                
                print(time.strftime("%H:%M:%S"), "| id:",  id, "| ","Started planning the path.")
                
                if robot == ex0:
                    other_robot = ex1
                else:
                    other_robot = ex0
                
                # go through all frontiers and try to plan path to the first one in the list, which is
                # the closest one to the robot in case of p1
                # if the path couldn't be planned, go to the next one
                for f in self.frontiers:
                    # if the other robot already goes to this frontier, pick the next one
                    if other_robot.goal_frontier is not None:
                        dist = (f.position - other_robot.goal_frontier.position).norm()
                        if dist < 0.2:
                            print(time.strftime("%H:%M:%S"), "| id:",  id, "| ","The other robot already picked this goal, trying another frontier.")
                            continue
                    
                    goal_frontier = f
                    robot.path   = self.explor.plan_path(self.gridmap_planning, start, goal_frontier) 
                    simple_path = self.explor.simplify_path(self.gridmap_planning, robot.path)
                    
                    if robot.path is not None:  
                        robot.goal_frontier = goal_frontier     
        
                        if simple_path is not None:
                            robot.path = simple_path
                            
                        print(time.strftime("%H:%M:%S"), "| id:",  id, "| ","Path was successfully planned.")
                        print(time.strftime("%H:%M:%S"), "| id:",  id, "| ","Going to frontier at x:", "{:.2f}".format(robot.goal_frontier.position.x), 
                                                                            ", y:", "{:.2f}".format(robot.goal_frontier.position.y))
                        
                        self.terminate_counts = 30
                        break
                        
                    else:
                        print(time.strftime("%H:%M:%S"), "| id:",  id, "| ","No path was found.")
                        
                # when no path could be planned for any of the frontiers, increase the termination counts
                if robot.path is None:
                    
                    self.frontiers = []
                    self.terminate_counts -= 1
                    print(time.strftime("%H:%M:%S"), "| id:",  id, "| ","Counts before termination: ", self.terminate_counts)
                id+=1                       
 
        
if __name__ == "__main__":
    
    #instantiate the robots
    ex0 = Explorer(robotID=0)
    ex1 = Explorer(robotID=1)
    
    #start the locomotion
    ex0.start()
    ex1.start()
    
    time.sleep(5)
    
    ex_multi = MultiExplorer()
    ex_multi.start()
    
    #continuously plot the map, targets and plan (once per second)
    fig, (ax, bx, cx, dx) = plt.subplots(nrows=4, ncols=1, figsize=(10,25))
    plt.ion()
    while not ex_multi.stop:
        plt.cla()
        ax.cla()
        bx.cla()
        cx.cla()
        dx.cla()
        
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
        ax.scatter(ex0.robot.odometry_.pose.position.x, ex0.robot.odometry_.pose.position.y,c='blue', s = 200)
        bx.scatter(ex0.robot.odometry_.pose.position.x, ex0.robot.odometry_.pose.position.y,c='blue', s = 200)
        
        """ __________________  """
        
        #plot the map
        ex1.gridmap.plot(cx)
        ex1.gridmap_inflated.plot(dx)
        
        # print all detected frontiers as red dots
        if ex1.frontiers is not None:
            for frontier in ex1.frontiers: 
                cx.scatter(frontier.position.x, frontier.position.y,c='red')
                
        #print simplified path points
        if ex1.path is not None and len(ex1.path.poses) > 0: 
            for pose in ex1.path.poses:
                cx.scatter(pose.position.x, pose.position.y,c='green', s=100, marker='x')
                dx.scatter(pose.position.x, pose.position.y,c='green', s=100, marker='x')
            
        #print the whole path as connected points
        if len(ex1.path_to_draw.poses) > 0:
            ex1.path_to_draw.plot(ax=cx, style = 'point')
            ex1.path_to_draw.plot(ax=dx, style = 'point')
            
        
        # draw current robot's pose
        cx.scatter(ex1.robot.odometry_.pose.position.x, ex1.robot.odometry_.pose.position.y,c='blue', s = 200)
        dx.scatter(ex1.robot.odometry_.pose.position.x, ex1.robot.odometry_.pose.position.y,c='blue', s = 200)
        
        
        
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
    
        #to throttle the plotting pause for 1s
        plt.pause(THREAD_SLEEP)
    ex0.__del__()
    ex1.__del__()