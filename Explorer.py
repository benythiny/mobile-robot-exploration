#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

ROBOT_SIZE = 0.4
THREAD_SLEEP = 0.5
FRONTIER_DIST = 0.5

 
class Explorer:
    """ Class to represent an exploration agent
    """
    def __init__(self, robotID = 0):
 
 
        """ VARIABLES
        """ 
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
        
        # terminating condition
        self.terminate_counts = 10
        
        # planning methods
        self.planning_method = 2
        
        
        """Connecting the simulator
        """
        #instantiate the robot
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
        #turn off the robot
        self.robot.stop_navigation()
        self.robot.turn_off()
 
    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map
        """
        while not self.stop:
            time.sleep(THREAD_SLEEP)
               
            #get the current laser scan and odometry and fuse them to the map
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, self.robot.laser_scan_, self.robot.odometry_)
            
            #obstacle growing
            self.gridmap_inflated = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE)
                    
            #...
 
    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path  
        """
        time.sleep(4*THREAD_SLEEP)
        while not self.stop:
            time.sleep(THREAD_SLEEP)
 
            #frontier calculation
            if self.planning_method == 1:
                self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap_inflated)
            else: 
                self.frontiers = self.explor.find_inf_frontiers(self.gridmap_inflated)
 
            #path planning and goal selection
            odometry = self.robot.odometry_
            
            
            # if there are no more frontiers, return the app 
            if self.path is not None and self.frontiers is None:
                if self.terminate_counts == 0:
                    print(time.strftime("%H:%M:%S"), "No frontiers detected. Terminating the script.")
                    self.stop = True
                else:
                    self.terminate_counts -= 1
                    print(time.strftime("%H:%M:%S"), "Counts before termination: ", self.terminate_counts)
            else:
                # restore the counter
                self.terminate_counts = 10
                
            
            if odometry is None or self.frontiers is None:
                continue
            
            # check if the goal is still a frontier, erase the path otherwise    
            if self.path is not None:
                if len(self.path.poses) > 0:
                    if self.goal_frontier is not None:
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
                        
            # check if there ane no new obstacles on the planned path 
            if self.path is not None:
                if len(self.path.poses) > 0:
                    collision = False
                    for p in range(len(self.path.poses) - 1):
                        y, x = self.path.poses[p].position.y, self.path.poses[p].position.x 
                        b_start = self.explor.world_to_map(self.gridmap_inflated, np.array([y, x]))
                        
                        y, x = self.path.poses[p+1].position.y, self.path.poses[p+1].position.x 
                        b_end = self.explor.world_to_map(self.gridmap_inflated, np.array([y, x]))
                         
                        map2d = self.gridmap_inflated.data.reshape(self.gridmap_inflated.height, self.gridmap_inflated.width)
                        '''if map2d.data[y, x] > 0.5:'''
                        b_line = self.explor.bresenham_line(b_start, b_end)
                        #collision = self.collision_on_path(grid_map, b_line)
                        
                        
                        for (y, x) in b_line: #check for collision
                            if map2d[y,x] >= 0.5: 
                                collision = True
                                break
                    #if collision == False or len(b_line) < 2:
                    if collision == True:
                        # collision detected
                        print(time.strftime("%H:%M:%S"),"Collision detected on the planned path. Rerouting...")
                        self.path = None
                        self.robot.navigation_goal = None
            
            if self.path is None or len(self.path.poses) == 0:
                start = odometry.pose
                
                if self.planning_method == 1:
                    # P1: sort all frontiers by their distance to current position
                    # and return only feasible ones
                    self.frontiers = self.explor.sort_frontiers_by_dist(self.gridmap_inflated, start, self.frontiers)
                    
                    # pick the closest one
                    if len(self.frontiers) > 0:
                        self.goal_frontier = self.frontiers[0]
                        print(time.strftime("%H:%M:%S"), "Started planning the path.")
                        self.path = self.explor.plan_path(self.gridmap_inflated, start, self.goal_frontier) 
                        simple_path = self.explor.simplify_path(self.gridmap_inflated, self.path)
                        print(time.strftime("%H:%M:%S"), "Path was successfully planned.")
                        if simple_path != None:
                            #print("... and its not None")
                            self.path = simple_path
                        if len(self.path.poses) > 10:
                            print(time.strftime("%H:%M:%S"), "Path is too long. Rerouting")
                            self.path = None
                            
                    else: 
                        # if list is empty, return the app - no more frontiers
                        if self.terminate_counts == 0:
                            print("No frontiers detected. Terminating the script.")
                            self.stop = True
                        else:
                            self.terminate_counts -= 1
                            print("Counts before termination: ", self.terminate_counts)
                            
                elif self.planning_method == 2:
                    # sort frontiers by information gain
                    # Sort the list of tuples based on the second value
                    if self.frontiers is not None:
                        
                        for f in self.frontiers:
                            self.goal_frontier = f
                            print(time.strftime("%H:%M:%S"), "Started planning the path.")
                            self.path = self.explor.plan_path(self.gridmap_inflated, start, self.goal_frontier) 
                            simple_path = self.explor.simplify_path(self.gridmap_inflated, self.path)
                            
                            if self.path is not None:       
                                print(time.strftime("%H:%M:%S"), "Path was successfully planned.")
                                if simple_path is not None:
                                    self.path = simple_path
                                if len(self.path.poses) > 10:
                                    print(time.strftime("%H:%M:%S"), "Path is too long. Rerouting")
                                    self.path = None
                                else:
                                    break
            
                       
                        
    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """ 
        time.sleep(6*THREAD_SLEEP)
        while not self.stop:
            #...
            time.sleep(THREAD_SLEEP)
            if self.robot.navigation_goal is None:
                if self.path is not None and len(self.path.poses) > 0:
                    
                    #fetch the new navigation goal                    
                    if len(self.path.poses) > 1:
                        nav_goal = self.path.poses[1]
                    
                        #give it to the robot
                        self.robot.goto(nav_goal)
                        #print(time.strftime("%H:%M:%S"),"Going to:")
                        # print(nav_goal)
                    
                    prev_goal = self.path.poses.pop(0)
                    
                    #print("Current self position: ", self.robot.odometry_.pose)
                    #print("Going to: ", nav_goal)
 
 
if __name__ == "__main__":
    
    #instantiate the robot
    ex0 = Explorer()
    
    #start the locomotion
    ex0.start()
    time.sleep(10*THREAD_SLEEP)
    
    #continuously plot the map, targets and plan (once per second)
    fig, (ax, bx) = plt.subplots(nrows=2, ncols=1, figsize=(10,25))
    plt.ion()
    while not ex0.stop:
        plt.cla()
        ax.cla()
        bx.cla()
        
        #plot the map
        ex0.gridmap.plot(ax)
        ex0.gridmap_inflated.plot(bx)
        
        if ex0.frontiers != None:
            for frontier in ex0.frontiers: #frontiers
                if type(frontier) != Pose:
                    frontier = frontier[0]
                ax.scatter(frontier.position.x, frontier.position.y,c='red')
        
        if ex0.path is not None: #print path points
            for pose in ex0.path.poses:
                ax.scatter(pose.position.x, pose.position.y,c='green', s=50, marker='x')
        
        # add current robot's pose
        ax.scatter(ex0.robot.odometry_.pose.position.x, ex0.robot.odometry_.pose.position.y,c='blue', s = 200)
        bx.scatter(ex0.robot.odometry_.pose.position.x, ex0.robot.odometry_.pose.position.y,c='blue', s = 200)
        
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
    
        #to throttle the plotting pause for 1s
        plt.pause(THREAD_SLEEP)
    ex0.__del__()
    