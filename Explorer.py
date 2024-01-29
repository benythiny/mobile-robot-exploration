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

ROBOT_SIZE = 0.5
THREAD_SLEEP = 0.5
 
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
        
        #prepare the gridmap
        self.gridmap = OccupancyGrid()
        self.gridmap.resolution = 0.1
        self.gridmap.width = 100
        self.gridmap.height = 100
        self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height*self.gridmap.width))
        
        # map for planning with obstacles
        self.gridmap_inflated = OccupancyGrid()
        
        
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
            self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap_inflated)
 
            #path planning and goal selection
            odometry = self.robot.odometry_
            
            if odometry is None or self.frontiers is None:
                continue
            
            if self.path is None or len(self.path.poses) == 0:
                start = odometry.pose
                # sort all frontiers by their distance to current position
                # and return only feasible ones
                self.frontiers = self.explor.sort_frontiers_by_dist(self.gridmap_inflated, start, self.frontiers)
                
                # pick the closest one
                if len(self.frontiers) > 0:
                    end = self.frontiers[0]
                    self.path = self.explor.plan_path(self.gridmap_inflated, start, end) 
                    self.path = self.explor.simplify_path(self.gridmap_inflated, self.path)
                else: 
                    # if list is empty, return the app - no more frontiers
                    print("No frontiers detected. Terminating the script.")
                    self.stop = True
                        
                        
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
                        print(time.strftime("%H:%M:%S"),"Going to:")
                        print(nav_goal)
                    
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
    fig, (ax, bx) = plt.subplots(nrows=2, ncols=1, figsize=(10,15))
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
 
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
    
        #to throttle the plotting pause for 1s
        plt.pause(THREAD_SLEEP)
    ex0.__del__()
    