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

ROBOT_SIZE = 0.3
 
class Explorer:
    """ Class to represent an exploration agent
    """
    def __init__(self, robotID = 0):
 
 
        """ VARIABLES
        """
        #occupancy grid map of the robot ... possibly extended initialization needed in case of 'm1' assignment
        self.gridmap = OccupancyGrid()
        self.gridmap.resolution = 0.1
 
        #current frontiers
        self.frontiers = None
 
        #current path
        self.path = None
 
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
            #fuse the laser scan   
            #get the current laser scan and odometry and fuse them to the map
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, self.robot.laser_scan_, self.robot.odometry_)
 
            #...
 
    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path  
        """
        while not self.stop:
            #obstacle growing
            gridmap_inflated = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE)
 
            #frontier calculation
            self.frontiers = self.explor.find_free_edge_frontiers(gridmap_inflated)
 
            #path planning and goal selection
            odometry = self.robot.odometry_
            #...
            # self.path = Path()
            if odometry is not None and self.frontiers is not None:
                start = odometry.pose
                # todo: add closest frontier search
                end = self.frontiers[0]
                
                self.path = self.explor.plan_path(gridmap_inflated, start, end)
                #self.path = self.explor.simplify_path(gridmap_inflated, self.path)
 
    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """ 
        while not self.stop:
            #...
            if self.robot.navigation_goal is None:
                if self.path is not None:
                    #fetch the new navigation goal
                    #nav_goal = path_nav.pop(0)
                    
                    nav_goal = self.path.poses.pop(0)
                    #give it to the robot
                    self.robot.goto(nav_goal)
                    time.sleep(1)
            #...
 
 
if __name__ == "__main__":
    
    #instantiate the robot
    ex0 = Explorer()
    
    #start the locomotion
    ex0.start()
    
    #continuously plot the map, targets and plan (once per second)
    fig, ax = plt.subplots()
    plt.ion()
    while(1):
        plt.cla()
        
        #plot the map
        ex0.gridmap.plot(ax)
        
        if ex0.frontiers != None:
            for frontier in ex0.frontiers: #frontiers
                if type(frontier) != Pose:
                    frontier = frontier[0]
                ax.scatter(frontier.position.x, frontier.position.y,c='red')
        
        if ex0.path is not None: #print simple path points
            for pose in ex0.path.poses:
                ax.scatter(pose.position.x, pose.position.y,c='green', s=150, marker='x')
                
        '''    
        #plot the gridmap
        if ex0.gridmap.data is not None:
            ex0.gridmap.plot(ax)
        #plot the navigation path
        if ex0.path is not None:
            ex0.path.plot(ax)'''
 
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
    
        #to throttle the plotting pause for 1s
        plt.pause(1)