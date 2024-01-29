#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import math
import time
from heapq import heappop, heappush
from queue import PriorityQueue
from warnings import warn
 
import numpy as np
import copy
 
# cpg network
import cpg.oscilator_network as osc
from scipy.ndimage import distance_transform_edt
 
# import messages
from messages import *
 
import matplotlib.pyplot as plt
 
import scipy.ndimage as ndimg
from sklearn.cluster import KMeans
 
import skimage.measure
 
import collections
import heapq
import skimage.measure as skm
 
 
class HexapodExplorer:
 
    def __init__(self):
        pass
 
    def fuse_laser_scan(self, grid_map, laser_scan, odometry):
        """ Method to fuse the laser scan data sampled by the robot with a given 
            odometry into the probabilistic occupancy grid map
        Args:
            grid_map: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update = copy.deepcopy(grid_map)
 
        if grid_map is None or laser_scan is None or odometry is None:
            return grid_map_update
 
        current_x = odometry.pose.position.x
        current_y = odometry.pose.position.y
        current_ori = odometry.pose.orientation.to_Euler()[0]  # z
 
        grid_map_update = self.filter_scan_msgs(laser_scan, current_x, current_y, current_ori, grid_map)
 
        '''
        #add points to map
        # Calculate the flat indices for the valid points
        flat_indices = points[:, 1] * grid_map.height + points[:, 0]
 
        # Assign the value 1 to the valid points in the modified_grid_map
        grid_map_update.data[flat_indices] = 1
        '''
        # TODO:[t1d_map] fuse the correctly aligned laser data into the probabilistic occupancy grid map
 
        return grid_map_update
 
    def filter_scan_msgs(self, scan_msg, current_x, current_y, current_ori, grid_map):
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
 
        N = len(distances)
 
        xy_projections = np.zeros((N, 2))
        # obs_angles = np.arange(start=angle_min, stop=angle_max, step=angle_incr)
        obs_angles = np.arange(0, N, 1) * angle_incr + angle_min
        distances = np.array(distances)
 
        # 1. Project the laser scan points to x,y plane with respect to the robot heading
        # + 2. Compensate for the robot odometry
        xy_projections[:, 1] = current_x + distances * np.cos(current_ori + obs_angles)
        xy_projections[:, 0] = current_y + distances * np.sin(current_ori + obs_angles)
 
        # 3. Transfer the points from the world coordinates to the map coordinates
        '''
        resolution = grid_map.resolution
 
        origin_x = grid_map.origin.position.x
        origin_y = grid_map.origin.position.y
 
        origin_xy = np.array([origin_x, origin_y])
 
        m = np.round((xy_projections - origin_xy) / resolution)
        '''
 
        m = self.world_to_map(grid_map, xy_projections)
 
        map_width = grid_map.width
        map_height = grid_map.height
 
        # Create a boolean mask to identify coordinates within the map
        within_map = (m[:, 0] >= 0) & (m[:, 0] < map_width) & (m[:, 1] >= 0) & (m[:, 1] < map_height)
 
        # Filter the coordinates using the boolean mask
        laser_scan_points_map = m[within_map].astype(np.int32)
 
        # 4.
        # get the position of the robot in the map coordinates
        odom_map = self.world_to_map(grid_map, np.array([current_y, current_x]))
 
        free_points = []
        occupied_points = []
 
        for pt in laser_scan_points_map:
            pt = tuple(pt)
            # raytrace the points
            pts = self.bresenham_line(odom_map, pt)
            #TODO: add check that there are no collisions in the line
 
            # save the coordinate of free space cells
            free_points.extend(pts[:-1])
            # save the coordinate of occupied cell
            occupied_points.append(pt)
        unique_free_point = list(set(free_points))
        unique_occupied_points = list(set(occupied_points))
 
        updated_map = self.bayes_update(grid_map, unique_free_point, unique_occupied_points)
 
        return updated_map
 
    def bayes_update(self, grid_map, free_points, occupied_points):
        grid_map_update = copy.deepcopy(grid_map)
 
        for point in free_points:
            p_occupied = grid_map.data[grid_map.height * point[0] + point[1]]
            p_free = 1 - p_occupied
            p_z_m = (1 + self.update_free(point, free_points) - self.update_occupied(point, occupied_points)) / 2
            p_z_free = 1 - p_z_m
            p = (p_z_m * p_occupied) / (p_z_m * p_occupied + p_z_free * p_free)
            grid_map_update.data[grid_map.height * point[0] + point[1]] = p
 
        for point in occupied_points:
            p_occupied = grid_map.data[grid_map.height * point[0] + point[1]]
            p_free = 1 - p_occupied
            p_z_m = (1 + self.update_free(point, free_points) - self.update_occupied(point, occupied_points)) / 2
            p_z_free = 1 - p_z_m
            p = (p_z_m * p_occupied) / (p_z_m * p_occupied + p_z_free * p_free)
            grid_map_update.data[grid_map.height * point[0] + point[1]] = p
 
        return grid_map_update
 
    def update_free(self, point, free_points):
        if point in free_points:
            p = 0
        else:
            p = 0.95
        return p
 
    def update_occupied(self, point, occupied_points):
        if point in occupied_points:
            p = 0
        else:
            p = 0.95
        return p
    
    def world_to_map(self, grid_map, world_coords):
        resolution = grid_map.resolution
 
        origin_x = grid_map.origin.position.x
        origin_y = grid_map.origin.position.y
 
        origin_xy = np.array([origin_x, origin_y])
 
        map_coords = np.round((world_coords - origin_xy) // resolution)
        return map_coords.astype(np.int32)
 
    def map_to_world(self, grid_map, map_coords):
        resolution = grid_map.resolution
 
        origin_x = grid_map.origin.position.x
        origin_y = grid_map.origin.position.y
 
        origin_xy = np.array([origin_x, origin_y])
 
        world_coords = (map_coords + 0.5) * resolution + origin_xy
        '''
        new_x = (map_coords[0] + 0.5) * resolution + origin_x
        new_y = (map_coords[1] + 0.5) * resolution + origin_y
 
        world_coords = (new_x, new_y)
        '''
        return world_coords
 
    def bresenham_line(self, start, goal):
        """Bresenham's line algorithm
        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            interlying points between the start and goal coordinate
        """
        (x0, y0) = start
        (x1, y1) = goal
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = int(x0), int(y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        x = goal[0]
        y = goal[1]
        line.append((x, y))
        return line
 
    def find_free_edge_frontiers(self, grid_map):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """
 
        if grid_map is None:
            return None
 
        # TODO:[t1e_expl] find free-adges and cluster the frontiers
        h = grid_map.height
        w = grid_map.width
        grid_data = grid_map.data.reshape((h, w))
 
        # transform the map:
        # assign 0, if P(x=occupied) = 1
        # assign 10, if 1 > P(x=occupied) >= 0.5
        # assign 1 otherwise
        transformed_map = np.where(grid_data == 1, 0, np.where((grid_data >= 0.5) & (grid_data < 1), 10, 1))
 
        # convolve the map
        mask = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]])
        data_c = ndimg.convolve(transformed_map, mask, mode='constant', cval=0.0)
 
        # filter the map with 2 thresholds
        lower_thr = 10
        higher_thr = 80
        free_edge_cells = ((data_c > lower_thr) & (data_c < higher_thr)).astype(int)
        free_coordinates = np.argwhere(free_edge_cells == 1)
 
        # check that the potential free cells are actually free
        for coord in free_coordinates:
            if grid_data[coord[0], coord[1]] >= 0.5:
                free_edge_cells[coord[0], coord[1]] = 0
 
        # Free-edge clustering
        labeled_image, num_labels = skm.label(free_edge_cells, connectivity=2, return_num=True)
 
        # Free-edge centroids
        free_cells = []
        for label in range(1, num_labels + 1):
            # Extract the coordinates of the labeled region (connected component)
            region = np.argwhere(labeled_image == label)
 
            # Calculate the centroid as the mean of x and y coordinates
            centroid_x = np.mean(region[:, 1])
            centroid_y = np.mean(region[:, 0])
            cell = self.map_to_world(grid_map, np.array([centroid_y, centroid_x]))
            free_cells.append(Pose(Vector3(cell[1], cell[0], 0), Quaternion(1, 0, 0, 0)))
        '''
         
        for coord in free_coordinates:
            if grid_data[coord[0], coord[1]] < 0.5:
                cell = self.map_to_world(grid_map, np.array(coord))
                free_cells.append(Pose(Vector3(cell[1], cell[0], 0), Quaternion(1, 0, 0, 0)))
        '''
 
        if len(free_cells) != 0:
            return free_cells
        else:
            return None
 
    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """
 
        # TODO:[project] find the information rich points in the environment
        return None
 
 
    def grow_obstacles(self, grid_map_original, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        """
 
        grid_map = copy.deepcopy(grid_map_original)
        h = grid_map.height
        w = grid_map.width
        res = grid_map.resolution
        thr = 0.5  # threshold for occupied cells
 
        # calculate the robot size in scale of grid cells
        robot_size_cells = math.ceil(robot_size / res)
 
        # reshape into a 2d array
        grid_data = grid_map.data.reshape((h, w))
 
        # get distance transform of the occupied cells
        dist_transform = distance_transform_edt(grid_data <= thr)
 
        # update the map, assign 1 when cells are within the robot's radius of obstacles
        grid_data[robot_size_cells > dist_transform] = 1
        grid_map.data = grid_data.reshape(h * w, )
 
        return grid_map
 
    '''
    def plan_path(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """
 
        h = grid_map.height
        w = grid_map.width
        res = grid_map.resolution
 
        path = Path()
        # Add the start pose
        path.poses.append(start)
 
        # Convert the grid_map data to a NumPy array
        grid_data = grid_map.data.reshape((h, w))
 
        # Define 8 possible movement directions (8-neighborhood)
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
 
        # Create a list of open cells to explore
        open_cells = [(0, (start.position.x, start.position.y))]
        # Create a dictionary to store the cost to reach each cell
        cell_costs = {(start.position.x, start.position.y): (0, (start.position.x, start.position.y))}
 
        while open_cells:
            # Get the cell with the lowest cost from the open_cells list
            cost, current = heappop(open_cells)
 
            # Check if we have reached the goal
            if current == (goal.position.x, goal.position.y):
                # Reconstruct the path
                path.poses = [start]  # Reset the path
                while current in cell_costs:
                    path.poses.append(Pose(Vector3(current[0], current[1], 0), Quaternion(1, 0, 0, 0)))
                    current = cell_costs[current][1]  # Get the previous cell
                path.poses.append(goal)
 
                return path
 
            # Explore neighboring cells
            for dx, dy in directions:
                x, y = self.world_to_map(grid_map, np.array([current[0] + dx * res, current[1] + dy * res]))
                x = int(x)
                y = int(y)
 
                neighbor = (current[0] + dx * res, current[1] + dy * res)
 
                if 0 <= x < w and 0 <= y < h and grid_data[y][x] == 0:
                    # Calculate the cost to reach the neighbor
                    neighbor_cost = cell_costs[current][0] + 1  # Assuming uniform cost
 
                    if neighbor not in cell_costs or neighbor_cost < cell_costs[neighbor][0]:
                        cell_costs[neighbor] = (neighbor_cost, current)
                        heappush(open_cells, (
                            neighbor_cost + self.heuristic(neighbor, (goal.position.x, goal.position.y)), neighbor))
 
        # If no path is found, return None
        return None
 
    def heuristic(self, current, goal):
        """ A simple Euclidean distance heuristic for A* """
        return np.linalg.norm(np.array([current[0], current[1]]) - np.array([[goal[0], goal[1]]]))
 
    '''
 
    def plan_path(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """
 
        path = Path()
        path.poses.append(goal)
 
        h = grid_map.height
        w = grid_map.width
 
        # Convert the grid_map data to a NumPy array
        grid_data = grid_map.data.reshape((h, w))
 
        map_start = tuple(self.world_to_map(grid_map, np.array([start.position.y, start.position.x])))
        map_goal = tuple(self.world_to_map(grid_map, np.array([goal.position.y, goal.position.x])))
        found_path = astar(grid_data, map_start, map_goal)
 
        if found_path is None:
            print("No path was found.")
            return None
        else:
            for point in found_path:
                [new_x, new_y] = self.map_to_world(grid_map, np.array(point))
                new_point = Pose(Vector3(new_y, new_x, 0), Quaternion(0, 0, 0, 0))
                path.poses.append(new_point)
        #path.poses[0] = goal
        #path.poses[-1] = start
        path.poses.append(start)
        path.poses = path.poses[::-1]
 
        return path
 

    def collision_on_path(self, map, line):
        h = map.height
        w = map.width
 
        # Convert the grid_map data to a NumPy array
        map = map.data.reshape((h, w))
 
        line_array = np.array(line)
 
        # Check if the corresponding cells in grid_data have a value of 0.5 or less
        cell_values = map[line_array[:, 1], line_array[:, 0]]
 
        # Check if any of the cell values are 0.5 or less
        result = np.all(cell_values <= 0.5)
 
        return not result
 
    def simplify_path(self, grid_map, path_orig):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map is None or path_orig is None:
            return None
 
        path_simplified = Path()
        # add the start pose
        path_simplified.poses.append(path_orig.poses[0])
 
        # iterate through the path and simplify the path
        goal = path_orig.poses[-1]
        previous_pose = path_orig.poses[0]
        previos_outer_pose = path_orig.poses[0]
        '''
        end = False
        idx = 1
        while not end:
            rest_of_path = path_orig.poses[idx: ]
            for new_pose in rest_of_path:
                b_end = self.world_to_map(grid_map, np.array([new_pose.position.y, new_pose.position.x]))
                b_start = self.world_to_map(grid_map, np.array([previos_outer_pose.position.y, previos_outer_pose.position.x]))
                b_line = self.bresenham_line(b_start, b_end)
 
                collision = self.collision_on_path(grid_map, b_line)
                if not collision:
                    previous_pose = new_pose
 
                    if previous_pose == goal:
                        end = True
                else:
                    path_simplified.poses.append(previous_pose)
                    #previous_pose = new_pose
                    previos_outer_pose = new_pose
                    break
                idx += 1
        path_simplified.poses.append(goal)
        return path_simplified
        '''
 
        for new_pose in path_orig.poses[1:]:
            b_end = self.world_to_map(grid_map, np.array([new_pose.position.y, new_pose.position.x]))
            b_start = self.world_to_map(grid_map, np.array([previos_outer_pose.position.y, previos_outer_pose.position.x]))
            b_line = self.bresenham_line(b_start, b_end)
            collision = self.collision_on_path(grid_map, b_line)
 
            if not collision:
                previous_pose = new_pose
            else:
                path_simplified.poses.append(previous_pose)
                previos_outer_pose = previous_pose
                previous_pose = new_pose
 
        path_simplified.poses.append(goal)
        return path_simplified
 

 
    ###########################################################################
    # INCREMENTAL Planner
    ###########################################################################
 
    def plan_path_incremental(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """
 
        if not hasattr(self, 'rhs'):  # first run of the function
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.g = np.full((grid_map.height, grid_map.width), np.inf)
 
        # TODO:[t1x-dstar] plan the incremental path between the start and the goal Pose
 
        return self.plan_path(grid_map, start, goal), self.rhs.flatten(), self.g.flatten()
 
 
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
 
 
##############################################################################
 
# path finding function
# credits to: https://www.analytics-link.com/post/2018/09/14/applying-the-a-path-finding-algorithm-in-python-part-1-2d-square-Grid
 
##############################################################################
 
 
def astar(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
 
    close_set = set()
 
    came_from = {}
 
    gscore = {start: 0}
 
    fscore = {start: heuristic(start, goal)}
 
    oheap = []
 
    heapq.heappush(oheap, (fscore[start], start))
 
    while oheap:
 
        current = heapq.heappop(oheap)[1]
 
        if current == goal:
 
            data = []
 
            while current in came_from:
                data.append(current)
 
                current = came_from[current]
 
            return data
 
        close_set.add(current)
 
        for i, j in neighbors:
 
            neighbor = current[0] + i, current[1] + j
 
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
 
            if 0 <= neighbor[0] < array.shape[0]:
 
                if 0 <= neighbor[1] < array.shape[1]:
 
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
 
                else:
 
                    # array bound y walls
 
                    continue
 
            else:
 
                # array bound x walls
 
                continue
 
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
 
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
 
                gscore[neighbor] = tentative_g_score
 
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
 
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
 
    return None
 
 