# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial
import math
from numpy import random
from PIL import Image


# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row  # coordinate
        self.col = col  # coordinate
        self.parent = None  # parent node
        self.cost = 0.0  # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array  # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]  # map size
        self.size_col = map_array.shape[1]  # map size

        self.start = Node(start[0], start[1])  # start node
        self.goal = Node(goal[0], goal[1])  # goal node
        self.vertices = []  # list of nodes
        self.found = False  # found flag

    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        distance = math.sqrt((node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2)
        return distance

    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        p10 = node1.row
        p11 = node1.col
        p20 = node2.row
        p21 = node2.col

        yslope = p21 - p11
        xslope = p20 - p10
        collision = False

        if p10 == p20:
            if p11 > p21:
                temp1 = p10
                temp2 = p11
                p10 = p20
                p11 = p21
                p20 = temp1
                p21 = temp2

            while (p10, p11) != (p20, p21):
                p11 = p11 + 1
                if self.map_array[p10][p11] == 0:
                    collision = True
                    break

        elif p11 == p21:
            if p10 > p20:
                temp1 = p10
                temp2 = p11
                p10 = p20
                p11 = p21
                p20 = temp1
                p21 = temp2
            while (p10, p11) != (p20, p21):
                p10 = p10 + 1
                if self.map_array[p10][p11] == 0:
                    collision = True
                    break

        else:
            slope = yslope / xslope
            c = (p11 - slope * p10)
            if p11 > p21:
                temp1 = p10
                temp2 = p11
                p10 = p20
                p11 = p21
                p20 = temp1
                p21 = temp2
            while (p10, p11) != (p20, p21):
                p11 = p11 + 0.05
                p11 = round(p11, 2)
                p10 = (p11 - c) / slope
                p10 = round(p10)
                temp = round(p11)
                if self.map_array[p10][temp] == 0:
                    collision = True
                    break

        if collision == False:
            return False
        else:
            return True

    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###
        if goal_bias == False:
            p1rows = random.randint(self.size_row)
            p1cols = random.randint(self.size_col)
            new_point = Node(p1rows, p1cols)
            return new_point

        else:
            return self.goal

    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        dist_new_node = []
        minindex = 0
        for i in range(len(self.vertices)):
            distance = self.dis(self.vertices[i], point)
            dist_new_node.append(distance)

        minindex = dist_new_node.index(min(dist_new_node))
        return self.vertices[minindex]

    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance
        '''
        ### YOUR CODE HERE ###
        neighbors = []
        freeneighbors = []
        for i in range(len(self.vertices)):
            distance_rrt = self.dis(new_node, self.vertices[i])
            if distance_rrt <= neighbor_size:
                neighbors.append(self.vertices[i])

        for i in range(len(neighbors)):
            collision_1 = self.check_collision(neighbors[i], new_node)
            if collision_1 == True:
                continue
            else:
                freeneighbors.append(neighbors[i])
        return freeneighbors

    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###
        checkdis = []
        free = []
        for i in range(len(neighbors)):
            collision = self.check_collision(neighbors[i], new_node)
            if collision == False:
                free.append(neighbors[i])
            else:
                continue
        for i in range(len(free)):
            cost1 = free[i].cost + self.dis(free[i], new_node)
            checkdis.append(cost1)

        minindex = checkdis.index(min(checkdis))
        new_node.parent = neighbors[minindex]
        new_node.cost = neighbors[minindex].cost + int(self.dis(neighbors[minindex], new_node))

        for i in range(len(free)):
            initcost = free[i].cost
            rewiredcost = int(self.dis(new_node, free[i])) + new_node.cost

            if initcost > rewiredcost:
                neighbors[i].parent = new_node
                neighbors[i].cost = rewiredcost
            else:
                continue

    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col and cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()

    # step=10

    def angle(self, p1, p2):
        step = 10
        distance_1 = self.dis(p1, p2)
        if distance_1 > step:
            distance_1 = step

        theta = math.atan2(p2.col - p1.col, p2.row - p1.row)
        new_node1 = Node((int((p1.row + distance_1 * math.cos(theta)))), (int((p1.col + distance_1 * math.sin(theta)))))

        return new_node1

    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point,
        # get its nearest node,
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.
        goal_bias = False

        for i in range(n_pts):
            step = 10
            # goal_bias=False
            new_point = self.get_new_point(goal_bias)
            nearest_node = self.get_nearest_node(new_point)
            if goal_bias == True:
                new_node_1 = self.goal
            else:
                new_node_1 = self.angle(nearest_node, new_point)

            new_node_1.cost = int(nearest_node.cost + self.dis(new_node_1, nearest_node))

            distance_to_goal = self.dis(new_node_1, self.goal)
            if distance_to_goal <= step:
                goal_bias = True
            collision = self.check_collision(nearest_node, new_node_1)
            if collision == False:
                self.vertices.append(new_node_1)
                new_node_1.parent = nearest_node
            else:
                continue
            if distance_to_goal == 0:
                self.found = True
                break

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()

    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points
            neighbor_size - the neighbor distance

        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point,
        # get its nearest node,
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.
        goal_bias = False

        for i in range(n_pts):
            print(i)
            step = 10
            # goal_bias=False
            new_point = self.get_new_point(goal_bias)
            if self.map_array[new_point.row][new_point.col] == 0:
                continue
            else:
                nearest_node = self.get_nearest_node(new_point)

                if goal_bias == True:
                    new_node_1 = self.goal
                else:
                    new_node_1 = self.angle(nearest_node, new_point)

                ckcollision = self.check_collision(new_node_1, nearest_node)
                if ckcollision == True:
                    continue
                else:
                    newneighbors = self.get_neighbors(new_node_1, 20)
                    self.rewire(new_node_1, newneighbors)
                    distance_to_goal = self.dis(new_node_1, self.goal)
                    if distance_to_goal <= step:
                        goal_bias = True
                    self.vertices.append(new_node_1)
                    if distance_to_goal == 0:
                        self.found = True
                        print('iteration',i,' cost', self.goal.cost)
                        self.draw_map()
                        goal_bias = False

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()