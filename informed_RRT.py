# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
import random
from math import *


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
        """Intialize the map before each search
        """
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    def dis(self, node1, node2):
        """Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        """
        ### YOUR CODE HERE ###
        distance = sqrt((node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2)
        return distance

    def check_collision(self, node1, node2):
        """Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        """
        ### YOUR CODE HERE ###
        rdis = abs(node1.row - node2.row)
        cdis = abs(node1.col - node2.col)
        c1 = node1.col
        r1 = node1.row
        r2 = node2.row
        c2 = node2.col
        list1 = []
        if rdis > cdis:
            for r in range(min(r2, r1), max(r2, r1)):
                list1.append([r, round(((node2.col - node1.col) / (node2.row - node1.row)) * (r - r1) + c1)])

        else:
            for c in range(min(c2, c1), max(c2, c1)):
                list1.append([round(((node1.row - node2.row) / (node1.col - node2.col)) * (c - c1) + r1), c])

        for coordinates in list1:
            if self.map_array[coordinates[0], coordinates[1]] == 0:
                return True, list1

        return False, list1

    def get_new_point(self, goal_bias, goal_region,Flag, Flag2=False, cbest=0):
        """Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        """
        ### YOUR CODE HERE ###

        if Flag2:
            while True:
                row = np.random.randint(0, self.size_col-1)
                col = np.random.randint(0, self.size_col-1)
                if (np.sqrt((self.start.row-row)**2 + (self.start.col-col)**2 ) + np.sqrt((self.goal.row-row)**2 + (self.goal.col-col)**2)) <= cbest:
                    return Node(row, col), self.found

        probablity_no = np.random.uniform()
        near_to_goal = self.get_nearest_node(self.goal)
        dis_to_goal = self.dis(near_to_goal, self.goal)

        if dis_to_goal > goal_region or Flag:
            if probablity_no > goal_bias:
                return Node(random.randint(0, self.size_row - 1), random.randint(0, self.size_col - 1)), False

            else:
                return self.goal, False

        if dis_to_goal < goal_region:
            return self.goal,True

    def get_nearest_node(self, point):
        """Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        """
        ### YOUR CODE HERE ###
        d = 100000
        for node in self.vertices:
            if self.dis(node, point) < d:
                d = self.dis(node, point)
                nearest_node = node

        return nearest_node

    def get_neighbors(self, new_node, neighbor_size):
        """Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance
        """
        ### YOUR CODE HERE ###
        neighbor_list = []
        for node in self.vertices:
            if self.dis(new_node, node) < neighbor_size:
                neighbor_list.append(node)
        return neighbor_list

    def rewire(self, new_node, neighbors):
        """Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        """
        ### YOUR CODE HERE ###
        d = 100000
        for node in neighbors:
            if (self.dis(new_node, node) + node.cost) < d:
                if not self.check_collision(new_node, node)[0]:
                    new_node.parent = node
                    new_node.cost = self.dis(new_node, node) + node.cost
                    d = self.dis(new_node, node) + node.cost

        # print(node.cost > self.dis(new_node, node) + new_node.cost)
        for node in neighbors:
            if not self.check_collision(new_node, node)[0]:
                if node.cost > self.dis(new_node, node) + new_node.cost:
                    node.parent = new_node
                    node.cost = self.dis(new_node, node) + new_node.cost

        self.vertices.append(new_node)

    def draw_map(self):
        """Visualization of the result
        """
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
                # print(cur.row,cur.col)

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()


    def informed_RRT_star(self, n_pts=2000, neighbor_size=20):
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
        max_dis = 10
        goal_region = 10
        goal_bias = 0.05                 # 0<=goal_bias<1
        Flag = False
        for k in range(n_pts):
            new_point, Flag1 = self.get_new_point(goal_bias, goal_region, Flag)
            nearest_point = self.get_nearest_node(new_point)

            d = self.dis(new_point, nearest_point)

            if new_point == self.goal and not self.check_collision(new_point, nearest_point)[0] and Flag1:
                self.found = True
                new_point.parent = nearest_point
                new_point.cost = nearest_point.cost + self.dis(new_point, nearest_point)
                self.vertices.append(new_point)
                Flag = True
                # print(f'################### goal reached on this node {k} #######################')

                print(f'initial goal cost was {self.goal.cost}')
                print(f'################### goal reached on this node {k} #######################\ncontinuing the loop')
                self.draw_map()
                kk = k
                break

            #################################### d > max dis ########################################

            if d > max_dis:
                list1 = self.check_collision(nearest_point, new_point)[1]
                list2 = [(i[0], i[1], self.dis(Node(i[0], i[1]), nearest_point)) for i in list1 if
                         self.dis(Node(i[0], i[1]), nearest_point) < max_dis]
                list2.sort(key=lambda x: -x[2])
                point = Node(list2[0][0], list2[0][1])


                if not self.check_collision(nearest_point, point)[0]:
                    point.parent = nearest_point
                    point.cost = nearest_point.cost + self.dis(nearest_point, point)
                    # self.vertices.append(point)
                    neighbors = self.get_neighbors(point, neighbor_size)
                    self.rewire(point, neighbors)
                    if Flag:
                        print(f'after goal found iteration number {k}')

                    else:
                        print(f'before goal found iteration number {k}')
                    continue


            else:
                if not self.check_collision(nearest_point, new_point)[0]:
                    new_point.parent = nearest_point
                    new_point.cost = nearest_point.cost + self.dis(nearest_point, new_point)
                    # self.vertices.append(new_point)
                    neighbors = self.get_neighbors(new_point, neighbor_size)
                    self.rewire(new_point, neighbors)
                    if Flag:
                        print(f'after goal found iteration number {k}')
                    else:
                        print(f'before goal found iteration number {k}')
                    continue

        if self.found:
            for i in range(n_pts-kk):
                cbest = self.goal.cost
                new_point, Flag1 = self.get_new_point(goal_bias, goal_region, Flag, self.found, cbest)
                nearest_point = self.get_nearest_node(new_point)

                d = self.dis(new_point, nearest_point)

                if d > max_dis:
                    list1 = self.check_collision(nearest_point, new_point)[1]
                    list2 = [(i[0], i[1], self.dis(Node(i[0], i[1]), nearest_point)) for i in list1 if
                             self.dis(Node(i[0], i[1]), nearest_point) < max_dis]
                    list2.sort(key=lambda x: -x[2])
                    point = Node(list2[0][0], list2[0][1])


                    if not self.check_collision(nearest_point, point)[0]:
                        point.parent = nearest_point
                        point.cost = nearest_point.cost + self.dis(nearest_point, point)
                        # self.vertices.append(point)
                        neighbors = self.get_neighbors(point, neighbor_size)
                        self.rewire(point, neighbors)
                        if Flag:
                            print(f'after goal found iteration number {i+kk}')

                        else:
                            print(f'before goal found iteration number {i+kk}')
                        continue


                else:
                    if not self.check_collision(nearest_point, new_point)[0]:
                        new_point.parent = nearest_point
                        new_point.cost = nearest_point.cost + self.dis(nearest_point, new_point)
                        # self.vertices.append(new_point)
                        neighbors = self.get_neighbors(new_point, neighbor_size)
                        self.rewire(new_point, neighbors)
                        if Flag:
                            print(f'after goal found iteration number {i+kk}')
                        else:
                            print(f'before goal found iteration number {i+kk}')
                        continue



        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
