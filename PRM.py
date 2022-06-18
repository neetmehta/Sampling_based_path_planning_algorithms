# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial


# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.kdtree = None                    # kd tree spatial.KDTree([[]])
        self.path = []                        # list of nodes of the found path


    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        # Check obstacle between nodes
        # get all the poitns in between
        points_between = zip(np.linspace(p1[0], p2[0], dtype=int), 
                             np.linspace(p1[1], p2[1], dtype=int))
        # check if any of these are obstacles
        for point in points_between:
            if self.map_array[point[0]][point[1]] == 0:
                return True
        return False


    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


    def get_sample_points(self, n_pts, random=True):
        '''Get the row and col coordinates of n sample points
        arguments:
            n_pts - total number of points to be sampled
            random - random or uniform?

        return:
            p_row - row coordinates of n sample points (1D)
            p_col - col coordinates of n sample points (1D)
        '''
        # number of points
        n_row = int(np.sqrt(n_pts * self.size_row / self.size_col))
        n_col = int(n_pts / n_row)
        # generate uniform points
        if not random:
            sample_row = np.linspace(0, self.size_row-1, n_row, dtype=int)
            sample_col = np.linspace(0, self.size_col-1, n_col, dtype=int)
            p_row, p_col = np.meshgrid(sample_row, sample_col)
            p_row = p_row.flatten()
            p_col = p_col.flatten()
        # generate random points
        else:
            p_row = np.random.randint(0, self.size_row-1, n_pts, dtype=int)
            p_col = np.random.randint(0, self.size_col-1, n_pts, dtype=int)
        return p_row, p_col


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and store valide points in self.samples
        '''
        # Initialize graph
        self.graph.clear()
        # Generate uniform points
        p_row, p_col = self.get_sample_points(n_pts, random=False)
        # Check obstacle
        for row, col in zip(p_row, p_col):
            if self.map_array[row][col] == 1:
                self.samples.append((row, col))

    
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and store valide points in self.samples
        '''
        # Generate random points
        p_row, p_col = self.get_sample_points(n_pts)
        # Check obstacle
        for row, col in zip(p_row, p_col):
            if self.map_array[row][col] == 1:
                self.samples.append((row, col))


    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and store valide points in self.samples
        '''
        # Generate random points
        p1_row, p1_col = self.get_sample_points(n_pts)
        # Generate random points at some distance from the preivous generated points
        scale = 10 # std
        p2_row = p1_row + np.random.normal(0.0, scale, n_pts).astype(int)
        p2_col = p1_col + np.random.normal(0.0, scale, n_pts).astype(int)
        # Check if the point is close to an obstacle
        for row1, col1, row2, col2 in zip(p1_row, p1_col, p2_row, p2_col):
            if not(0 <= row2 < self.size_row) or not(0 <= col2 < self.size_col):
                continue
            # one of them is obstacle and the other is free space
            if self.map_array[row1][col1] == 1 and self.map_array[row2][col2] == 0:
                self.samples.append((row1, col1))
            elif self.map_array[row1][col1] == 0 and self.map_array[row2][col2] == 1:
                self.samples.append((row2, col2))
    

    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and store valide points in self.samples
        '''
        # Generate random points
        p1_row, p1_col = self.get_sample_points(n_pts)
        # Generate random points at some distance from the preivous generated points
        scale = 15 # std
        p2_row = p1_row + np.random.normal(0.0, scale, n_pts).astype(int)
        p2_col = p1_col + np.random.normal(0.0, scale, n_pts).astype(int)
        # check if it is the "bridge" form
        for row1, col1, row2, col2 in zip(p1_row, p1_col, p2_row, p2_col):
            # both are obstacles or outside the boundary
            if ((not(0 <= col2 < self.size_col) or not(0 <= row2 < self.size_row)) or \
               self.map_array[row2][col2] == 0) and \
               self.map_array[row1][col1] == 0:
                # make sure the midpoint is inside the map
                mid_row, mid_col = int(0.5*(row1+row2)), int(0.5*(col1+col2))
                if 0 <= mid_row < self.size_row and 0 <= mid_col < self.size_col and \
                   self.map_array[mid_row][mid_col] == 1:
                    self.samples.append((mid_row, mid_col))


    def add_vertices_pairs_edge(self, pairs):
        '''Add pairs of vertices to graph as weighted edge
        arguments:
            pairs - pairs of vertices of the graph

        check collision, compute weight and add valide edges to self.graph
        '''
        for pair in pairs:
            if pair[0] == "start":
                point1 = self.samples[-2]
            elif pair[0] == "goal":
                point1 = self.samples[-1]
            else:
                point1 = self.samples[pair[0]]
            point2 = self.samples[pair[1]]

            if not self.check_collision(point1, point2):
                d = self.dis(point1, point2)
                edge = [(pair[0], pair[1], d)]
                self.graph.add_weighted_edges_from(edge)


    def connect_vertices(self, kdtree_d=15):
        '''Add nodes and edges to the graph from sampled points
        arguments:
            kdtree_d - the distance for kdtree to search for nearest neighbors

        Add nodes to graph from self.samples
        Build kdtree to find neighbor pairs and add them to graph as edges
        '''
        # Finds k nearest neighbors
        # kdtree 
        self.kdtree = spatial.cKDTree(list(self.samples))
        pairs = self.kdtree.query_pairs(kdtree_d)

        # Add the neighbor to graph
        self.graph.add_nodes_from(range(len(self.samples)))
        self.add_vertices_pairs_edge(pairs)


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform", kdtree_d=15):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method
            kdtree_d - the distance for kdtree to search for nearest neighbors

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialization
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)

        # Connect the samples
        self.connect_vertices(kdtree_d)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))


    def search(self, start, goal, kdtree_d=15):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]
            kdtree_d - the distance for kdtree to search for nearest neighbors
                       for start and goal node

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        self.graph.add_nodes_from(['start', 'goal'])

        # Connect start and goal nodes to the surrounding nodes
        start_goal_tree = spatial.cKDTree([start, goal])
        neighbors = start_goal_tree.query_ball_tree(self.kdtree, kdtree_d)
        start_pairs = ([['start', neighbor] for neighbor in neighbors[0]])
        goal_pairs = ([['goal', neighbor] for neighbor in neighbors[1]])

        # Add the edge to graph
        self.add_vertices_pairs_edge(start_pairs)
        self.add_vertices_pairs_edge(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        