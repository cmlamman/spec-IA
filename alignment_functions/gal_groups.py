import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from astropy.cosmology import LambdaCDM, z_at_value
from alignment_functions.basic_alignment import *
from geometry_functions.coordinate_functions import get_proj_dist


H0 = 69.6
cosmo = LambdaCDM(H0=H0, Om0=0.286, Ode0=0.714)
h = H0/100
from geometry_functions.coordinate_functions import *

from collections import defaultdict


############################################
# FINDING groupS WITH UNION-FIND
############################################
class Node:
    def __init__(self, key):
        self.key = key
        self.parent = self
        self.size = 1

class UnionFind(dict):
    def find(self, key):
        node = self.get(key, None)
        if node is None:
            node = self[key] = Node(key)
        else:
            while node.parent != node: 
                # walk up & perform path compression
                node.parent, node = node.parent.parent, node.parent
        return node

    def union(self, key_a, key_b):
        node_a = self.find(key_a)
        node_b = self.find(key_b)
        if node_a != node_b:  # disjoint? -> join!
            if node_a.size < node_b.size:
                node_a.parent = node_b
                node_b.size += node_a.size
            else:
                node_b.parent = node_a
                node_a.size += node_b.size
                
def find_components(line_iterator, max_size):
    forest = UnionFind()

    for line in line_iterator:
        forest.union(*line.split())

    result = defaultdict(list)
    for key in forest.keys():
        root = forest.find(key)
        if root.size <= max_size:
            result[root.key].append(int(float(key)))  # Convert key to float and then to int

    # return list of integers
    return list(result.values())

def find_groups_UnionFind(points, max_n=10, transverse_max = 0.5, los_max = 6):
    
    # find nearest neighbor for each galaxy
    tree = cKDTree(points)
    max_pair_sep = np.sqrt(transverse_max**2 + los_max**2)
    dd, ii = tree.query(points, distance_upper_bound=max_pair_sep, k=2) 

    # only keep indices where neighbor exists
    ii = ii[ii[:,1] < len(points)]
    
    # remove pairs that are too far away
    transverse_seps = np.abs(get_proj_dist(points[ii[:,1]], points[ii[:,0]]))
    los_dist = np.sqrt(np.sum(points**2, axis=1))
    los_seps = np.abs(los_dist[ii[:,1]] - los_dist[ii[:,0]])
    too_far = (transverse_seps > transverse_max) | (los_seps > los_max)
 
    #too_far[:,:1]=False # don't want to remove indices of centers
    ii = ii[~too_far]  # where the pairs are too close, functionally remove them from the list of pairs
    
    # find groups
    group_results = find_components([' '.join(map(str, row)) for row in ii], max_n)
    
    return group_results
    
# remove groups that have an outside object within group_sep_min
def remove_close_groups(points, group_result, group_transverseSep_min=10, max_n=10, los_max=100):
    # find the center coordinates of each group
    group_centers = np.array([np.mean(points[group], axis=0) for group in group_result])
    
    # build the tree of all galaxies
    point_tree = cKDTree(points)
    
    # find neighbors of each group center
    dd, ii = point_tree.query(group_centers, k=max_n+1, distance_upper_bound=np.sqrt(group_transverseSep_min**2 + los_max**2))
    
    # add row of infinite values to points
    points = np.vstack((points, np.full(len(points[0]), np.inf)))
    # for each line in ii, if an index is in the corresponding line of the group_result, then replace it with the length of points
    ii = np.array([[len(points)-1 if idx in group else idx for idx in line] for group, line in zip(group_result, ii)])
    # find the transverse separation between each group center and its neighbors
    transverse_seps = np.array([np.abs(get_proj_dist(points[ii[i]], group_centers[i])) for i in range(len(group_result))])
    
    sufficiently_isolated = (np.nanmin(transverse_seps, axis=1) > group_transverseSep_min)
    
    return [group_result[i] for i in range(len(group_result)) if sufficiently_isolated[i]]
    
    
###############################################
# CALCULATE GROUP PROJECTED SHAPE
###############################################


def calculate_2D_group_orientation(points_3d):
    '''
    Calculate the orientation of a group of points, measured "E of N", aka clockwise from the y-axis
    points_2d: array of shape (n_points, 2)
    returns: array of angles in radians, shape (n_points,)
    '''
    points_2d = get_points_in_plane(points_3d)
    # convert to polar coordinates
    r = np.linalg.norm(points_2d, axis=1)
    theta = np.arctan2(points_2d[:,0], points_2d[:,1])
    
    # compute complex number representation of the points
    points_complex = r * np.exp(2j*theta)  # 2 so invariant under rotation
    
    average_points_complex = np.mean(points_complex)
    
    return np.angle(average_points_complex) / 2

#-----------------------
# backup functions

# cutting list down to requirements
def trim_groups(points, group_indices, transverse_max, los_max):
    # assumes observer is as orgin
    
    # find groupenters
    # find the center coordinates of each group
    group_centers = np.array([np.mean(points[cl], axis=0) for cl in group_indices])
    
    # los-distance for groupenters
    los_dist = np.sqrt(np.sum(group_centers**2, axis=1))
    
    max_transverse_dist_to_center = []
    max_los_dist_to_center = []
    
    for i in range(len(group_indices)):
        cl = group_indices[i]
        max_transverse_dist_to_center.append(np.max(get_proj_dist(points[cl], group_centers[i])))
        max_los_dist_to_center.append(np.max(los_dist[i] - np.sqrt(np.sum(points[cl]**2, axis=1))))
    
    groupto_keep = (np.asarray(max_transverse_dist_to_center) < transverse_max)
    groupto_keep &= (np.asarray(max_los_dist_to_center) < los_max)
    
    return [group_indices[i] for i in range(len(group_indices)) if groupto_keep[i]]