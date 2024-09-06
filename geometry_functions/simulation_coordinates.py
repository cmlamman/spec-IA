# SIMPLE FUNCTION TO MODIFY A PERIODIC BOX FOR THE REST OF THE CODE
import numpy as np

def make_tracers_periodic(origional_points, extend_by=100):
    '''
    Function to extend a set of points in a periodic box by a given distance. Used for extended a tracer sample to use multiplets / shapes in the full box.
    origional_points: np.array of shape (n,3) where n is the number of points and the columns are the x,y,z coordinates
    extend_by: float, the distance to extend the box by. In same units as the origional points.
    
    '''
    new_orgins = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]) * np.max(origional_points[:,0])  # assumes box is a cube with one corner at 0,0,0
    extended_points = np.array([origional_points + new_orgins[i] for i in range(len(new_orgins))])
    extended_points = np.concatenate(extended_points, axis=0)

    # trim to points within some distance of origional box
    i_keep = (extended_points[:,0] > np.min(origional_points[:,0])-extend_by) & (extended_points[:,0] < np.max(origional_points[:,0])+extend_by) 
    i_keep &= (extended_points[:,1] > np.min(origional_points[:,1])-extend_by) & (extended_points[:,1] < np.max(origional_points[:,1])+extend_by)
    i_keep &= (extended_points[:,2] > np.min(origional_points[:,2])-extend_by) & (extended_points[:,2] < np.max(origional_points[:,2])+extend_by)
    periodic_tracer_points = extended_points[i_keep]
    
    return periodic_tracer_points