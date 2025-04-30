"""
Extracted skeleton utility functions to be used in api
"""
import numpy as np
def calculate_skeleton_distance(my_skeleton):
    if len(my_skeleton) == 0:
        return 0
    total_distance = np.sum(np.sqrt(np.sum((my_skeleton[:,0] - my_skeleton[:,1])**2,axis=1)))
    return float(total_distance)

def empty_skeleton():
    return np.array([]).reshape(-1,2,3)