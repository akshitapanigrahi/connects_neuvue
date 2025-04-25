import networkx as nx
import numpy as np

def add_bounding_boxes_to_graph(graph,verbose = False):
    """
    Adds bounding box attributes (x, y, z) to each node with 'skeleton_data'.
    
    Parameters:
        graph (nx.Graph): A NetworkX graph with nodes containing 'skeleton_data' (Nx3 NumPy arrays).
    
    Returns:
        None (modifies the graph in place).
    """
    for node, data in graph.nodes(data=True):
        skeleton = data.get('skeleton_data')
        if skeleton is not None and isinstance(skeleton, np.ndarray) and skeleton.shape[1] == 3:
            min_coords = skeleton.min(axis=0)
            max_coords = skeleton.max(axis=0)
            graph.nodes[node]['bounding_box'] = np.vstack([min_coords,max_coords])
        else:
            if verbose:
                print(f"Skipping node {node}: 'skeleton_data' missing or not a valid Nx3 array")


    