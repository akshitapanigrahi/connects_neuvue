import numpy as np
import trimesh

def is_array_like(current_data,include_tuple=False):
    types_to_check = [type(np.ndarray([])),type(np.array([])),list,trimesh.caching.TrackedArray]
    if include_tuple:
        types_to_check.append(tuple)
    return type(current_data) in types_to_check


def convert_to_array_like(array,include_tuple=False):
    """
    Will convert something to an array
    """
    if not nu.is_array_like(array,include_tuple=include_tuple):
        return [array]
    return array

from . import numpy_utils as nu