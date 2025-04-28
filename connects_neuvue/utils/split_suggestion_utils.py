import numpy as np
import pandas as pd
def red_blue_suggestion_dicts(
    red_blue_suggestions,
    segment_id=None,
    split_index = 0,
    key = None,
    return_df = False,
    verbose = False,
    debug = False,
    attributes_to_include = (
        "valid_points",
        "valid_pre_coordinates",
        "valid_post_coordinates",
        "error_points",
        "error_branches_skeleton_points",
        "error_pre_coordinates",
        "error_post_coordinates",
    ),
    name_ending = "_red_blue_suggestions"
    ):
    
    if key is None:
        key = dict(segment_id=segment_id,split_index=split_index)

    filters = [k[:k.find(name_ending)] for k in red_blue_suggestions]

    if debug:
        print(f"filters = {filters}")
    
    filter_keys = []
    global_cut_id = 0
    add_blank_key_if_none = True
    
    for filter_name in filters:
        curr_filter_dict = red_blue_suggestions[f"{filter_name}_red_blue_suggestions"]
        if verbose:
            print(f"---Working on {filter_name}: # of edits = {len(curr_filter_dict)}---")
    
        
        total_keys = []
        counter = 0
    
        if len(curr_filter_dict) == 0:
            if add_blank_key_if_none:
                new_key = dict(key,
                               cut_id = global_cut_id,
                               filter_cut_id = counter,
                                skeletal_length = 0,
                                error_type = filter_name,
                                n_error_branches = 0)
                total_keys.append(new_key)
    
    
        pre_post_coord_dict_nm = None
    
        total_keys = []
        for limb_name,limb_dict in curr_filter_dict.items():
            for split_idx,split_info in limb_dict.items():
                for curr_d in split_info:
                    if verbose:
                        print(f"-- Working on {limb_name} cut_idx {counter} --")
    
                    new_key = dict(
                                key,
                                error_type = filter_name,
                                cut_id = global_cut_id,
                                filter_cut_id = counter,
                                limb_name = limb_name,
                                limb_split_idx = split_idx,
                                skeletal_length = np.round(curr_d["error_branches_skeleton_length"],3),
    
                                parent_branch_width = np.round(curr_d["parent_branch_width"],3),
                                n_error_branches = curr_d["n_error_branches"],
    
                                # blue_points_with_syn = blue_points_with_syn,
                                # red_points_with_syn=red_points_with_syn,
                                # n_red_pts_with_syn = n_red_pts_with_syn,
                                # n_blue_pts_with_syn = n_blue_pts_with_syn,
                                
    
                                # red_points = red_points,
                                # blue_points = blue_points,
                                # n_red_pts = n_red_pts,
                                # n_blue_pts = n_blue_pts,
    
                            )
    
                    # adding on extra attributes
                    add_on_dict = {k:curr_d[k] for k in attributes_to_include}
                    new_key.update(add_on_dict)
                                   
                    try:
                        merge_dict = dict(
                            merge_coordinate_x_nm = curr_d["coordinate"][0][0],
                            merge_coordinate_y_nm = curr_d["coordinate"][0][1],
                            merge_coordinate_z_nm = curr_d["coordinate"][0][2],
                        )
                    except:
                        merge_dict = dict(
                            merge_coordinate_x_nm = curr_d["coordinate"][0],
                            merge_coordinate_y_nm = curr_d["coordinate"][1],
                            merge_coordinate_z_nm = curr_d["coordinate"][2],
                        )
                        
                    new_key.update(merge_dict)
    
                    counter += 1
                    global_cut_id += 1
    
                    total_keys.append(new_key)
        filter_keys += total_keys

    if return_df:
        filter_keys = pd.DataFrame.from_records(filter_keys)
    return filter_keys