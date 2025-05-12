import numpy as np
import pandas as pd
try:
    from datasci_tools import ipyvolume_utils as ipvu
except:
    ipvu = None

proofread_split_type_to_color_map = {
    "axon_on_dendrite_merges":"red",
    "high_degree_branching":"blue",
    "low_degree_branching":"purple",
    "high_degree_branching_dendrite":"orange",
    "width_jump_up_dendrite":"black",
    "width_jump_up_axon":"yellow",
    "double_back_dendrite":"pink",
    "dendrite_internal_bend":"brown"
}

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
        "n_error_branches",
        "error_points",
        "error_branches_skeleton_points",
        "error_pre_coordinates",
        "error_post_coordinates",
        
    ),
    name_ending = "_red_blue_suggestions",
    include_downstream_stats = False,
    include_parent_stats = True,

    ):
    
    if key is None:
        key = dict(segment_id=segment_id,split_index=split_index)

    filters = [k[:k.find(name_ending)] for k in red_blue_suggestions]

    if debug:
        print(f"filters = {filters}")
    
    filter_keys = []
    global_cut_id = 0
    add_blank_key_if_none = True
    parent_stats = []
    downstream_stats = []
    
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
                    
                    if len(parent_stats) == 0 and include_parent_stats:
                        parent_stats = [k for k in curr_d if "parent_" in k]
                    parent_dict = {k:curr_d.get(k,None) for k in parent_stats}  
                    
                    
                    if len(downstream_stats)==0 and include_downstream_stats:
                        downstream_stats = [k for k in curr_d if "downstream_" in k]
                    down_dict = {k:curr_d.get(k,None) for k in downstream_stats}    
                    
                    
                    new_key.update({**add_on_dict,**parent_dict,**down_dict})
                                   
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


columns_to_sum_default = (
        "downstream_skeletal_length_sum",
        "downstream_n_synapses_sum",
        "downstream_n_synapses_post_sum",
        "downstream_n_synapses_pre_sum",
        "downstream_n_spines_sum",
)

def error_type_red_blue_df_from_red_blue_df(
    df,
    columns_to_sum = None,
    ):
    if columns_to_sum is None:
        columns_to_sum = columns_to_sum_default
    groupby_cols = ["error_type"]
    agg_dict= dict(count=('error_type','count'))
    agg_dict.update({k:(k,"sum") for k in columns_to_sum})
    return_df = df.groupby(groupby_cols).agg(**agg_dict).reset_index()
    return return_df
        
def segment_error_type_red_blue_df_from_red_blue_df(
    df,
    columns_to_sum = None,
):
    if columns_to_sum is None:
        columns_to_sum = columns_to_sum_default
    groupby_cols = ['segment_id', 'split_index','error_type',]
    agg_dict= dict(count=('error_type','count'))
    agg_dict.update({k:(k,"sum") for k in columns_to_sum})
    return_df = df.groupby(groupby_cols).agg(**agg_dict).reset_index()
    return return_df

def check_ipvu():
    if ipvu is None:
        raise ModuleNotFoundError(f"datasci tools must be installed for visualizations: run pip3 install datasci_stdlib_tools")


error_points_color_default = "red"
valid_points_color_default = "blue"
error_skeleton_points_color_default = "grey"
scatter_size_default = 0.15
def plot_red_blue_split_dict(
    split_dict,
    mesh,
    error_points_color = error_points_color_default,
    valid_points_color = valid_points_color_default,
    error_skeleton_points_color = error_skeleton_points_color_default,
    plot_skeleton = True,
    verbose = True,
    metrics_to_print = ("skeletal_length",),
    scatter_size = scatter_size_default,
    ):
    
    if type(split_dict) == dict:
        split_dict = [split_dict]
    
    check_ipvu()
    
    if metrics_to_print is None:
        metrics_to_print = columns_to_sum_default

    segment_id,split_index = split_dict[0]['segment_id'],split_dict[0]['split_index']
    
    if verbose:
        print(f"Segment_id {segment_id} split_index {split_index}:")
    
    #gets the decimated mesh to overlay the edits
    # if mesh is None:
    #     mesh = vdi.fetch_segment_id_mesh(segment_id=segment_id)
    # gets the red blue points
    e_pts_list = []
    v_pts_list = []
    sk_list = []

    for s_dict in split_dict:
        e_pts_list.append( s_dict["error_points"])
        v_pts_list.append(s_dict["valid_points"])
        e_sk_pts = s_dict["error_branches_skeleton_points"]
        
        if verbose:
            (error_type,
            filter_cut_id,
            limb_name) = s_dict['error_type'],s_dict['filter_cut_id'],s_dict["limb_name"]
            
            print(f"{error_type} split #{filter_cut_id} (on limb {limb_name})")
            for k in metrics_to_print:
                print(f"  {k} = {s_dict[k]:.2f}")
        
        if plot_skeleton:
            sk_list.append(e_sk_pts)
            
    scatters = [np.vstack(e_pts_list),np.vstack(v_pts_list)]
    scatters_colors = [error_points_color,valid_points_color]
    
    if plot_skeleton:
        scatters.append(np.vstack(sk_list))
        scatters_colors.append(error_skeleton_points_color)
    
    
    ipvu.plot_objects(
        mesh,
        scatters=scatters,
        scatters_colors=scatters_colors,
        scatter_size=scatter_size,
    )
    
def plot_red_blue_for_segment(
    segment_id,
    red_blue_df,
    mesh,
    split_index = 0,
    verbose = True,
    error_points_color = error_points_color_default,
    valid_points_color = valid_points_color_default,
    error_skeleton_points_color = error_skeleton_points_color_default,
    plot_skeleton = True,
    scatter_size = scatter_size_default,
    error_types_to_plot = None,
    error_types_to_skip = None,
    ):
    """
    Purpose
    -------
    plot all of th red blue points and coordinates that distinguish different 
    types of merge errors
    
    Pseudocode
    ----------
    1. Get all of the error dicts from the data trame for that segmenet
    2. For each dict:
        - get the blue,red points and coordinate
        - map the error type to a color
        - add all the scatters and their colors to scatters to be potted
    3. 
    """
    # #if mesh is None:
    # mesh = vdi.fetch_segment_id_mesh(segment_id=segment_id)
    check_ipvu()
    
    seg_df = red_blue_df.query(f"(segment_id == {segment_id}) and (split_index == {split_index})")
    
    node = f"{segment_id}_{split_index}"
    if verbose:
        print(f"For {node}: # of total splits = {len(seg_df)}")
    
    scatters=[]
    scatters_colors = []
    epts_list = []
    vpts_list = []
    sk_pts_list = []
    
    for groups,split_df in seg_df.groupby(['error_type']):
        error_name =groups[0]
        split_color = proofread_split_type_to_color_map[error_name]
        n_splits = len(split_df)
        if verbose:
            print(f"  {error_name}: {n_splits} ({split_color})")
        
        if error_types_to_plot is not None:
            if error_name not in error_types_to_plot:
                if verbose:
                    print(f"  --> skipping plot")
                continue
            
        if error_types_to_skip is not None:
            if error_name in error_types_to_skip:
                if verbose:
                    print(f"  --> skipping plot")
                continue
    
        epts = np.vstack(split_df['error_points'].to_numpy())
        vpts = np.vstack(split_df['valid_points'].to_numpy())
        merge_coords = split_df[['merge_coordinate_x_nm','merge_coordinate_y_nm','merge_coordinate_z_nm']].to_numpy()
        sk_pts = np.vstack(split_df["error_branches_skeleton_points"].to_numpy())
        
        scatters.append(merge_coords)
        scatters_colors.append(split_color)
    
        epts_list.append(epts)
        vpts_list.append(vpts)
        sk_pts_list.append(sk_pts)
    
    scatters += [np.vstack(epts_list),np.vstack(vpts_list)]
    scatters_colors += [error_points_color,valid_points_color]
    
    if plot_skeleton:
        scatters.append(np.vstack(sk_pts_list))
        scatters_colors.append(error_skeleton_points_color)
    
    ipvu.plot_objects(
        mesh,
        scatters=scatters,
        scatters_colors=scatters_colors,
        scatter_size=scatter_size,
    )