import datajoint as dj
from ..utils import (
    file_utils as fileu,
    nx_graph_utils as nxgu,
    dj_utils as dju,
    split_suggestion_utils as ssu,
    trimesh_utils as tu,
    skeleton_utils as sk,
    neuron_utils as nru,
    numpy_utils as nu,
)
import trimesh
import numpy as np

# -- trying an import of datasci_tools to be used for visualizations
try:
    from datasci_tools import ipyvolume_utils as ipvu
except:
    pass

# -- utilities from other packages --

class API:
    def __init__(
        self,
        secret_password = None,
        host = None,
        secret_dict=None,
        **kwargs):
        
        if secret_dict is None:
            secret_dict = self.secret_dict_from_password(secret_password)
            
        if host is None:
            host = "neurd-datajoint.cluster-cjc6cqmcqirl.us-east-1.rds.amazonaws.com"
            
        dj.config['database.host'] = host
        
        dj.config['database.password'] = secret_dict['password']
        dj.config['database.username'] = secret_dict['username']
        dj.config['database.user'] = secret_dict['username']
        dj.conn()
        
        from .schema import (
            AutoProofreadNeuron,
            MeshDecimation,
            SomaExtraction
        )
        
        self.soma_table = SomaExtraction.SomaInfo
        self.soma_mesh_table = SomaExtraction.SomaObj
        
        self.autoproof_table = AutoProofreadNeuron
        self.autoproof_obj_table = self.autoproof_table.Obj
        
        self.decimated_mesh_obj_table = MeshDecimation.Obj
        
        self.nucleus_id_name = "nucleus_id",
        self.centroid_name = "centroid"
        
    def secret_dict_from_password(password,username='admin'):
        return {'username': username, 'password': password}
    
    # -- table aliases --
    @property
    def proofreading_neurons_table_raw(self):
        return self.autoproof_table
    
    @property
    def proofreading_object_table(self):
        return self.autoproof_obj_table
    
    # -- utility functions for fetching
    
    def segment_id_and_split_index_from_node_name(self,name):
        return [int(k) for k in name.split("_")]
    
    def segment_id_and_split_index(
        self,
        segment_id,
        split_index = 0,
        return_dict = False):
        
        single_flag = False
        if not nu.is_array_like(segment_id):
            segment_id=[segment_id]
            split_index = [split_index]*len(segment_id)
            single_flag = True
        else:
            if not nu.is_array_like(split_index):
                split_index = [split_index]*len(segment_id)
        
        seg_ids_final = []
        sp_idxs_final = []
        for seg_id,sp_idx in zip(segment_id,split_index):
            if sp_idx == None:
                sp_idx = 0

            try:
                if "_" in seg_id:
                    seg_id,sp_idx = self.segment_id_and_split_index_from_node_name(seg_id)
            except:
                pass
            
            seg_ids_final.append(seg_id)
            sp_idxs_final.append(sp_idx)
            
        if single_flag:
            seg_ids_final = seg_ids_final[0]
            sp_idxs_final = sp_idxs_final[0]

        if not return_dict:
            return seg_ids_final,sp_idxs_final
        else:
            if single_flag:
                return dict(segment_id=seg_ids_final,split_index = sp_idxs_final)
            else:
                return [dict(segment_id=k,split_index = v) for k,v in zip(seg_ids_final,sp_idxs_final)]
    
    def segment_id_to_autoproofread_neuron_features(
        self,
        segment_id,
        statistic_names,
        split_index = 0,
        return_dict = False,
        table=None,
        ):

        """
        Purpose: To get a list of statistcs for a certain 
        neuron from the current proofreading tables

        Pseudocode: 
        1) Get the current dj stats table
        2) Restrict the table according to the neuron
        3) Fetch the features from the table

        Ex: 
        stats_names= ["cell_type_predicted",
                "spine_category",
                "nucleus_id",
                  ]

        # stats_dict = dict()
        curr_output = du.segment_id_to_autoproofread_neuron_features(segment_id=curr_key["segment_id"],
            statistic_names=stats_names,
            split_index=curr_key["split_index"],
            return_dict=True)

        """


        #1) Get the current dj stats table
        if table is None:
            curr_table = self.proofreading_neurons_table_raw
        else:
            curr_table = table

        #2) Restrict the table according to the neuron
        restr_table = curr_table & dict(segment_id = segment_id,split_index=split_index)

        #3) Fetch the features from the table
        feature_values = restr_table.fetch1(*statistic_names)

        feature_values = nu.convert_to_array_like(feature_values,include_tuple = True)

    #     if len(statistic_names) == 0:
    #         return feature_values[0]
    #     else:
        if return_dict:
            stats_dict = dict()
            for s_name,st_val in zip(statistic_names,feature_values):
                stats_dict[s_name] = st_val
            return stats_dict
        else:

            return feature_values
    
    def features_from_proofread_table(
        self,
        segment_id,
        feature_names,
        split_index = None,
        return_dict = False,
        ):
        """
        Purpose: To get any features of a segment id
        and split index 

        """

        feature_names = nu.convert_to_array_like(feature_names)

        segment_id,split_index = self.segment_id_and_split_index(segment_id,
                                     split_index)

        ret_value = self.segment_id_to_autoproofread_neuron_features(
            segment_id=segment_id,
            split_index = split_index,
            statistic_names=feature_names,
            return_dict = return_dict
        )

        if len(feature_names) == 1:
            return ret_value[0]
        else:
            return ret_value
    
    
    # -- fetching soma information
    def closest_soma_idx_from_segment_id_soma_center(
        self,
        segment_id,
        soma_center,
        verbose = False):
        """
        Purpose: to find the closest soma
        index to a given coordinates of a segment
        id

        Pseudocode: 
        1) Get the coordinates of all the somas
        2) find the lowest distance to soma center

        Ex: 
        du.closest_soma_idx_from_segment_id_soma_center(

        segment_id= 864691135697376277,
        soma_center = [229587,100033,17961],
        verbose = False,
        )
        """

        soma_center= np.array(soma_center).astype("float")

        soma_x,soma_y,soma_z,soma_idx = (self.soma_table & dict(
                    segment_id = segment_id,
                )).fetch("centroid_x","centroid_y","centroid_z","soma_idx")

        soma_centers = np.vstack([soma_x,soma_y,soma_z]).T.astype("int")
        soma_dists = np.linalg.norm(soma_centers - soma_center,axis=1)
        winning_soma_idx = soma_idx[np.argmin(soma_dists)]

        if verbose:
            print(f"soma_dists = {soma_dists}")
            print(f"winning_soma_idx = {winning_soma_idx}")

        return winning_soma_idx
    
    def fetch_soma_center(
        self,
        segment_id,
        split_index = None,
        verbose = False,
        return_nm = False,
        return_xyz = False,
        ):
        """
        Purpose: To fetch the soma center

        """
        segment_id,split_index = self.segment_id_and_split_index(segment_id,
                                         split_index)
        
        centroid_name = self.centroid_name

        if return_nm:
            feature_names = [f"{centroid_name}_x_nm",f"{centroid_name}_y_nm",f"{centroid_name}_z_nm"]
        else:
            feature_names = [f"{centroid_name}_x",f"{centroid_name}_y",f"{centroid_name}_z"]

        soma_x,soma_y,soma_z = self.features_from_proofread_table(
        segment_id = segment_id,
        split_index = split_index,
        feature_names = feature_names,
        return_dict = False,
        )
        if return_xyz:
            return soma_x,soma_y,soma_z 

        return np.array([soma_x,soma_y,soma_z])
    
    def get_soma_faces(
        self,
        segment_id=None,
        soma_index=None,
        table=None,
        soma_center = None, 
        verbose = False,):
        if soma_center is not None:
            soma_index = self.closest_soma_idx_from_segment_id_soma_center(
                segment_id,
                soma_center,
                verbose = verbose)
        else:
            soma_index = 1
            
        if table is None:
            key = dict(segment_id=segment_id,soma_index=soma_index)
            table = (self.soma_table * self.soma_mesh_table & key)

        return table.fetch1("soma_obj")
    
    def fetch_soma_mesh(
        self,
        segment_id=None,
        split_index=None,
        soma_index=None,
        original_mesh = None,
        plot_soma = False,
        verbose = False,
        **kwargs
        ):
        
        soma_center = self.fetch_soma_center(
            segment_id,
            split_index,
        )
        soma_x,soma_y,soma_z = soma_center[0],soma_center[1],soma_center[2]
        
        if verbose:
            print(f"For segment_id = {segment_id}, split_index = {split_index} ")
            print(f"{self.centroid_name}_x = {soma_x}, {self.centroid_name}_y= {soma_y}, {self.centroid_name}_z= {soma_z}")
        
        
        soma_faces = self.get_soma_faces(
            segment_id=segment_id,
            soma_index=soma_index,
            soma_center=soma_center,
            **kwargs)
        
        if original_mesh is None:
            original_mesh = self.fetch_segment_id_mesh(segment_id)
        soma_mesh = tu.submesh(original_mesh,soma_faces)
        
        if plot_soma:
            print(f"Plotting soma: {soma_mesh}")
            ipvu.plot_objects(soma_mesh)
            
        return soma_mesh
        
    
    # -- fetching proofread meshes and skeletons --
    def fetch_segment_id_mesh(
        self,
        segment_id:int=None,
        verbose = False,
        plot = False,
        ) -> trimesh.Trimesh:
        """
        Purpose: retrieve a decimated segment id mesh. 

        Parameters
        ----------
        segment_id : int, optional
            neuron segment id, by default None


        Returns
        -------
        trimesh.Trimesh
            decimated mesh for segment id
        """
        if verbose:
            print(f'fetching decimated mesh for segment {segment_id}')
        mesh = (self.decimated_mesh_obj_table & dict(segment_id = segment_id)).fetch1('obj')
            
        if plot:
            ipvu.plot_objects(mesh)
        return mesh
    
    def fetch_compartment_faces(
        self,
        compartment,
        segment_id,
        split_index=0,
        ):
        mesh_faces = (
            self.proofreading_object_table & dict(
                segment_id=segment_id,
                split_index=split_index)
            ).fetch1(f"{compartment}_mesh_faces")
        return mesh_faces
    
    def fetch_compartment_mesh(
        self,
        compartment,
        segment_id,
        split_index=0,
        original_mesh=None,
        verbose=False,
        plot_mesh = False):
        """
        Purpose: To get the mesh belonging to a certain compartment

        Ex: 
        original_mesh = self.fetch_segment_id_mesh(segment_id)

        comp_mesh = pv.fetch_compartment_mesh("apical_shaft",
                                  segment_id,
                                  split_index,
                                original_mesh=original_mesh,
                                              verbose = True,
                                              plot_mesh = True,
                                 )
        """
        if compartment == "apical_total":
            compartment_faces = np.concatenate([self.fetch_compartment_faces(compartment = c,
                                                                 segment_id=segment_id,
                                                                 split_index=split_index,
                                                             ) for c in nru.apical_total]).astype("int")
        else:
            compartment_faces = self.fetch_compartment_faces(compartment = compartment,
                                                                 segment_id=segment_id,
                                                                 split_index=split_index,
                                                             )
        if verbose:
            print(f"# of faces = {len(compartment_faces)}")

        if original_mesh is None:
            original_mesh = self.fetch_segment_id_mesh(segment_id)

        compartment_mesh = original_mesh.submesh([compartment_faces],append=True)

        if not tu.is_mesh(compartment_mesh):
            compartment_mesh = tu.empty_mesh()

        if plot_mesh:
            print(f"Plotting {compartment}")
            ipvu.plot_objects(original_mesh,
                             meshes=[compartment_mesh],
                             meshes_colors="red")

        return compartment_mesh
    
    def fetch_proofread_mesh(
        self,
        segment_id,
        split_index = 0,
        original_mesh = None,
        #return_error_mesh = False,
        plot_mesh = False
        ):
        if original_mesh is None:
            original_mesh = self.fetch_segment_id_mesh(segment_id)

        proof_mesh = self.fetch_compartment_mesh("neuron",
                                              segment_id,
                                              split_index,
                                              original_mesh=original_mesh,
                                              )
        # if return_error_mesh:
        #     error_mesh = tu.subtract_mesh(original_mesh,proof_mesh)

        #     if plot_mesh:
        #         nviz.plot_objects(proof_mesh,
        #                          meshes=[error_mesh],
        #                          meshes_colors=["red"])

        #     return proof_mesh,error_mesh

        if plot_mesh:
            nviz.plot_objects(proof_mesh,
                             #meshes=[error_mesh],
                             #meshes_colors=["red"]
                             )
        return proof_mesh
    

    def fetch_proofread_mesh_axon(
        self,
        segment_id,
        split_index = 0,
        plot_mesh = False,
        original_mesh = None,
        ):

        return self.fetch_compartment_mesh(
            compartment = "axon",
            segment_id=segment_id,
            split_index = split_index,
            original_mesh=original_mesh,
            plot_mesh=plot_mesh)
    
    def fetch_proofread_mesh_dendrite(
        self,
        segment_id,
        split_index = 0,
        plot_mesh = False,
        original_mesh = None,
        ):

        return self.fetch_compartment_mesh(
            compartment = "dendrite",
            segment_id=segment_id,
            split_index = split_index,
            plot_mesh=plot_mesh,
            original_mesh=original_mesh,
            
        )
    
    def fetch_compartment_skeleton(
        self,
        compartment,
        segment_id,
        split_index = 0,
        verbose = False,
        plot_skeleton = False,
        plot = False,
        original_mesh = None,
        ):
        """
        Purpose: To retrieve the datajoint
        stored skeleton for that compartment

        Ex: 
        comp_skeleton = pv.fetch_compartment_skeleton("apical_shaft",
                                 segment_id,
                                 split_index,
                                plot_skeleton = True)
        """
        segment_id,split_index = self.segment_id_and_split_index(segment_id,split_index)

        if compartment == "apical_total":
            comp_skeleton = sk.stack_skeletons([(self.proofreading_object_table & dict(segment_id=segment_id,
                                            split_index=split_index)).fetch1(f"{c}_skeleton") for c in apu.apical_total])
        else:
            comp_skeleton = (self.proofreading_object_table & dict(segment_id=segment_id,
                                            split_index=split_index)).fetch1(f"{compartment}_skeleton")

        if len(comp_skeleton) == 0:
            comp_skeleton = sk.empty_skeleton()
        if verbose:
            print(f"{compartment} skeleton = {sk.calculate_skeleton_distance(comp_skeleton)}")

        if plot_skeleton or plot:
            if original_mesh is None:
                original_mesh = self.fetch_segment_id_mesh(segment_id)
            nviz.plot_objects(original_mesh,
                             skeletons = [comp_skeleton])

        return comp_skeleton
        
    def fetch_proofread_skeleton(
        self,
        segment_id,
        split_index=0,
        verbose = False,
        plot_skeleton = False,
        **kwargs
        ):
        """
        Ex: 
        pv.fetch_proofread_skeleton(segment_id,
                               split_index,
                               plot_skeleton=True,
                               original_mesh=original_mesh)

        """
        return self.fetch_compartment_skeleton("neuron",
                                            segment_id,
                                            split_index,
                                            verbose = verbose,
                                            plot_skeleton=plot_skeleton,
                                            **kwargs)
        
    def fetch_proofread_skeleton_axon(
        self,
        segment_id,
        split_index = 0,
        plot_skeleton = False
        ):

        return self.fetch_compartment_skeleton(
            compartment = "axon",
            segment_id=segment_id,
            split_index = split_index,
            plot_skeleton=plot_skeleton)
    
    def fetch_proofread_skeleton_dendrite(
        self,
        segment_id,
        split_index = 0,
        plot_skeleton = False
        ):

        return self.fetch_compartment_skeleton(
            compartment = "dendrite",
            segment_id=segment_id,
            split_index = split_index,
            plot_skeleton=plot_skeleton
        )
        
    def fetch_proofread_skeleton(
        self,
        segment_id,
        split_index=0,
        verbose = False,
        plot_skeleton = False,
        **kwargs
        ):
        """
        Ex: 
        pv.fetch_proofread_skeleton(segment_id,
                               split_index,
                               plot_skeleton=True,
                               original_mesh=original_mesh)

        """
        return self.fetch_compartment_skeleton("neuron",
                                            segment_id,
                                            split_index,
                                            verbose = verbose,
                                            plot_skeleton=plot_skeleton,
                                            **kwargs)
    
    
    
    # -- graph fetching --

    def nx_graph_autoproof_from_segment_id(
        self,
        segment_id,
        split_index = 0,
        add_bounding_box = True):
        restr_table = (self.autoproof_obj_table & dict(
            segment_id=segment_id,
            split_index = split_index
            )
        )
        G_file = restr_table.fetch1('neuron_graph')
        G = fileu.decompress_pickle(G_file)
        if add_bounding_box:
            nxgu.add_bounding_boxes_to_graph(G)
        return G
    
    def autoproof_segment_metadata_df(self,table=None):
        if table is None:
            table = self.autoproof_table
        return dju.df_from_table(table)
    
    
    def red_blue_split_df_from_segment_id(self,segment_id,split_index = 0,return_df = True):
        
        key = dict(segment_id=segment_id,split_index = split_index)
        red_blue_suggestions = (self.autoproof_obj_table & key).fetch1("red_blue_suggestions")
        if red_blue_suggestions is None:
            print(f"the red_blue_suggestions saved were None")
            return None
        return ssu.red_blue_suggestion_dicts(red_blue_suggestions,return_df=return_df,**key)
    