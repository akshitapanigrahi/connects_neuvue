import datajoint as dj
from .adapters import (
    raw_mesh,
    decimated_mesh,
    soma_faces,
    glia_faces,
    nucleus_faces,
    decomposition,
    skeletons,
    graph,
    auto_proof_meshes_dtype,
    auto_proof_skeletons_dtype,
)

h01 = dj.create_virtual_module('h01_process', 'h01_process')
schema = dj.Schema('h01_process')


@schema
class MeshDecimation(dj.Computed):
    class Obj(dj.Part):
        definition = """
        -> master
        ---
        obj : <decimated_mesh>
        """
        
@schema
class SomaExtraction(dj.Computed):
    class SomaInfo(dj.Part):
        pass
    class SomaObj(dj.Part):
        definition = """                               # Soma mesh face indices, saved to .npy file
        -> SomaExtraction
        -> SomaExtraction.SomaInfo
        ---
        soma_obj : <soma_faces>
        """

@schema
class AutoProofreadNeuron(dj.Computed):
    definition='''
    -> h01.DecompositionCellType
    ---
    '''
    class Obj(dj.Part):
        definition="""
        ->master
        ---
        neuron_mesh_faces: <auto_proof_meshes_dtype>     # face indices for neuron_mesh_faces of final proofread neuron
        neuron_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for neuron_skeleton of final proofread neuron


        dendrite_mesh_faces: <auto_proof_meshes_dtype>     # face indices for dendrite_mesh_faces of final proofread neuron
        axon_mesh_faces: <auto_proof_meshes_dtype>     # face indices for axon_mesh_faces of final proofread neuron
        basal_mesh_faces: <auto_proof_meshes_dtype>     # face indices for basal_mesh_faces of final proofread neuron
        apical_mesh_faces: <auto_proof_meshes_dtype>     # face indices for apical_mesh_faces of final proofread neuron
        apical_tuft_mesh_faces: <auto_proof_meshes_dtype>     # face indices for apical_tuft_mesh_faces of final proofread neuron
        apical_shaft_mesh_faces: <auto_proof_meshes_dtype>     # face indices for apical_shaft_mesh_faces of final proofread neuron
        oblique_mesh_faces: <auto_proof_meshes_dtype>     # face indices for oblique_mesh_faces of final proofread neuron

        dendrite_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for dendrite_skeleton of final proofread neuron
        axon_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for axon_skeleton of final proofread neuron
        basal_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for basal_skeleton of final proofread neuron
        apical_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for apical_skeleton of final proofread neuron
        apical_tuft_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for apical_tuft_skeleton of final proofread neuron
        apical_shaft_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for apical_shaft_skeleton of final proofread neuron
        oblique_skeleton: <auto_proof_skeletons_dtype>     # skeleton array for oblique_skeleton of final proofread neuron


        limb_branch_to_cancel: longblob # stores the limb information from 
        red_blue_suggestions=NULL: longblob
        split_locations=NULL: longblob
        split_locations_before_filter=NULL: longblob
        neuron_graph=NULL: <graph> #the graph for the 
        decomposition=NULL: <decomposition>
        """