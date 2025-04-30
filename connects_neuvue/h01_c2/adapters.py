import datajoint as dj
import numpy as np
import json
import h5py
import trimesh
from pathlib import Path
from collections import namedtuple
import  datasci_tools.system_utils as su 
import os

os.environ['DJ_SUPPORT_ADAPTED_TYPES'] = "TRUE"  
os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"
BUCKET = "neurd-datalake"
RAW_MESH_LOC = "h01_raw_meshes"
ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
SECRET_KEY = os.environ['AWS_SECRET_KEY']
STAGE = os.path.abspath('./stage')

from pathlib import Path
class Adapter(dj.AttributeAdapter):
    attribute_type = ''

    def __init__(self, attribute_type):
        self.attribute_type = attribute_type
        super().__init__()

class FilePathAdapter(Adapter):    
    def put(self, filepath):
        return validate_filepath(filepath)

    def get(self, filepath):
        return validate_filepath(filepath)

class NumpyAdapter(FilePathAdapter):
    def get(self, filepath):
        filepath = super().get(filepath)
        return np.load(filepath, mmap_mode='r', allow_pickle = True)['arr_0']

class TrimeshAdapter(FilePathAdapter):
    def get(self, filepath):
        filepath = super().get(filepath)
        if filepath.suffix == '.h5':
            mesh = adapt_mesh_hdf5(filepath, parse_filepath_stem=False, return_type='namedtuple')
            vertices=mesh.vertices
            faces=mesh.faces
            return trimesh.Trimesh(vertices = vertices, faces = faces)
    
class PbzPickleAdapter(FilePathAdapter):
    ''' read compressed pickle files of .pbz2 format
    return numpy array'''
    def get(self, filepath):
        filepath = super().get(filepath)
        return su.decompress_pickle(filepath)
    
def validate_filepath(filepath):
    filepath = Path(filepath)
    assert filepath.exists()
    return filepath

def adapt_mesh_hdf5(filepath, parse_filepath_stem=True, filepath_has_timestamp=False, separator='__', timestamp_fmt="%Y-%m-%d_%H:%M:%S", return_type='namedtuple', as_lengths=False):
    """
    Reads from a mesh hdf5 and returns vertices, faces and additional information in the form of a namedtuple or optionally
        as a dictionary, or optionally as separate variables.
        
    :param filepath: File path pointing to the hdf5 mesh file. 
    :param parse_filepath_stem: (bool) 
        If True: Attempts to parse filepath stem
        If False: Skips parsing of filepath stem
    :param filepath_has_timestamp: (bool) Toggles format of expected filename stem to parse.
        If True: expects format '<segment_id><separator><timestamp>'.h5
        If False: expects format '<segment_id>'.h5
    :param timestamp_separator: (str) 
    :param timestamp_fmt:
    :param return_type: Options = {
        namedtuple = return a namedtuple with the following fields {
            vertices = vertex array
            faces = face array
            segment_id = segment id of mesh if parsed else np.nan
            timestamp = timestamp mesh was computed if parsed else ''
            filepath = filepath of mesh
        }
        dict = return a dictionary with keys as in namedtuple
        separate: returns separate variables in the following order {
            vertex array 
            face array
            dictionary with keys segment_id, timestamp, filepath as in namedtuple
            }
    :param as_lengths: Overrides return_type and instead returns:
            Length of the vertex array
            Length of face array
            dictionary with keys segment_id, timestamp, filepath as in namedtuple
        This is done without pulling the mesh into memory, which makes it far more space and time efficient.
    }
    """
    Mesh = namedtuple('Mesh', ['vertices', 'faces', 'segment_id', 'timestamp', 'filepath'])
    filepath = Path(filepath)
    #print('adapt', filepath)
    # try to parse filepath
    info_dict = {'filepath': filepath}
    defaults = {'segment_id': np.nan, 'timestamp': ''}
    if parse_filepath_stem:
        try:
            if not filepath_has_timestamp:
                segment_id = filepath.stem
                info_dict.update({**{'segment_id': int(segment_id), 'timestamp': ''}})
            else:
                segment_id, timestamp = filepath.stem.split(separator)
                timestamp = datetime.strptime(timestamp, timestamp_fmt)
                info_dict.update({**{'segment_id': int(segment_id), 'timestamp': timestamp}})
        except:
            info_dict.update({**defaults})
            logger.warning('Could not parse mesh filepath.')
    else:
        info_dict.update({**defaults})

    # Load the mesh data
    with h5py.File(filepath, 'r') as f:
        if as_lengths:
            n_vertices = f['vertices'].shape[0]
            n_faces = int(f['faces'].shape[0] / 3)
            return n_vertices, n_faces, info_dict
        
        vertices = f['vertices'][()].astype(np.float64)
        faces = f['faces'][()].reshape(-1, 3).astype(np.uint32)
    
    # Return options
    return_dict = dict(
                vertices=vertices,
                faces=faces,
                **info_dict
                )
    if return_type == 'namedtuple':
        return Mesh(**return_dict)
    elif return_type == 'dict':
        return return_dict
    elif return_type == 'separate':
        return vertices, faces, info_dict
    else:
        raise TypeError(f'return_type does not accept {return_type} argument')

dj.config['stores'] = {
'raw_meshes': dict(  # s3 storage for raw meshes
               secure = True,
               stage = STAGE,
               protocol='s3',
               endpoint='s3.amazonaws.com:9000',
               bucket = BUCKET,
               location = RAW_MESH_LOC,
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY),
'decimated_meshes': dict(  # s3 storage for decimated meshes
               secure = True,
               stage = STAGE,
               protocol='s3',
               endpoint='s3.amazonaws.com',
               bucket = BUCKET,
               location = 'h01_decimated_meshes',
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY),
'soma_faces': dict(  # s3 storage for somas
               secure = True,
               stage = STAGE,
               protocol='s3',
               endpoint='s3.amazonaws.com',
               bucket = BUCKET,
               location = 'h01_soma_faces',
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY),
'nucleus_faces': dict(  # s3 storage for nuclei
               secure = True,
               stage = STAGE,
               protocol='s3',
               endpoint='s3.amazonaws.com',
               bucket = BUCKET,
               location = 'h01_nucleus_faces',
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY),
'glia_faces': dict(  # s3 storage for glia
               secure = True,
               stage = STAGE,
               protocol='s3',
               endpoint='s3.amazonaws.com',
               bucket = BUCKET,
               location = 'h01_glia_faces',
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY),
'decomposition': dict(  # s3 storage for decompositions
               secure = True,
               stage = STAGE,
               protocol='s3',
               endpoint='s3.amazonaws.com',
               bucket = BUCKET,
               location = 'h01_decomposition',
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY),
'skeletons': dict(  # s3 storage for skeletons
               secure = True,
               stage = STAGE,
               protocol='s3',
               endpoint='s3.amazonaws.com',
               bucket = BUCKET,
               location = 'h01_skeletons',
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY),
'graph': dict(  # s3 storage for graph
                secure = True,
                stage = STAGE,
                protocol='s3',
                endpoint='s3.amazonaws.com',
                bucket = BUCKET,
                location = 'h01_graph',
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY),
'auto_proof_meshes': dict(  # s3 storage for autoproof_mesh
                secure = True,
                stage = STAGE,
                protocol='s3',
                endpoint='s3.amazonaws.com',
                bucket = BUCKET,
                location = 'h01_auto_proof_meshes',
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY),
'auto_proof_skeletons': dict(  # s3 storage for autoproof_skeletons
                secure = True,
                stage = STAGE,
                protocol='s3',
                endpoint='s3.amazonaws.com',
                bucket = BUCKET,
                location = 'h01_auto_proof_skeletons',
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY),
}

raw_mesh = TrimeshAdapter('filepath@raw_meshes')
decimated_mesh = TrimeshAdapter('filepath@decimated_meshes')
soma_faces = NumpyAdapter('filepath@soma_faces')
glia_faces = NumpyAdapter('filepath@glia_faces')
nucleus_faces = NumpyAdapter('filepath@nucleus_faces')
decomposition = FilePathAdapter('filepath@decomposition')
skeletons = PbzPickleAdapter('filepath@skeletons')
graph = FilePathAdapter("filepath@graph")
auto_proof_meshes_dtype = PbzPickleAdapter('filepath@auto_proof_meshes')
auto_proof_skeletons_dtype = PbzPickleAdapter('filepath@auto_proof_skeletons')

