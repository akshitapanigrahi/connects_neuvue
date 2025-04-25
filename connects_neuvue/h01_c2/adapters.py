import os
import datajoint as dj

os.environ['DJ_SUPPORT_ADAPTED_TYPES'] = "TRUE"  
os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"
BUCKET = "neurd-datalake"
RAW_MESH_LOC = "h01_raw_meshes"
ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
SECRET_KEY = os.environ['AWS_SECRET_KEY']
STAGE = os.path.abspath('./stage')

dj.config['stores'] = {
'graph': dict(  # s3 storage for graph
                secure = True,
                stage = STAGE,
                protocol='s3',
                endpoint='s3.amazonaws.com',
                bucket = BUCKET,
                location = 'h01_graph',
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY),

}

from pathlib import Path
def validate_filepath(filepath):
    filepath = Path(filepath)
    assert filepath.exists()
    return filepath
    
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

graph = FilePathAdapter("filepath@graph")

