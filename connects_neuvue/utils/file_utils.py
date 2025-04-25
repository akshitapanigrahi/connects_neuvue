import bz2
import pickle
from pathlib import Path

def decompress_pickle(filename):
    """
    Example: 
    data = decompress_pickle('example_cp.pbz2') 
    """
    if isinstance(filename, Path):
        filename = str(filename.absolute())
    if not filename.endswith(".pbz2"):
        filename += ".pbz2"
        
    with bz2.BZ2File(filename, 'rb') as f:
        data = pickle.load(f)
    return data