import trimesh
def empty_mesh():
    return trimesh.Trimesh(vertices=np.array([]),
                          faces=np.array([]))
    
def is_mesh(obj):
    if type(obj) == type(trimesh.Trimesh()):
        return True
    else:
        return False
    
def submesh(mesh,face_idx,always_return_mesh=True):
    new_submesh = mesh.submesh([list(face_idx)],only_watertight=False,append=True)
    if not tu.is_mesh(new_submesh) and always_return_mesh:
        return tu.empty_mesh()
    else:
        return new_submesh
    
from . import trimesh_utils as tu