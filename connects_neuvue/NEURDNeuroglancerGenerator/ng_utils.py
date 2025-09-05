# ng_utils.py
import importlib
import json
import boto3
import numpy as np
import yaml
from caveclient import CAVEclient
from nglui.skeletons import SkeletonManager
from nglui.statebuilder import SegmentationLayer
from connects_neuvue.utils import aws_utils as aws
from cloudvolume.mesh import Mesh
from nglui import statebuilder
from nglui.statebuilder.ngl_components import Source
import random

def load_config(path="config.yaml"):
    with open(path, "r") as fp:
        return yaml.safe_load(fp)

def ensure_public_read(bucket, prefixes):
    """
    Grant public GET for all objects under each prefix (and common child paths like /segment_properties).
    """
    resources = []
    for p in prefixes:
        # main prefix
        resources.append(f"arn:aws:s3:::{bucket}/{p}/*")
        resources.append(f"arn:aws:s3:::{bucket}/{p}/info")
        # explicit child for skeleton segment_properties
        resources.append(f"arn:aws:s3:::{bucket}/{p}/segment_properties/*")
        resources.append(f"arn:aws:s3:::{bucket}/{p}/segment_properties/info")

    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowPublicReadForNeuroglancerAssets",
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:GetObject"],
            "Resource": resources
        }]
    }

    boto3.client("s3").put_bucket_policy(Bucket=bucket, Policy=json.dumps(policy))

def ensure_segment_properties_stub(bucket, s3_base_path):
    """
    Write a minimal segment_properties/info so Neuroglancer won't 404/403.
    """
    s3 = boto3.client("s3")
    key = f"{s3_base_path}/skeletons/segment_properties/info"
    body = json.dumps({
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [],
            "properties": []
        }
    })
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

def fetch_proofread_meshes(segment_ids, api_dataset, aws_secret_name, resolution):
    """
    Returns a dict mapping segment_id -> cloudvolume fragment bytes.
    """
    secret = aws.get_secret(aws_secret_name) if aws_secret_name else aws.get_secret()
    api_mod = importlib.import_module(f"connects_neuvue.{api_dataset}.api")
    fetcher = api_mod.API(secret_dict=secret)
    
    all_vertices = []
    out = {}
    for sid in segment_ids:
        decim = fetcher.fetch_segment_id_mesh(segment_id=sid)
        proof = fetcher.fetch_proofread_mesh(original_mesh=decim, segment_id=sid)
        verts = np.array(proof.vertices, dtype=np.float32)
        faces = np.array(proof.faces, dtype=np.uint32)
        cv_mesh = Mesh(verts, faces, segid=sid)
        out[sid] = cv_mesh.to_precomputed() # convert mesh to binary format compatible with neuroglancer
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    global_center = (all_vertices.mean(axis=0)/resolution).astype(int).tolist()
    # center all meshes around the global center
    return out, global_center

def upload_meshes_to_s3(fragments, bucket, s3_base_path):
    """
    1) Fetches proofread meshes as Neuroglancer‐precomputed binary arrays
    2) Uploads each array to S3 at `<prefix>/<segment_id>`.
    3) Writes a manifest index referencing that blob.
    4) Writes `info` with type 'neuroglancer_legacy_mesh' so NG can auto-detect.
    5) Enables CORS for GET/HEAD on the bucket.
    """

    seg_props = {
    "@type": "neuroglancer_segment_properties",
    "inline": {
        "ids": [],       
        "properties": []
    }
    }

    s3 = boto3.client('s3')

    s3.put_object(
        Bucket=bucket,
        Key=f"{s3_base_path}/info",
        Body=json.dumps(seg_props),
        ContentType="application/json"
    )

    cors_rules = {
        'CORSRules': [{
            'AllowedOrigins': ['*'],
            'AllowedMethods': ['GET', 'HEAD'],
            'AllowedHeaders': ['*'],
        }]
    }
    s3.put_bucket_cors(Bucket=bucket, CORSConfiguration=cors_rules)
    
    for sid, mesh_bytes in fragments.items():
        data_key = f"{s3_base_path}/{sid}"
        s3.put_object(
            Bucket=bucket,
            Key=data_key,
            Body=mesh_bytes,
            ContentType='application/octet-stream'
        )
        manifest = {"fragments": [str(sid)]}
        manifest_key = f"{s3_base_path}/{sid}:0"
        s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=json.dumps(manifest),
            ContentType='application/json'
        )

    info = {"@type": "neuroglancer_legacy_mesh"}
    info_key = f"{s3_base_path}/info"
    s3.put_object(
        Bucket=bucket,
        Key=info_key,
        Body=json.dumps(info),
        ContentType='application/json',
    )

def fetch_proofread_skeletons(segment_ids, api_dataset, aws_secret_name=None):
    """
    Returns dict mapping segment_id -> (vertices, edges) arrays for proofread skeletons.
    """
    secret = aws.get_secret(aws_secret_name) if aws_secret_name else aws.get_secret()
    api_mod = importlib.import_module(f"connects_neuvue.{api_dataset}.api")
    fetcher = api_mod.API(secret_dict=secret)

    out = {}
    for sid in segment_ids:
        # fetch ndarray of shape (M,2,3) or flattened list of points
        raw = fetcher.fetch_proofread_skeleton(segment_id=sid)
        # ensure shape [N,3] and edges list
        all_pts = raw.reshape(-1, 3)
        verts, inv = np.unique(all_pts, axis=0, return_inverse=True)
        edges = inv.reshape(-1, 2)
        out[sid] = (verts, edges)
    return out

def init_skeleton_manager(seg_source, bucket, s3_base_path, vertex_attributes=["radius"], shader=None):
    """
    Initialize and return a SkeletonManager instance.
    """
    cloudpath = f"s3://{bucket}/{s3_base_path}/skeletons"
    skmgr = SkeletonManager(
        segmentation_source=seg_source,
        cloudpath=cloudpath,
        vertex_attributes=vertex_attributes,
        initialize_info=True,
        shader=shader
    )

    return skmgr

def upload_skeletons(skeletons, skmgr):
    """
    Upload each skeleton via the SkeletonManager.
    Returns the Neuroglancer skeleton source URL.
    """
    for sid, (verts, edges) in skeletons.items():
        skmgr.upload_skeleton(
            root_id=sid,
            vertices=verts,
            edges=edges,
            vertex_attribute_data={"radius": np.full(len(verts), 10.0, dtype=np.float32)}
        )
    return skmgr.skeleton_source

def random_hex_color():
    """Return a random bright-ish hex color."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def build_viewer_link(
    bucket,
    image_layer_name,
    seg_layer_name,
    raw_layer_name,
    proofread_layer_name,
    s3_base_path,
    image_source,
    seg_source,
    segment_ids,
    dimensions,
    global_center,
    cave_client_name,
    skeleton_manager,
):
    """
    Returns a Neuroglancer URL with image, segmentation, meshes, and optional skeletons.
    """
    # Choose Neuroglancer backend (spelunker)    
    statebuilder.site_utils.set_default_neuroglancer_site(site_name='spelunker')

    vs = statebuilder.ViewerState(dimensions=dimensions, position=global_center)
    vs = (
        vs.add_image_layer(source=image_source, name=image_layer_name)
        .add_segmentation_layer(source=seg_source, name=seg_layer_name)
    )

    # raw mesh/seg layer (no proofread)
    raw_layer = SegmentationLayer(name=raw_layer_name)
    raw_layer = raw_layer.add_source(seg_source).add_segments(segment_ids)
    vs.add_layer(raw_layer)
    
    proof_skel_src = skeleton_manager.skeleton_source
    proof_mesh_src = f"precomputed://https://{bucket}.s3.amazonaws.com/{s3_base_path}"

    # build an anchoring Source that disables the dataset's mesh/skeleton subsources so the raw seg layer doesn't show in proofread layer
    seg_anchor = Source(
        url=seg_source,
        subsources={            
            "default": True,
            "bounds":  True,
            "properties": True,
            "mesh": False,        
            "skeletons": False   
        },
        enable_default_subsources=False
    )

    color_map = {str(sid): random_hex_color() for sid in segment_ids}

    # proofread mesh/seg layer
    proof_layer = SegmentationLayer(name=proofread_layer_name)
    proof_layer = (
        proof_layer
        .add_source(seg_anchor)           # anchor the layer
        .add_segments(segment_ids)        # add all your segments
        .add_source(proof_mesh_src)       # proofread mesh
        .add_source(proof_skel_src)       # proofread skeleton
        .add_segment_colors(color_map)    # assign random colors
    )
    vs.add_layer(proof_layer)

    client = CAVEclient(cave_client_name) # requires cave to generate shortcut url
    return vs.to_browser(shorten=True, client=client, browser='safari')
