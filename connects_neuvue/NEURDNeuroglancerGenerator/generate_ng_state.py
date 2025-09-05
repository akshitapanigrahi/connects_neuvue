#!/usr/bin/env python3
# generate_ng_state.py
import argparse
import boto3

from ng_utils import (
    ensure_public_read,
    ensure_segment_properties_stub,
    load_config,
    fetch_proofread_meshes,
    fetch_proofread_skeletons,
    init_skeleton_manager,
    upload_meshes_to_s3,
    upload_skeletons,
    build_viewer_link,
)

def generate_ng_state(segment_ids, config_path="config.yaml"):
    cfg = load_config(config_path)
    if segment_ids is not None:
        cfg["segment_ids"] = segment_ids

    # meshes
    frags, global_center = fetch_proofread_meshes(
        cfg["segment_ids"], cfg["api_dataset"], cfg.get("aws_secret_name"), cfg.get("resolution")
    )
    upload_meshes_to_s3(frags, cfg["bucket"], cfg["s3_base_path"])

    # skeletons
    skmgr = init_skeleton_manager(
        cfg["segmentation_layer_source"],
        cfg["bucket"],
        cfg["s3_base_path"]
    )
    skeletons = fetch_proofread_skeletons(
        cfg["segment_ids"], cfg["api_dataset"], cfg.get("aws_secret_name")
    )
    upload_skeletons(skeletons, skmgr)
    ensure_segment_properties_stub(cfg["bucket"], cfg["s3_base_path"])

    # bucket CORS + public read
    s3 = boto3.client("s3")
    s3.put_bucket_cors(
        Bucket=cfg["bucket"],
        CORSConfiguration={
            "CORSRules": [{
                "AllowedOrigins": ["*"],
                "AllowedMethods": ["GET", "HEAD"],
                "AllowedHeaders": ["*"]
            }]
        }
    )
    ensure_public_read(
        cfg["bucket"],
        prefixes=[
            cfg["s3_base_path"],                 # meshes
            f'{cfg["s3_base_path"]}/skeletons'  # whole skeletons tree
        ]
    )

    # build and return the link (pass skmgr to include skeletons)
    url = build_viewer_link(
        cfg["bucket"],
        cfg["image_layer_name"],
        cfg["segmentation_layer_name"],
        cfg["raw_layer_name"],
        cfg["proofread_layer_name"],
        cfg["s3_base_path"],
        cfg["image_layer_source"],
        cfg["segmentation_layer_source"],
        cfg["segment_ids"],
        cfg["viewer_dimensions"],
        global_center,
        cfg.get("cave_client_name"),
        skeleton_manager=skmgr,
    )
    return url

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()

    # CLI behavior preserved: uses segment_ids from config.yaml
    url = generate_ng_state(segment_ids=None, config_path=args.config)
    if url:
        print(url)

if __name__ == "__main__":
    main()