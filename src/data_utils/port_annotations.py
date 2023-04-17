"""
Given paths to two COCO annotated datasets (directories with images and each containing
an _annotations.coco.json file), this script ports annotations for same name files from
the first dataset to the second, writing the annotations into its _annotations.coco.json.
Note that this does not deconflict with any existing annotations there may be in the
destination dir. Any annotations added for a file will be in addition to any annotations
it may already have even if these result in duplicate annotations.
"""

import os
import sys
import shutil
import json
from copy import deepcopy
import torchvision


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Call is: python {sys.argv[0]} <src dataset dir> <dest dataset dir>")
        sys.exit(0)

    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]

    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(1)

    if not os.path.exists(dest_dir):
        print(f"ERROR: {dest_dir} already exists")
        sys.exit(2)


    # Verify src_dir has _annotations.coco.json
    src_annots_path = os.path.join(src_dir, "_annotations.coco.json")
    if not os.path.exists(src_annots_path):
        print(f"ERROR: No annotations file f{src_annots_path}")
        sys.exit(3)

    # Verify dest_dir has _annotations.coco.json
    dest_annots_path = os.path.join(dest_dir, "_annotations.coco.json")
    if not os.path.exists(dest_annots_path):
        print(f"ERROR: No annotations file f{dest_annots_path}")
        sys.exit(3)

    src_annots = None
    with open(src_annots_path,"r") as fd:
        src_annots = json.load(fd)
    if src_annots is None:
        print(f"ERROR: Reading {src_annots_path}")

    dest_annots = None
    with open(dest_annots_path,"r") as fd:
        dest_annots = json.load(fd)
    if dest_annots is None:
        print(f"ERROR: Reading {dest_annots}")

    src_images = {i["file_name"]:["id"] for i in src_annots["images"]}
    dest_images = {i["file_name"]:["id"] for i in dest_annots["images"]}

    imgs_to_port = list(set(dest_images.keys()).intesection(src_images.keys()))

    src_dataset = torchvision.datasets.CocoDetection(src_dir, src_annots_path)
    dest_dataset = torchvision.datasets.CocoDetection(dest_dir, dest_annots_path)

    # The destination dataset may already have some annotations. We need to make
    # sure not to reuse annotation ids.
    dest_ann_ids = dest_dataset.coco.getAnnIds()
    next_ann_id = 1
    if len(dest_ann_ids) > 0:
        next_ann_id = max(dest_ann_ids) + 1

    """
    The COCO json format has an "images" list and an "annotations" list.
    The images list has entries like:

    {
        "id":1107,
        "license":1,
        "file_name":"filename.jpg",
        "height":1080,
        "width":1920,
        "date_captured":"2023-04-05T18:01:08+00:00"
    }

    The annotations liost has entries like

    {
        "id":1028,
        "image_id":354,
        "category_id":2,
        "bbox":[234,512,114.6,132.36],
        "area":15168.456,
        "segmentation":[],
        "iscrowd":0
    }

    The image_id field ties the annotation to the image.
    
    """

    for fn in imgs_to_port:
        src_img_id = src_images[fn]
        dest_img_id = dest_images[fn]

        src_img_annots = src_dataset.coco.imgToAnns[src_img_id]
        for annot in src_img_annots:
            new_annot = deepcopy(annot)

            # Tie the annotation to the dest image
            new_annot["image_id"] = dest_img_id

            # Give it a new annotation id that doesn't conflict with preexisting annotations
            new_annot["id"] = next_ann_id
            next_ann_id += 1

            # Add the new annotation to the dest json
            dest_annots["annotations"].append(new_annot)

    # Create backup of destination json
    backup_path = dest_annots_path + "_bak"
    shutil.copyfile(dest_annots_path, backup_path)

    with open(dest_annots_path, "w") as fd:
        json.dump(dest_annots, dest_annots_path)
