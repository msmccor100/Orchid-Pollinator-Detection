"""
Given one coco dataset (the child) that has no test or val split and where this 
dataset has images filenames which are also represented in another dataset 
(the parent) that does have a test split, this script splits the child dataset in 
train/val/test to match the parent dataset. The child dataset is not altered in 
place, but rather a new dataset is created called <child dir>_split. (Why not just 
use the parent directly? Because the parent may have alterations, like augmentations
that we don't want.)

The directory structure is expected to be:
<parent dataset dir>/
    train/
        _annotations.coco.json
        <images>
    test/
        _annotations.coco.json
        <images>

    [and optionally]
    val/ 
        _annotations.coco.json
        <images>

<child dataset dir>/
    train/
        _annotations.coco.json
        <images>

    
call is: 
coco_match_test_split.py <parent dataset dir> <child dataset dir>
"""

import os
import sys
import shutil
import json
from copy import deepcopy
import supervision as sv
import torchvision
import numpy as np


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(f"Call is: python {sys.argv[0]} <parent dataset dir> <child dataset dir>")
        sys.exit(0)

    prt_dir = sys.argv[1]
    chd_dir = sys.argv[2]
    if chd_dir[-1] in ['\\','/']:
        chd_dir = chd_dir[:-1]
    dst_dir = chd_dir + "_split"

    if not os.path.exists(prt_dir):
        print(f"ERROR: {prt_dir} does not exist")
        sys.exit(1)

    if not os.path.exists(chd_dir):
        print(f"ERROR: {chd_dir} does not exist")
        sys.exit(2)

    chd_train = os.path.join(chd_dir, "train")
    chd_annots_path = os.path.join(chd_train,"_annotations.coco.json")
    if not os.path.exists(chd_annots_path):
        print(f"ERROR: {chd_annots_path} does not exist")
        sys.exit(2)

    if os.path.exists(os.path.join(chd_dir, "test")):
        print(f"ERROR: child dataset already has test split")
        sys.exit(3)

    if os.path.exists(os.path.join(chd_dir, "val")):
        print(f"ERROR: child dataset already has validation split")
        sys.exit(4)

    os.mkdir(dst_dir)
    with open(chd_annots_path,"r") as fd:
        chd_annots = json.load(fd)

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

    chd_fn2ID = {im["file_name"]:im["id"] for im in chd_annots["images"]}

    chd_dataset = torchvision.datasets.CocoDetection(chd_train, chd_annots_path)

    for split_name in ["train", "test", "val"]:

        prt_split_path = os.path.join(prt_dir, split_name)
        if not os.path.exists(prt_split_path):
            print(f"WARNING: Parent dataset has no {split_name} split.")
            continue

        dst_split_path = os.path.join(dst_dir, split_name)
        os.mkdir(dst_split_path)
        dst_annots = deepcopy(chd_annots)
        dst_annots["images"] = []
        dst_annots["annotations"] = []

        prt_annots_path = os.path.join(prt_split_path, "_annotations.coco.json")
        with open(prt_annots_path,"r") as fd:
            prt_annots = json.load(fd)
        prt_fns = {im["file_name"] for im in prt_annots["images"]}

        for img_fn in chd_fn2ID.keys():

            # If the child dataset image is represented in the parent split add it to the destination
            # split
            if img_fn in prt_fns:
                chd_file_path = os.path.join(chd_train,img_fn)
                dst_file_path = os.path.join(dst_split_path,img_fn)
                shutil.copyfile(chd_file_path,dst_file_path)

                # Add the image and its annotations to the destination split annotations file
                chd_img_id = chd_fn2ID[img_fn]
                img_info = chd_dataset.coco.loadImgs(chd_img_id)[0]
                dst_annots["images"].append(deepcopy(img_info))
                img_annots = chd_dataset.coco.imgToAnns[chd_img_id]
                for annot in img_annots:
                    dst_annots["annotations"].append(deepcopy(annot))

        # Save the destination split annotations file
        dst_split_annots_path = os.path.join(dst_split_path,"_annotations.coco.json")
        with open(dst_split_annots_path,"w") as fd:
            json.dump(dst_annots,fd)
        


