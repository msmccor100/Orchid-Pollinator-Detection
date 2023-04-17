"""This script takes parameters <dirpath> and percent (int) where the 
directory is assumed to contain image files and an _annotations.coco.json 
file. This script splits dataset into two datasets containing <percent> of 
the images and another contianing the remainder. Each of these new datasets
will have their own _annotations.coco.json file. The two datasets are
placed in directories called 
    <dirname>_1_<percent> 
    <dirname>_2_<100 - percent>"""

import os
import sys
import shutil
import json
from copy import copy, deepcopy
import cv2
import supervision as sv
import torchvision
import random
import numpy as np

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(f"Call is: python {sys.argv[0]} <dataset dir> percent1")
        sys.exit(0)

    src_dir = sys.argv[1]
    percent_str = sys.argv[2]
    percent = int(sys.argv[2])

    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(1)
    # Verify src_dir has _annotations.coco.json
    annots_path = os.path.join(src_dir, "_annotations.coco.json")
    if not os.path.exists(annots_path):
        print(f"ERROR: No annotations file f{annots_path}")
        sys.exit(3)
    
    # Load annotations
    with open(annots_path, "r") as fd:
        annots = json.load(fd)

    # Prepare to create annotations for the two datasets
    annots1 = deepcopy(annots)
    annots1["images"] = []
    annots1["annotations"] = []

    annots2 = deepcopy(annots)
    annots2["images"] = []
    annots2["annotations"] = []

    # Create directories
    dir1_path = src_dir + "_1_" + percent_str
    dir2_path = src_dir + "_2_" + str(100-percent)
    if not os.path.exists(dir1_path):
        os.mkdir(dir1_path)
    if not os.path.exists(dir2_path):
        os.mkdir(dir2_path)


    # Get image and annotations ids for the dataset
    print(src_dir,annots_path)

    dataset = torchvision.datasets.CocoDetection(src_dir, annots_path)

    img_ids = dataset.coco.getImgIds()
    ann_ids = dataset.coco.getAnnIds()

    # Determine sizes and image_ids for new datasets
    size1 = int(len(img_ids)*percent/100)
    size2 = len(img_ids) - size1


    img_ids1 = np.random.choice(img_ids,size=size1,replace=False).tolist()
    img_ids2 = list( set(img_ids).difference(set(img_ids1)) )


    # Build the first dataset and annotations
    for id in img_ids1:
        img_info = dataset.coco.loadImgs(id)[0]
        img_annots = dataset.coco.imgToAnns[id]
        image_path = os.path.join(dataset.root, img_info['file_name'])

        new_image_path = os.path.join( dir1_path, img_info['file_name'] )
        shutil.copyfile(image_path, new_image_path)

        annots1["images"].append(img_info)

        img_annots = dataset.coco.imgToAnns[id]
        for ia in img_annots:
            annots1["annotations"].append(ia)

    annots1_path = os.path.join(dir1_path, "_annotations.coco.json")
    with open(annots1_path, "w") as fd:
        json.dump(annots1, fd)

    # Build the second dataset and annotations
    for id in img_ids2:
        img_info = dataset.coco.loadImgs(id)[0]
        img_annots = dataset.coco.imgToAnns[id]
        image_path = os.path.join(dataset.root, img_info['file_name'])

        new_image_path = os.path.join( dir2_path, img_info['file_name'] )
        shutil.copyfile(image_path, new_image_path)

        annots2["images"].append(img_info)

        img_annots = dataset.coco.imgToAnns[id]
        for ia in img_annots:
            annots2["annotations"].append(ia)

    annots2_path = os.path.join(dir2_path,"_annotations.coco.json")
    with open(annots2_path, "w") as fd:
        json.dump(annots2, fd)










