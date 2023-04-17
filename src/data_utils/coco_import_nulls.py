"""
Given a directory of null images (images with no annotations) and a path to a coco dataset 
(a directory of images with an _annotations.coco.json file), this routine copies these null 
images to the coco dataset, editing the _annotations.coco.json appropriately. This script
does not match adjust the size of the imported images in any way. This script does not
deconflict file names.

Call is: python coco_import_nulls.py <nulls dir> <dataset dir>
"""

import os
import sys
import shutil
import json
from copy import copy, deepcopy
import torchvision
import cv2
import datetime

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f"Call is: python {sys.argv[0]} <nulls dir> <dataset dir>")
        sys.exit(0)

    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]

    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(1)

    if not os.path.exists(dest_dir):
        print(f"ERROR: {dest_dir} does not exist")
        sys.exit(2)

    # Verify src_dir has _annotations.coco.json
    annots_path = os.path.join(dest_dir, "_annotations.coco.json")
    if not os.path.exists(annots_path):
        print(f"ERROR: No annotations file f{annots_path}")
        sys.exit(3)
    
    # Load annotations
    with open(annots_path, "r") as fd:
        annots = json.load(fd)

    dest_dataset = torchvision.datasets.CocoDetection(src_dir, annots_path)

    img_ids = dest_dataset.coco.getImgIds()
    next_img_id = 1
    if len(img_ids) > 0:
        next_img_id = max(img_ids) + 1

    license = 1
    if len(annots["images"]) > 0 and "license" in annots["images"][0]:
        license = annots["images"][0]["license"]

    # Loop through the source files and add them to the dataset.
    for fn in os.listdir(src_dir):

        fp = os.path.join(src_dir, fn)
        if not os.path.isfile(fp):
            continue

        ext = os.path.splitext(fn)[-1]
        if ext not in [".jpg",".png"]:
            continue

        img = cv2.imread(fp)

        img_entry = {
            "id": next_img_id,
            "license":license,
            "file_name":fn,
            "height":img.shape[0],
            "width":img.shape[1],
            "date_captured": str(datetime.datetime.fromtimestamp(os.path.getctime(fp)))
        }
        next_img_id += 1

        new_fp = os.path.join(dest_dir, fn)
        shutil.copyfile(fp,new_fp)

        annots["images"].append(img_entry)

    with open(annots_path,"w") as fd:
        json.dump(annots, fd)

  




