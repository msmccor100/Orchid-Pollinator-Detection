"""
Given a the path to a directory containing a COCO formated dataset (i.e. containing images and 
an _annotations.coco.json file), this routine creates a new directory with the suffix _bb 
containing images with bounding boxes drawn in.
"""

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
    if len(sys.argv) < 2:
        print(f"Call is: python {sys.argv[0]} <dataset dir>")
        sys.exit(0)

    src_dir = sys.argv[1]
    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(1)
    if not os.path.isdir(src_dir):
        print(f"ERROR: Specified path {src_dir} is not a directory")
        sys.exit(2)

    # Verify src_dir has _annotations.coco.json
    annots_path = os.path.join(src_dir, "_annotations.coco.json")
    if not os.path.exists(annots_path):
        print(f"ERROR: No annotations file f{annots_path}")
        sys.exit(3)

    dest_dir = src_dir.rstrip("/\\") + "_bb"
    os.mkdir(dest_dir)

    annots_path = os.path.join(src_dir, "_annotations.coco.json")
    annots = None
    with open(annots_path,"r") as fd:
        annots = json.load(fd)
    if annots is None:
        print(f"ERROR: Reading {annots_path}")


    dataset = torchvision.datasets.CocoDetection(src_dir, annots_path)

    categories =dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}

    img_ids = dataset.coco.getImgIds()
    ann_ids = dataset.coco.getAnnIds()

    print(len(img_ids))

    count = 0
    for img_id in img_ids:

        if (count+1)%10 == 0:
            print(count+1)
        count += 1

        # Read in image and obtain its annotations

        img_info = dataset.coco.loadImgs(img_id)[0]
        img_annots = dataset.coco.imgToAnns[img_id]
        image_path = os.path.join(dataset.root, img_info['file_name'])
        img = cv2.imread(image_path)

        # Draw bounding boxes and ave to new directory.

        try:
            detections = sv.Detections.from_coco_annotations(coco_annotation=img_annots)
            box_annotator = sv.BoxAnnotator()
            labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
            frame = box_annotator.annotate(scene=img, detections=detections, labels=labels)

            file_name = os.path.split(img_info['file_name'])[-1]
            new_file_path = os.path.join(dest_dir,file_name)
            if not cv2.imwrite(new_file_path, frame):
                print("cv2.imwrite failed to write ", file_name)

        except Exception as exc:
            import traceback
            traceback.print_tb(exc.__traceback__, limit=1, file=sys.stdout)
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
            print(''.join(tb.format_exception_only()))

