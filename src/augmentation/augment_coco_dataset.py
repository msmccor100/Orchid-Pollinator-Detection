"""Given an object detection dataset with annotations in COCO format, this program
creates a new dataset with new annotations, combining the original dataset with
augmented images, where a list of possible augmentations is applied according to
configured probabilities."""

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
import albumentations as A


DEBUG = False


# NUM_AUGS_PROBS = [0.0, 0.3, 0.5, 0.2] says that when making an augmented image we have a 
# a 30% chance of using on only one augmentation, a 50% chance of using two, etc.
NUM_AUGS_PROBS = [0.0, 0.3, 0.7]

augment_choices = [
    A.GaussNoise(var_limit=(10.0, 40.0), mean=0, per_channel=True, always_apply=True),
    A.HorizontalFlip(always_apply=True),
    A.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True),
    A.RandomBrightnessContrast (brightness_limit=0.15, contrast_limit=0.15, brightness_by_max=True, always_apply=True)

]

augment_names = [
    "GaussNoise",
    "HorizontalFlip",
    "HueSaturationValue",
    "RandomBrightnessContrast"
]


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Call is: python {sys.argv[0]} <dataset dir> <output dirpath (must not exist)> <max num images to add>")
        sys.exit(0)

    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    num_to_add = int(sys.argv[3])

    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(1)

    if os.path.exists(dest_dir):
        print(f"ERROR: {dest_dir} already exists")
        sys.exit(2)


    # Verify src_dir has _annotations.coco.json
    annots_path = os.path.join(src_dir, "_annotations.coco.json")
    if not os.path.exists(annots_path):
        print(f"ERROR: No annotations file f{annots_path}")
        sys.exit(3)

    # Copy src_dir to dest_dir
    shutil.copytree(src_dir, dest_dir)

    annots_path = os.path.join(dest_dir, "_annotations.coco.json")
    annots = None
    with open(annots_path,"r") as fd:
        annots = json.load(fd)
    if annots is None:
        print(f"ERROR: Reading {annots_path}")


    dataset = torchvision.datasets.CocoDetection(dest_dir, annots_path)

    categories =dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    img_ids = dataset.coco.getImgIds()
    ann_ids = dataset.coco.getAnnIds()
    new_img_ids = copy(img_ids)
    new_ann_ids = copy(ann_ids)

    next_img_id = max(img_ids) + 1
    next_ann_id = max(ann_ids) + 1

    for num in range(num_to_add):

        if ((num+1) %20) == 0:
            print(num+1)

        # Select which image to augment
        id = random.choice(img_ids)

        # Read in image and obtain its annotations

        img_info = dataset.coco.loadImgs(id)[0]
        img_annots = dataset.coco.imgToAnns[id]
        image_path = os.path.join(dataset.root, img_info['file_name'])
        img = cv2.imread(image_path)

        # Select how many augments to apply
        num_augs = np.random.choice(range(len(NUM_AUGS_PROBS)), p=NUM_AUGS_PROBS)

        # Select the augmentations to apply
        aug_ids = np.random.choice(range(len(augment_choices)), size=num_augs, replace=False).tolist()
        
        augments_string = ""
        for a in aug_ids:
            augments_string += augment_names[a] + " "

        if DEBUG:
            print(augments_string)

        augs = [augment_choices[i] for i in aug_ids ]
        transform = A.Compose(augs, bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

        # albumentations is buggy
        old_bboxes = [a['bbox'] for a in img_annots]
        try:
            transformed = transform(image=img, bboxes=old_bboxes, labels=[a['category_id'] for a in img_annots])
        except Exception as exc:
            continue
        new_img = transformed['image']
        bboxes = transformed['bboxes']

        # Sometimes albumentation mysteriously throws out bounding boxes.
        if len(old_bboxes) != len(bboxes):
               continue

        # Save augmented image in dest_dir with filename suffix indicating which augments were applied
        suffix_num = 0
        for aug_id in aug_ids:
            suffix_num += (1 << aug_id)
        suffix = "_" + hex(suffix_num)

        base,ext= os.path.splitext(img_info['file_name'])
        new_img_filename = base + suffix + ext
        new_img_path = os.path.join(dest_dir, new_img_filename)
        cv2.imwrite(new_img_path, new_img)

        # Add new img info to annotations json
        
        new_img_info = deepcopy(img_info)
        new_img_info["id"] = next_img_id
        next_img_id += 1
        new_img_info["file_name"] = new_img_filename
        annots["images"].append(new_img_info)

        # Update the annotations
        new_img_annots = deepcopy(img_annots)
        for i, anno in enumerate(new_img_annots ):
            anno["id"] = next_ann_id
            anno["image_id"] = new_img_info["id"]
            next_img_id += 1
            anno["bbox"] = bboxes[i]
            anno["area"] = bboxes[i][2]*bboxes[i][3] # Coco uses format [x,y,w,h]
            annots["annotations"].append(anno)

        if DEBUG:

            try:    
                detections = sv.Detections.from_coco_annotations(coco_annotation=new_img_annots)
            except Exception as exc:
                import traceback
                traceback.print_tb(exc.__traceback__, limit=1, file=sys.stdout)
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
                print(''.join(tb.format_exception_only()))
            box_annotator = sv.BoxAnnotator()
            labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
            frame = box_annotator.annotate(scene=new_img, detections=detections, labels=labels)
            cv2.imshow("", frame)
            cv2.waitKey(0)


    # Save the new annotations
    with open(annots_path, "w") as fd:
        json.dump(annots, fd)






        









