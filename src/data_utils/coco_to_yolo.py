"""
This script take source and destination directories where the source represents a COCO annotated
data split across subdirectories such as test/, train/, val/, or valid/. This routine creates
the destination directory and formats the COCO dataset as in YOLO format
"""

import sys
import os
import json
import torchvision
import shutil
from pybboxes import BoundingBox
import cv2

if len(sys.argv) != 3:
    print(f"Call is: {sys.argv[0]} <coco dir> <dest dir>")
    sys.exit(1)

src_dir = sys.argv[1]
if not os.path.exists(src_dir):
    print(f"ERROR {src_dir} does not exist")
    sys.exit(2)

dest_dir = sys.argv[2]
if os.path.exists(dest_dir):
    print(f"ERROR: {dest_dir} already exists")
    sys.exit(3)

data_dirs = {}
if os.path.exists(os.path.join(src_dir, "test")):
    data_dirs["test"] = "test"
if os.path.exists(os.path.join(src_dir, "train")):
    data_dirs["train"] = "train"
if os.path.exists(os.path.join(src_dir, "val")):
    data_dirs["val"] = "val"
if os.path.exists(os.path.join(src_dir, "valid")):
    data_dirs["val"] = "valid"

if "train" not in data_dirs:
    print(f"ERROR: Source must have train subdir")
    sys.exit(4)

train_annots_path = os.path.join(src_dir,"train")
train_annots_path = os.path.join(train_annots_path, "_annotations.coco.json")
if not os.path.exists(train_annots_path):
    print(f"ERROR: source train dir has no _annotations.coco.json")
    sys.exit(5)

with open(train_annots_path,"r") as fd:
    train_annots = json.load(fd)

os.mkdir(dest_dir)

#
# Build the YOLO yaml file
#

yaml = "names:\n"
count = 0
for cat in train_annots["categories"]:
    # For some reason, Roboflow gave each label a super category of "Pollinators"
    if cat["name"] == "Pollinators":
        continue
    yaml += "- " + cat["name"] + "\n"
    count= count + 1
yaml += f"nc: {count}\n"
for data_dir in data_dirs.keys():
    yaml += f"{data_dir}: ./{data_dir}/images\n"

with open(os.path.join(dest_dir,"data.yaml"),"w") as fd:
    fd.write(yaml)

# Loop through the source dirs and copy over images and annotations
for data_dir_type in data_dirs:
    src_data_dir = os.path.join(src_dir, data_dirs[data_dir_type])
    src_annots_path = os.path.join(src_data_dir,"_annotations.coco.json")
    dest_images = os.path.join(dest_dir, data_dir_type)
    dest_images = os.path.join(dest_images,"images")
    dest_labels = os.path.join(dest_dir, data_dir_type)
    dest_labels = os.path.join(dest_labels, "labels")
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    src_dataset = torchvision.datasets.CocoDetection(src_data_dir, src_annots_path)
    img_ids = src_dataset.coco.getImgIds()
    
    for img_id in img_ids:
        img_info = src_dataset.coco.loadImgs(img_id)[0]
        img_annots = src_dataset.coco.imgToAnns[img_id]
        src_image_path = os.path.join(src_dataset.root, img_info['file_name'])

        dest_img_path = os.path.join(dest_images, img_info['file_name'])
        shutil.copyfile(src_image_path, dest_img_path)

        # Make the labels file
        labels = ""
        for annot in img_annots:
            img = cv2.imread(src_image_path)
            image_size = img.shape[:2]

            try:
                coco_bbox = BoundingBox.from_coco(*annot["bbox"],image_size=image_size)
            except:
                continue
            yolo_bbox = coco_bbox.to_yolo()
            x1,y1,x2,y2 = yolo_bbox.values

            # Subtracting 1 here because the yolo format indexes from 0 and
            # the 0th category in the coco format is the "supercategory"
            label = annot["category_id"] - 1
            labels += f"{label} {x1} {y1} {x2} {y2}\n"

        dest_labels_file_path = os.path.join(dest_labels, img_info['file_name'] + ".txt")
        with open(dest_labels_file_path,"w") as fd:
            fd.write(labels)





    

