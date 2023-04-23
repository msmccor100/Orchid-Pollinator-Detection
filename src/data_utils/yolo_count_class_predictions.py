"""
Given a directory of yolo format predicted labels (with a separate file for each image)
this sprint simply counts how often each class appears.

call is: yolo_compute_accuracy.py <predicted labels dir> 
"""

import sys
import os
from collections import defaultdict

def get_label_set(labels_fp):
    labels_set = set()
    with open(labels_fp,"r") as fd:
        for line in fd:
            line = line.strip()
            if len(line) < 6:
                continue
            try:
                label = int(line.split(' ')[0])
            except:
                continue
            labels_set.add(label)
    return labels_set

if len(sys.argv) != 2:
    print(f"Call is: {sys.argv[0]} <predicted labels dir>")
    sys.exit(0)

predicted_dir = sys.argv[1]
if not os.path.exists(predicted_dir):
    print(f"ERROR: {predicted_dir} does not exist")
    sys.exit(1)

class_count = defaultdict(lambda : 0)
image_count = 0
label_count = 0
for fn in os.listdir(predicted_dir):
    predicted_fp = os.path.join(predicted_dir, fn)
    if ".txt" not in predicted_fp:
        continue
    image_count += 1
    with open(predicted_fp,"r") as fd:
        for line in fd:
            line = line.strip()
            if len(line) < 6:
                continue
            try:
                label = int(line.split(' ')[0])
            except:
                continue
            label_count += 1
            class_count[label] += 1

print(f"Counted {label_count} labels for {image_count} images:\n")
labels = list(class_count.keys())
labels.sort()

print("CLASS     COUNT")
for label in labels:
    print(f"{label:<6}    {class_count[label]:<10}")



  