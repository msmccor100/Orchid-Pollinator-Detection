"""
The point of this script is to compute classification accuracy score for yolo detections
ignoring bounding boxes.

call is: yolo_compute_accuracy.py <predicted labels dir> <true labels dir> <ignore labels string>

<predicted labels dir> is a directory consisting of yolo format predicted annotations. That is,
this contains a text file like 
    motion_2021-05-31_15_08_59_201_mp4-44_jpg.rf.b6a725ee387c785cc3d34757476d02d1.txt
containing predictions like

3 0.482812 0.338281 0.015625 0.0234375
5 0.351562 0.499219 0.315625 0.998438
5 0.585938 0.598437 0.290625 0.79375

<true labels dir> is the similarly formated true labels directory

<ignore labels string> is a quoted string contain a list of labels to ignore, like "2,5"

"""

import sys
import os

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

if len(sys.argv) != 4:
    print(f"Call is: {sys.argv[0]} <predicted labels dir> <true labels dir> <ignore labels string>")
    sys.exit(0)

predicted_dir = sys.argv[1]
if not os.path.exists(predicted_dir):
    print(f"ERROR: {predicted_dir} does not exist")
    sys.exit(1)

true_dir = sys.argv[2]
if not os.path.exists(true_dir):
    print(f"ERROR: {true_dir} does not exist")
    sys.exit(2)

ignore_labels = set()
if len(sys.argv[3]) > 0:
    ignore_labels = set([int(x) for x in sys.argv[3].split(',')])

total_count = 0
accuracy_count = 0
for fn in os.listdir(predicted_dir):
    predicted_fp = os.path.join(predicted_dir, fn)

    predicted_labels = get_label_set(predicted_fp)
    predicted_labels = predicted_labels.difference(ignore_labels)

    true_fp = os.path.join(true_dir, fn)
    if not os.path.exists(true_fp):
        print(f"WARNING: {true_fp} does not exist")

    true_labels = get_label_set(predicted_fp)
    true_labels = true_labels.difference(ignore_labels)

    print(f"predicted {predicted_labels}, true {true_labels}")

    accuracy_count += (predicted_labels == true_labels)
    total_count += 1

print(f"Accuracy {accuracy_count/total_count}")