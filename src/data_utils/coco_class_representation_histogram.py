"""
Given a path to a coco annotations json file, this routine produces a histogram of
class occurence counts.

call is:  coco_class_representation_histogram.py <annotations path> <output path> <class names>

<class names> is a comma delimited quoted string of class names, e.g. "Bees & Wasps, Bumblebees"
"""

import os
import sys
import json
from matplotlib import pyplot as plt
import numpy as np




if len(sys.argv) != 4:
    print(f"Call is: {sys.argv} <annotations path> <output path> <class names>")
    sys.exit(1)

with open(sys.argv[1], "r") as fd:
    annots = json.load(fd)

# There's a useless "supercategory to which all the real categories belong"
num_classes = len(annots["categories"]) - 1
class_counts = [0]*num_classes


for annot in annots["annotations"]:
    curr_cat = annot["category_id"] - 1
    class_counts[curr_cat] += 1



class_counts = np.array(class_counts)
class_names = sys.argv[3].split(',')

fig = plt.figure(figsize = (7, 5))
plt.bar(class_names, class_counts, color ='blue',
        width = 0.4)
plt.xlabel("Class Representation")
plt.savefig(sys.argv[2])
plt.close()


