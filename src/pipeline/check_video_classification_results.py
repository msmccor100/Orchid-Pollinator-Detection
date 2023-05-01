""""
This script computes metrics on video label predictions made by classify_vides_by_labels.py
against a csv file of true labels. This script ignores whether or not the pollinator is
pollinating. classify_vides_by_labels.py outputs with rows like

motion_2021-06-17_22.40.28_173,Butterflies-Moths,pollinating
motion_2021-07-03_22.04.43_35,Fly,pollinating
motion_2021-07-05_14.25.22_128,Fly,not pollinating
motion_2021-07-05_15.11.30_156,Fly,pollinating
motion_2021-07-05_22.19.33_267,Fly,unknown
motion_2021-07-06_02.55.25_126,no pollinator
motion_2021-07-06_03.33.01_132,no pollinator
motion_2021-07-06_04.59.22_186,no pollinator

The true labels csv is assumed to look like. Note that the video nmames have file extension

motion_2021-07-05_14.25.22_128.mp4,fly
motion_2021-07-05_15.11.30_156.mp4,fly
motion_2021-07-05_17.10.10_205.mp4,fly
motion_2021-07-06_04.03.54_166.mp4,fly
motion_2021-07-07_13.24.36_14.mp4,fly
motion_2021-07-11_07.01.17_233.mp4,bee
motion_2021-07-11_08.06.37_326.mp4,bee
motion_2021-07-11_08.06.58_327.mp4,bee

"""

import os
import sys
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# The true labels file uses some shorted names
classname_to_shortname = {
    "Bees-Wasps":"bee",
    "Bumble Bee":"bumblebee",
    "Butterflies-Moths":"butterfly",
    "Fly":"fly",
    "Hummingbird":"hummingbird",
    "no pollinator":"no pollinator",
    "Other":"other"
}

shortname_to_classname = {
    "bee":"Bees-Wasps",
    "bumblebee":"Bumble Bee",
    "butterfly":"Butterflies-Moths",
    "fly":"Fly",
    "hummingbird":"Hummingbird",
    "no pollinator":"no pollinator",
    "other":"Other"
}

shortname_to_classnum = {
    "bee":0,
    "bumblebee":1,
    "butterfly":2,
    "fly":3,
    "hummingbird":4,
    "no pollinator":5,
    "other":6
}

def read_true_labels(true_labels_fp):
    true_labels_dict = {}
    with open(true_labels_fp,"r") as fd:
        for line in fd:
            line = line.strip()
            line = line.rstrip()
            split = line.split(',')
            if len(split) != 2:
                continue
            vidname, cls = split
            basename = os.path.splitext(vidname)[0]
            basename = basename.lower()
            cls = cls.lower()
            true_labels_dict[basename] = cls
    return true_labels_dict

def read_predicted_labels(predicted_labels_fp):
    predicted_labels_dict = {}
    with open(predicted_labels_fp,"r") as fd:
        for line in fd:
            line = line.strip()
            line = line.rstrip()
            split = line.split(',')
            if len(split) <2:
                continue
            vidname, cls = split[:2]
            vidname = vidname.lower()
            predicted_labels_dict[vidname] = classname_to_shortname[cls]

    return predicted_labels_dict


def verify_paths(paths_list, exit_code):
    for path in paths_list:
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist")
            sys.exit(exit_code)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Call is: {sys.argv[0]} <true labels csv> <predicted labels csv> <outdir>")
        sys.exit(-1)

    verify_paths(sys.argv[1:], -2)
    [true_labels_fp, predicted_labels_fp,outdir] = sys.argv[1:]

    true_labels_dict = read_true_labels(true_labels_fp)
    predicted_labels_dict = read_predicted_labels(predicted_labels_fp)

    confusion = np.zeros(shape=(7,7))
    for vidname,predicted in predicted_labels_dict.items():
        true = true_labels_dict[vidname]

        predicted_id = shortname_to_classnum[predicted]
        true_id = shortname_to_classnum[true]

        confusion[predicted_id,true_id] += 1

        labels = ["bee","bbee","bttrfly","fly","hmmbrd","none","other"]

        df_cm = pd.DataFrame(confusion, index =labels,
                  columns = labels)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
cm_path = os.path.join(outdir,"confusion.png")
plt.savefig(cm_path)


