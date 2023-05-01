"""
Given a directory for a yolo annotated dataset, the name of a (non_existent) destination 
directory. and a list of classes to remove, this copies the dataset to the destination
and then edits the labels to remove the classes. This version 2 differs from the
original version in that it does not renumber the labels. This will allow a model
trained with more classes to be subsequently refined with fewer.

For simplicity, we assume the yolo data.yaml begins with names and nc. For example,

    names:
    - Bees-Wasps
    - Bumble Bee
    - Butterflies-Moths
    - Fly
    - Hummingbird
    - Inflorescence
    - Other
    nc: 7

Call is: yolo_remove_classes.py <yolo dataset dir> <dest dir> <classes to remove>

<classes to remove> is a quoted, comma-delimited string like "2,4"

The caller is responsible for editing the paths in the destination data.yaml

"""
import os
import sys
import shutil

def main():
    if len(sys.argv) != 4:
        print(f"Call is: {sys.argv[0]} <yolo dataset dir> <dest dir> <classes to remove>")
        sys.exit(1)

    [dataset_dir, dest_dir, remove_classes] = sys.argv[1:]
    if not os.path.exists(dataset_dir):
        print(f"ERROR: {dataset_dir} does not exist")
        sys.exit(2)

    if os.path.exists(dest_dir):
        print(f"ERROR: {dest_dir} already exists")
        sys.exit(3)

    remove_classes = [int(x) for x in remove_classes.split(",")]

    shutil.copytree(dataset_dir, dest_dir)

    # Unlike version 1, we do not need to edit the yaml file

    # Now edit all of the label files
    subdirs = ["train","val", "test"]
    for subdir in subdirs:

        train_etc_dir = os.path.join(dest_dir, subdir)

        # Delete labels cache, if any, since we're changing labels
        labels_cache_fp = os.path.join(train_etc_dir,"labels.cache")
        if os.path.exists(labels_cache_fp):
            os.remove(labels_cache_fp)

        # Loop over the labels files, editing the labels
        labels_dir = os.path.join(train_etc_dir,"labels")
        for label_fn in os.listdir(labels_dir):
            if ".txt" not in label_fn:
                continue
            label_fp = os.path.join(labels_dir,label_fn)
            new_label_data = ""

            with open(label_fp, "r") as fd:

                for line in fd.readlines():
                    sline = line.strip()
                    sline = sline.rstrip()
                    if len(sline) < 6:
                        continue

                    sline_split = sline.split(' ')
                    label = int(sline_split[0])
                    if label in remove_classes:
                        continue
                    new_label_data += line

            with open(label_fp, "w") as fd:
                fd.write(new_label_data)


if __name__ == "__main__":
    main()



