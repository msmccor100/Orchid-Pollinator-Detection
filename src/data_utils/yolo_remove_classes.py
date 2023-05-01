"""
Given a directory for a yolo annotated dataset, the name of a (non_existent) destination 
directory. and a list of classes to remove, this copies the dataset to the destination
and then edits the labels to remove the classes, renumbering the remaining ones
appropriately.

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

    data_yaml_fp = os.path.join(dest_dir,"data.yaml")
    class_count = 0
    new_class_count = 0
    old_label_to_new = {}
    name_to_new_label = {}
    new_yaml = ""
    processing_names = True
    with open(data_yaml_fp, "r") as fd:

        for count, line in enumerate(fd):
            sline = line.strip()
            sline = sline.rstrip()
            if count == 0:
                if 0 != sline.find("names"):
                    print("ERROR: data.yaml does  not begin with names")
                    sys.exit(4)
                new_yaml += "names:\n"
                continue

            if processing_names:

                # Are we done processing names?
                if sline[0] != '-':
                    processing_names = False
                    if 0 != sline.find("nc"):
                        print("ERROR: Expected nc after names")
                        sys.exit(5)
                    new_yaml += f"nc: {new_class_count}\n"
                    continue

                # Still processing names
                if class_count in remove_classes:
                    class_count += 1
                    # skip it
                    continue
                else:
                    class_name = sline[1:].strip()
                    name_to_new_label[class_name] = new_class_count
                    old_label_to_new[class_count] = new_class_count
                    class_count += 1
                    new_class_count += 1
                    new_yaml += line
                    continue

            # No longer processing names but just the remaining body of the yaml
            # Just copy that as is

            new_yaml += line

    print("New class mapping:")
    for k,v in name_to_new_label.items():
        print(f"\t{k} -> {v}")

    # Replace the yaml file.
    with open(data_yaml_fp, "w") as fd:
        fd.write(new_yaml)

    # Now edit all of the damn label files
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
                    new_label = old_label_to_new[label]
                    new_line_split = [str(new_label)] + sline_split[1:]
                    new_line = " ".join(new_line_split)
                    new_line += "\n"
                    new_label_data += new_line

            with open(label_fp, "w") as fd:
                fd.write(new_label_data)


if __name__ == "__main__":
    main()



