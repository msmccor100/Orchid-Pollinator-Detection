"""
Randomly splits files in a directory into two directories, the first of which 
contains a specified number of files, with the other containing the remaining
files. The directories are named 
    <dirpath>_1_<count>
    <dirpath>_2_<total-count>
"""

import sys
import os
import numpy as np
import shutil

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(f"Call is: python {sys.argv[0]} <dir> count")
        sys.exit(0)

    src_dir = sys.argv[1]
    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(1)

    count = int(sys.argv[2])
    total = 0
    fns = []
    for fn in os.listdir(src_dir):
        if os.path.isfile(os.path.join(src_dir, fn)):
            fns.append(fn)
            total += 1

    if total < count:
        print(f"ERROR: {src_dir} has fewer than {count} files")
        sys.exit(2)

    if src_dir[-1] == "/" or src_dir[-1] == "\\":
        src_dir = src_dir[:-1]

    dir1 = f"{src_dir}_1_{count}"
    dir2 = f"{src_dir}_2_{total - count}"

    if not os.path.exists(dir1):
        os.mkdir(dir1)
    if not os.path.exists(dir2):
        os.mkdir(dir2)

    # Choose file indices
    indices1 = np.random.choice(range(len(fns)), size=count, replace=False)
    indices2 = list( set( range(len(fns)) ).difference(indices1) )

    for idx in indices1:
        src_file = os.path.join(src_dir,fns[idx])
        dst_file = os.path.join(dir1,fns[idx])
        shutil.copyfile(src_file,dst_file)

    for idx in indices2:
        src_file = os.path.join(src_dir,fns[idx])
        dst_file = os.path.join(dir2,fns[idx])
        shutil.copyfile(src_file,dst_file)

    print("Done.")

    