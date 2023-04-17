import sys
import os
import cv2

"""
Call is 
crop_images_to.py <dir> <width> <height>

This script creates a new directory called <dir>_cropped
For each image in <dir> this script center crops it to the
specified dimensions. 

Only supports .jpg or .png images
"""

if len(sys.argv) != 4:
    print("Call is sys.argv[0] <dir> <width> <height>")
    sys.exit(1)

src_dir = sys.argv[1]
if not os.path.exists(src_dir):
    print(f"ERROR: {src_dir} does not exist")
    sys.exit(2)
if not os.path.isdir(src_dir):
    print(f"ERROR: {src_dir} is not a directory")
    sys.exit(3)

dest_dir = src_dir.rstrip("\\/") + "_crop"
if os.path.exists(dest_dir):
    print(f"ERROR: {dest_dir} already exists")
    sys.exit(4)

width = int(sys.argv[2])
height = int(sys.argv[3])

os.mkdir(dest_dir)

count = 0
for fn in os.listdir(src_dir):
    if (count+1) % 10 == 0:
        print(count+1)
    count += 1

    fp = os.path.join(src_dir, fn)
    if not os.path.isfile(fp):
        continue

    ext = os.path.splitext(fn)[-1]
    if ext not in [".jpg",".png"]:
        continue

    img = cv2.imread(fp)
    h,w,c = img.shape

    if h < height:
        print(f"ERROR: Height {h} to small to crop to {height} for {fn}")
        continue
    if w < width:
        print(f"ERROR: Widtrh {w} to small to crop to {width} for {fn}")
        continue

    y_offset = (h - height)//2
    x_offset = (w-width)//2
    cropped = img[y_offset:y_offset + height,x_offset:x_offset + width ,:]

    new_fp = os.path.join(dest_dir,fn)
    cv2.imwrite(new_fp,cropped)






