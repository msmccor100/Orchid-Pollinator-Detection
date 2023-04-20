"""
This script splits a video into frames at the specified frame rate, resizes these
to specified dimensions, and saves these to the specified directory.
"""

import sys
import os
import cv2

if len(sys.argv) < 6:
    print(f"Call is: {sys.argv[0]} <video path> <dest dir> <FPS> <width> <height>")
    sys.exit(1)

vidpath = sys.argv[1]
if not os.path.exists(vidpath):
    print(f"ERROR: {vidpath} does not exist")
    sys.exit(2)
[viddir,vidname] = os.path.split(vidpath)

destdir = sys.argv[2]
desired_fps = int(sys.argv[3])
width = int(sys.argv[4])
height = int(sys.argv[5])

vcdata = cv2.VideoCapture(vidpath)
num_frames = vcdata.get(cv2.CAP_PROP_FRAME_COUNT)
actual_fps = vcdata.get(cv2.CAP_PROP_FPS)
  
# calculate duration of the video
seconds = num_frames / actual_fps
desired_frames = int(seconds * desired_fps)
delta = num_frames/desired_frames

(W, H) = (None, None)

# Read in all the frames and store in an array
frame_num = 0
frame_array = [None]*desired_frames
next_delta = 0
count = 0
while True:

    # read the next frame from the file
    (grabbed, frame) = vcdata.read()
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    if frame_num >= next_delta:
        next_delta += delta
        if H != height or W != width:
            frame_array[count] = cv2.resize(frame,dsize=(width,height),interpolation=cv2.INTER_AREA)
        else:
            frame_array[count] = frame
        count += 1
    frame_num += 1
    
if not os.path.exists(destdir):
    os.mkdir(destdir)

[root_fn,_] = os.path.splitext(vidname)
for count, frame in enumerate(frame_array):
    destfn = root_fn + "_" + str(count) + ".jpg"
    destpath = os.path.join(destdir, destfn)

    cv2.imwrite(destpath, frame)


