"""
Loops through videos in a specified directory and splits each video into frame at
the specified frame rate, resizes these to the specified dimensions and saves these
to the specified directory.
"""

import sys
import os
import cv2

# This routine returns a list of frames. If destdir is not None it also 
# writes the frames to the destdir as jpg files.
def split_video(vidpath, destdir, desired_fps, width,height):
    [viddir,vidname] = os.path.split(vidpath)

    vcdata = None
    try:
        vcdata = cv2.VideoCapture(vidpath)
    except:
        print(f"WARNING: cv2.VideoCapture failed on {vidpath}")
        return None
    
    num_frames = vcdata.get(cv2.CAP_PROP_FRAME_COUNT)
    actual_fps = vcdata.get(cv2.CAP_PROP_FPS)

    # calculate duration of the video
    try:
        seconds = num_frames / actual_fps
    except ZeroDivisionError:
        print(f"ERROR: Division by zero: {vidpath} has FPS 0 and {num_frames} frames")
        return None


    desired_frames = int(seconds * desired_fps)
    delta = num_frames/desired_frames

    (W, H) = (None, None)

    # Select frames at the desired frame rate
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

    # Write the frames to the destination dir

    if not os.path.exists(destdir):
        os.mkdir(destdir)
    if destdir is not None:
        [root_fn,_] = os.path.splitext(vidname)
        for count, frame in enumerate(frame_array):
            destfn = root_fn + "_" + str(count) + ".jpg"
            destpath = os.path.join(destdir, destfn)
            if frame is None:
                continue
            try:
                cv2.imwrite(destpath, frame)
            except:
                print(f"ERROR: Exception in cv2.write for frame size {frame.shape} for file {destpath}")

    return frame_array


if len(sys.argv) < 6:
    print(f"Call is: {sys.argv[0]} <videos dir> <dest dir> <FPS> <width> <height>")
    sys.exit(1)

viddir = sys.argv[1]
if not os.path.exists(viddir):
    print(f"ERROR: {viddir} does not exist")
    sys.exit(2)

destdir = sys.argv[2]
desired_fps = int(sys.argv[3])
width = int(sys.argv[4])
height = int(sys.argv[5])

count = 0
for fn in os.listdir(viddir):
    if not os.path.isfile(fn):
        vidpath = os.path.join(viddir, fn)
        split_video(vidpath, destdir, desired_fps, width,height)
        count += 1
    else:
        print(f"{fn} is not a file, skipping...")
    if (count % 10) == 0:
        print(count)

print("Done.")


