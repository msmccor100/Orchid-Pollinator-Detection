"""
This script takes a directory of yolo predicted labels for image frames taken from videos and classifies the
videos according to the pollinator classes they contain and whether they pollinate an inflorescence. It 
assumes yolo formated label files have names of the following form:

    <video base name>_<count>.txt

This name expectation is consistent with using split_videos.py to split videos into frames and then
using YOLOv7's detect.py to produce labels. This script outputs a csv file with lines like:

    <video base name>,(<class number 1>,yes),(<class number 2>,no)

or
    <video base name>,no pollinator

In the first example a pollinator of class number 1 is found and it does visit an inflorescence (yes). 
A pollinator of class number 2 is also found, but it does not visit an inflorescence (no). In the second
example, no pollinators were found in the video.

This script takes a motion threshold, a list of class numbers to which it applies, and an overlap 
threshold which specifies how much bounding box overlap there must be to count a pollinator as visiting 
an inflorescence.

The motion threshold specifies how big the standard deviation of the distances of the pollinator bounding
boxes from the median position must be in order to count the pollinator detection as real: we expect
most of our pollinator classes to move around a lot.

Call is: yolo_classify_videos.py <labels dir> <outfile path> <motion threshold> <classes> <overlap threshold>

<motion threshold> is between 0 and 1 and represents a threshold for the average of the distances of the 
pollinator bounding boxes from the median of these, treating the image as a unit square. If the pollinator 
strays a lot from their median position, they'll pass the threshold.

<classes> is a quoted, comma delimited string, like "0,1,2,4,6,7". this specifies what classes we want
the motion threshold to apply to.

<overlap> is between 0 and 1 represents the relative portion of the pollinator's bounding box that must overlap 
with an inflorescence in order to count that as pollinating, i.e. it is (area of overlap)/(area pollinator BB).
"""

import os
import sys
from collections import defaultdict
from pybboxes import BoundingBox
import numpy as np

INFLORESCENCE_CLASS = 5

def get_video_base_names(labels_dir):
    """Returns a sorted list of all the video base names associated with the label files
    in the specified directory. It assumes yolo formated label files have names of the 
    following form:
        <video base name>_<count>.txt
    """
    base_names_set = set()
    for fn in os.listdir(labels_dir):
        if fn[-4:] != ".txt":
            continue
        if '_' not in fn:
            continue
        vidbase = fn[:fn.rfind('_')]
        base_names_set.add(vidbase)
    base_names = list(base_names_set)
    base_names.sort()
    return base_names

def get_label_filenames_for_video(labels_dir, vid_base_name):
    filenames = []
    for fn in os.listdir(labels_dir):
        if fn.find(vid_base_name) == 0:
            filenames.append(fn)
    filenames.sort()
    return filenames

def avg_distance(pt_list,pt):

    avg_dist = 0.0
    for pt_i in pt_list:
        avg_dist += np.sqrt((pt_i[0] - pt[0])**2 + (pt_i[1] - pt[1])**2)
    avg_dist /= len(pt_list)
    return avg_dist

def get_max_distance(pt_list,pt):
    dists = []
    for pt_i in pt_list:
        dists.append(np.sqrt((pt_i[0] - pt[0])**2 + (pt_i[1] - pt[1])**2))
    return max(dists)

def get_motion_deviation(bb_list):
    """
    Computes the means distance of the bounding boxe centers from the
    median of the centers.

    bb_list is expect to be a list of yolo formatted bounding boxes.
    [ctr_x, ctr_y, width_ratio, height ration]
    """
    if len(bb_list) == 0:
        return None
    if len(bb_list) == 1:
        return 0.0
    med_x = np.median([bb[0] for bb in bb_list])
    med_y = np.median([bb[1] for bb in bb_list])
    return avg_distance(bb_list, (med_x,med_y))

def visits_inflorescence(pollinator_bbs, inflorescence_bbs, overlap_threshold):
    """
    This routine determines whether a pollinator visits an inflorescence by detecting
    whether a polinator BB has sufficient overlap with that of some inflorescence
    expressed as a ratio the area overlap region as compared to the area of the 
    pollinator's bounding box.
    """

    if len(pollinator_bbs) == 0 or len(inflorescence_bbs) == 0:
        return True

    for pbb in pollinator_bbs:

        # Since we're computing a relative area, the image size doesn't matter
        pbb_obj = BoundingBox.from_yolo(*pbb, image_size=(640,640))
        for ibb in inflorescence_bbs:
            ibb_objs = BoundingBox.from_yolo(*ibb, image_size=(640,640))

            intersection_area = pbb_obj * ibb_objs
            relative_overlap_area = intersection_area/pbb_obj.area

            if relative_overlap_area >= overlap_threshold:
                return True
            
    return False

            
if __name__ == "__main__":

    if len(sys.argv) != 6:
        print(f"Call is: {sys.argv[0]} <labels dir> <outfile path> <motion threshold> <classes> <overlap threshold>")
        sys.exit(1)

    [_,labels_dir,outpath,motion_thresh,motion_classes,overlap_thresh] = sys.argv
    motion_thresh = float(motion_thresh)
    motion_classes = [int(c) for c in motion_classes.split(',')]
    overlap_thresh = float(overlap_thresh)

    if not os.path.exists(labels_dir):
        print(f"ERROR: {labels_dir} does not exit")
        sys.exit(2)

    vid_base_names = get_video_base_names(labels_dir)

    csv = ""

    for vbn in vid_base_names:
        csv_line = vbn

        # Get all the label file names for this video
        fns = get_label_filenames_for_video(labels_dir, vbn)

        # Make a dictionary keyed by class number with value giving a list of
        # all of the bound boxes associated with that class that where present
        # in some frame of the video

        bbs_dict = defaultdict(lambda : list())

        for fn in fns:

            fp = os.path.join(labels_dir, fn)
            with open(fp, "r") as fd:
                for line in fd:
                    line = line.strip()
                    line = line.rstrip()
                    if len(line) < 6:
                        continue
                    label_info = line.split(' ')
                    if len(label_info) != 5:
                        continue
                    cls = int(label_info[0])
                    bb = [float(x) for x in label_info[1:]]
                    bbs_dict[cls].append(bb)

        # Determine whether we have pollinators and if they do pollinate

        pollinator_found = False
        inflorescence_bbs = bbs_dict[INFLORESCENCE_CLASS]
        for cls in bbs_dict.keys():

            if cls == INFLORESCENCE_CLASS:
                continue

            pbb_list = bbs_dict[cls]
            if len(pbb_list) == 0:
                continue

            if cls in motion_classes:

                # If the pollinator motion doesn't meet the specified threshold,
                # consider it a false hit and ignore it.

                motion_dev = get_motion_deviation(pbb_list)
                print("\tmotion deviation", motion_dev)
                if motion_dev < motion_thresh:
                    continue

            visits = "yes" if visits_inflorescence(pbb_list, inflorescence_bbs, overlap_thresh) else "no"
            csv_line += f",({cls},{visits})"
            pollinator_found = True

        if not pollinator_found:
            csv_line += ",no pollinator"
        print(csv_line)

        csv_line += "\n"
        csv += csv_line

    with open(outpath,"w") as fd:
        fd.write(csv)

            





            

            

        






    

