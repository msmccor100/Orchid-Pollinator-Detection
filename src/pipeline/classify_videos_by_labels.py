"""
This script takes a path to Yolo labels files, a path to classifier1 labels files, 
a path to classifer2 labels files, an output file path, a motion threshold, 
a class list, and an overlap threshold. 

The label files names are assumed to be of the form
    <image file base>.txt
where the assocated image file base name has the form
    <video base name)>_<counter>
This is consistent with frames pulled from images usign split_videos.py.

The Yolo labels have 0 or more lines of the form
    <class num> bb1 bb2 bb3 bb4 <prob>
where the floats bb1,... are the yolo format bounding box values and <prob>
is the prediction probability. The class numbers are assigned as
    0 - Bees-Wasps
    1 - Bumble Bee
    2 - Butterflies-Moths
    3 - Fly
    4 - Hummingbird
    5 - Inflorescence
    6 - Other

The classifier label files have exactly one line of the form
    <class num> <prob>
where the class numbers are assigned as
    0 - Bees-Wasps
    1 - Bumble Bee
    2 - Butterflies-Moths
    3 - Fly
    4 - Hummingbird
    5 - Null image
    6 - Other
Thus, the classifiers are assumed not to classify inflorescences. Rather, the
class 5 Null images simply represent images predicted not to contain a pollinator.

Using this info we will determine for each video whether it has a pollinator and
whether it is pollinating. We assume upfront that there is only one class of pollinator.
If Yolo identifies multiple pollinators, we will throw out the ones of lower
probability.

This script identifies all of the label files associated with an originating video.
For each detector/classifier this identifies the maximum probability pollinator class 
(if any) and associated maximum probability across these label files. This is
taken to be THE pollinator prediction for the method. Across all three methods,
a probability weighted vote is taken to it determine the maximum probability pollinator.
This is assumed to the the real pollinator, even if Yolo concluded differently, but the
Yolo bounding box Yolo obtained for its maximal pollinator is still assumed to be correct.
The overlap threshold is what fraction of the bounding box must overlap with that of
some inflorescence in order to consider the pollinator to be pollinating.

These conclusions are written to the output file in csv format with line per video
of the form
    video base name, no pollinator
or
    video base name, <pollinator class name>, <yes/no>

Call is: yolo_classify_videos.py <labels dir> <yolo label> <clfr1 labels> <clfr2 labels> <outfile path> <motion threshold> <classes> <overlap threshold>")

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
from copy import copy

INFLORESCENCE_CLASS = 5

CLS2NAME = [
    "Bees-Wasps",
    "Bumble Bee",
    "Butterflies-Moths",
    "Fly",
    "Hummingbird",
    "Inflorescence",
    "Other",
]


# The classifies don't classify inflorescences, but rathe ruse 5 for null iamges (no pollinators)
NULL_IMAGE = 5

def verify_paths(paths_list, exit_code):
    for path in paths_list:
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist")
            sys.exit(exit_code)

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
        # Throw out the confidence if any
        pbb = pbb[:4]

        # Since we're computing a relative area, the image size doesn't matter
        pbb_obj = BoundingBox.from_yolo(*pbb, image_size=(640,640))
        for ibb in inflorescence_bbs:
            # Throw out the confidence, if any
            ibb = ibb[:4]
            ibb_objs = BoundingBox.from_yolo(*ibb, image_size=(640,640))

            # Conveniently, the BoundingBox class implements * as intersection area.
            intersection_area = pbb_obj * ibb_objs
            relative_overlap_area = intersection_area/pbb_obj.area

            if relative_overlap_area >= overlap_threshold:
                return True
            
    return False

def get_yolo_bbs_dict(yolo_labels_dir,fns):
    """
    Given a list of label filenames assumed to be associatied with the
    same video, this routine builds a dictionary of lists of the form
    [bb1,bb2,bb3,bb3,prob] indexed by classnums, where bb1...bb4 
    respresent the yolo bounding box.
    """

    bbs_dict = {}
    for i in range(7):
        bbs_dict[i] = []

    for fn in fns:
        fp = os.path.join(yolo_labels_dir, fn)
        with open(fp, "r") as fd:
            for line in fd:
                line = line.strip()
                line = line.rstrip()
                if len(line) < 6:
                    continue
                label_info = line.split(' ')
                if len(label_info) != 6:
                    continue
                cls = int(label_info[0])
                bb = [float(x) for x in label_info[1:]]
                bbs_dict[cls].append(bb)

    return bbs_dict

def get_yolo_prediction_for_video(bbs_dict):
    """
    Returns tuple (best pollinator class, prob pollinator, prob nul video)
    Note that best pollinator class can be null
    """

    # For pollinator class, determine the max prob of its bounding boxes
    maxprobs = []

    for cls in bbs_dict.keys():
        bb_probs  = [bb[-1] for bb in bbs_dict[cls]]
        maxprob = 0
        if len(bb_probs) > 0:
            maxprob = max([bb[-1] for bb in bbs_dict[cls]])
        maxprobs.append(maxprob)
    maxprobs[INFLORESCENCE_CLASS] = 0 # Only considering pollinators

    if max(maxprobs) == 0:
        # Don't want to be too confident here about null video and possibly override
        # someone else' prediction of a pollinator
        return None, 0, 0.4
    
    # Determine the maxprob class and its prob
    maxprob_class = np.argmax(maxprobs)
    maxprob = maxprobs[maxprob_class]

    # In order for the video to be a nul video all of the pollinator probs have to fail
    nullprob = np.prod(1 - np.array(maxprobs))

    return maxprob_class, maxprob, nullprob

def get_cls_probs_dict(clfr_labels_dir, fns):
    """
    Given a list of file names of label files produced by the classifier associated
    with a particular video, this returns a dictionary of lists of probabilities
    keyed by class num. Note that the class indices are slightly different for the
    classifiers
    """

    probs_dict = {}
    for i in range(7):
        probs_dict[i] = []

    for fn in fns:
        fp = os.path.join(clfr_labels_dir, fn)
        with open(fp, "r") as fd:
            line = fd.read()
            line = line.strip()
            line = line.rstrip()
            label_info = line.split(' ')
            if len(label_info) != 2:
                continue
            cls = int(label_info[0])
            prob = float(label_info[1])
            probs_dict[cls].append(prob)

    return probs_dict

def get_classifier_prediction_for_video(probs_dict):
    """
    Returns tuple (best pollinator class, prob pollinator)
    Note that best pollinator class can be null
    """

    maxprobs = []
    for cls in probs_dict.keys():
        probs = probs_dict[cls]
        maxprob = 0
        if len(probs) > 0:
            maxprob = max(probs)
        maxprobs.append(maxprob)

    # Not using the null image prob
    pollinator_maxprobs = copy(maxprobs)
    pollinator_maxprobs[NULL_IMAGE] = 0 # Not considering null prob here.
    
    if max(pollinator_maxprobs) == 0:
        # Don't want to be too confident here about null video and possibly override
        # someone else' prediction of a pollinator
        return None, 0, 0.4 # Don't want to be too confident here about null video
    
    maxprob_class = np.argmax(pollinator_maxprobs)
    maxprob = max(pollinator_maxprobs)

    # In order for the video to be a nul video all of the pollinator probs have to fail
    nullprob = np.prod(1 - np.array(maxprobs))
    
    return maxprob_class, maxprob, nullprob


            
if __name__ == "__main__":

    if len(sys.argv) != 8:
        print(f"Call is: {sys.argv[0]} <yolo labels dir> <clfr1 labels> <clfr2 labels> <outfile path> <motion threshold> <classes> <overlap threshold>")
        sys.exit(1)

    [_,yolo_labels_dir,clfr1_labels_dir, clfr2_labels_dir, outpath,motion_thresh,motion_classes,overlap_thresh] = sys.argv
    motion_thresh = float(motion_thresh)


    if len(motion_classes) == 0:
        motion_classes = []
    else:
        motion_classes = [int(c) for c in motion_classes.split(',')]

    overlap_thresh = float(overlap_thresh)

    if not os.path.exists(yolo_labels_dir):
        print(f"ERROR: {yolo_labels_dir} does not exit")
        sys.exit(2)

    vid_base_names = get_video_base_names(yolo_labels_dir)

    csv = ""

    count = 0
    for vbn in vid_base_names:
        csv_line = vbn

        # Get all the label file names for this video
        fns = get_label_filenames_for_video(yolo_labels_dir, vbn)

        # Make a dictionary keyed by class number with value giving a list of
        # all of the bound boxes associated with that class that where present
        # in some frame of the video

        yolo_bbs_dict = get_yolo_bbs_dict(yolo_labels_dir, fns)

        # Max dictionary of with lists of class probs of images in the video

        clfr1_probs_dict = get_cls_probs_dict(clfr1_labels_dir, fns)
        clfr2_probs_dict = get_cls_probs_dict(clfr2_labels_dir, fns)


        # Tally probability weighted votes
        yolo_votes = [0.05]*7
        yolo_cls,prob,nullprob = get_yolo_prediction_for_video(yolo_bbs_dict)
        if yolo_cls is not None:
            yolo_votes[yolo_cls] = prob
        yolo_votes[NULL_IMAGE] = nullprob
    
        clfr1_votes = [0.05]*7
        cls,prob,nullprob = get_classifier_prediction_for_video(clfr1_probs_dict)
        if cls is not None:
            clfr1_votes[cls] += prob
        clfr1_votes[NULL_IMAGE] = nullprob

        clfr2_votes = [0.05]*7
        cls,prob,nullprob = get_classifier_prediction_for_video(clfr2_probs_dict)
        if cls is not None:
            clfr2_votes[cls] += prob
        clfr2_votes[NULL_IMAGE] = nullprob

        # Form the vote tally as the geometric mean of the probabilities
        votes = np.power(np.array(yolo_votes)*np.array(clfr1_votes)*np.array(clfr2_votes),1/3)
        cls = np.argmax(votes)

        visits = "not pollinating"
        if cls != NULL_IMAGE:

            # If yolo thinks there are no pollinators, we don't have any bounding boxes
            # So we have to say its pollination status is unknown.
            if yolo_cls is None:
                visits = "unknown"
            else:

                pbb_list = yolo_bbs_dict[yolo_cls]
                inflorescence_bbs = yolo_bbs_dict[INFLORESCENCE_CLASS]

                # It is possible that cls is different from the yolo prediction. In this case,
                # we'll assume it got the bounding box right but the class wrong. We'll use bounding
                # boxes for that class to determine if it is pollinating

                visits = "pollinating" if visits_inflorescence(pbb_list, inflorescence_bbs, overlap_thresh) else "not pollinating"

        if cls == NULL_IMAGE:
            csv_line += ",no pollinator"
        else:
            csv_line += f",{CLS2NAME[cls]},{visits}"

        csv_line += "\n"
        csv += csv_line

        count += 1
        if count % 10 == 0:
            print(count)

    with open(outpath,"w") as fd:
        fd.write(csv)

    sys.exit(0)

            





            

            

        






    

