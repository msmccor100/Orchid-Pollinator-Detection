"""
This r
Given a dataset of image frame files take from videos where the imge file names are of
of the form 
    <video base file name>_<separator><more stuff>.jpg
where the associated video has name
    <video base file name>.mpr
and given a directory tree containing the source videos, this routine finds all of the
videos associate with the image files and copies them to a specified directory.

call is:
    find_videos_for_files.py <image dir> <separator> <video directory tree> <destination dir>

"""

import sys
import os
import shutil

VIDEO_FILE_EXT = ".mp4"

def get_videos_filepaths(vid_tree):
    vid_dict = {} # Keyed by video file names and with values their file paths
    subdir_stack = [vid_tree]

    while(len(subdir_stack)):
        curdir = subdir_stack.pop()

        for fn in os.listdir(curdir):
            fp = os.path.join(curdir,fn)
            if os.path.isdir(fp):
                subdir_stack.append(fp)
            else:
                base,ext = os.path.splitext(fn)

                ext = ext.lower()
                if ext == VIDEO_FILE_EXT:

                    # Roboflow changes video names when creating images in some bizarre ways:
                    base = base.replace(".","_")
                    base = base.replace(" Copy of ","Copy-of-")                
                    vid_dict[base] = fp





    return vid_dict



if __name__ == "__main__":

    if len(sys.argv) < 5:
        print(f"Call is: {sys.argv[0]} <image dir> <separator> <video directory tree> <destination dir>")
        sys.exit(1)

    img_dir = sys.argv[1]
    if not os.path.exists(img_dir):
        print(f"ERROR: {img_dir} does not exist.")
        sys.exit(1)

    vid_tree = sys.argv[3]
    if not os.path.exists(vid_tree):
        print(f"ERROR: {vid_tree} does not exist.")
        sys.exit(1)

    separator = sys.argv[2]
    dest_dir = sys.argv[4]

    vid_dict = get_videos_filepaths(vid_tree)
    print(f"{len(list(vid_dict.keys()))} videos found")

    #for fn in vid_dict.keys():
    #    print(f"Found video {fn}")

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    videos_written = set()

    copy_count = 0
    not_found_count = 0;
    for fn in os.listdir(img_dir):
        if os.path.isdir(fn):
            continue
        if ".jpg" not in fn:
            continue
        if separator not in fn:
            print(f"WARNING: Separator not in {fn}")
            continue
        basename = fn[:fn.find(separator)]
        if basename[-3:] == "-1-":
            basename = basename[:-3]
        vidname = basename + VIDEO_FILE_EXT
        if basename not in vid_dict:
            print(f"WARNING: {vidname} not found" )
            not_found_count += 1
            continue

        # We have many images coming from the same video
        if vidname in videos_written:
            continue

        vid_path = vid_dict[basename]
        dest_vid_path = os.path.join(dest_dir,vidname)
        shutil.copyfile(vid_path,dest_vid_path)
        videos_written.add(vidname)
        copy_count += 1

    print(f"\nCopied {copy_count} videos")
    print(f"{not_found_count} videos not found")
