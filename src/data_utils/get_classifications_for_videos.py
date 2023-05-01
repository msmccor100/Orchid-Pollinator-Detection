"""
Given a directory containing videos and a CSV file containing video file names and a classification (in 
some indicated column), this routine creates a csv file in the video directory called classifications.csv
and containing rows

<video name>,<classification>

call is: get_classifications_for_videos.py <videos dir> <csv file> <classification column>

The classificaiton column is an integer counting from 0
"""

import sys
import os
import csv

if len(sys.argv) != 4:
    print("Call is: get_classifications_for_videos.py <videos dir> <csv file> <classification column>")
    sys.exit(1)

viddir = sys.argv[1]
if not os.path.exists(viddir):
    print(f"ERROR: {viddir} does not exist")
    sys.exit(2)

csv_fp = sys.argv[2]
if not os.path.exists(csv_fp):
    print(f"ERROR: {csv_fp} does not exist")
    sys.exit(2)

col = int(sys.argv[3])

classification_dict = {}
with open(csv_fp, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        if len(row) > col:
            classification_dict[row[0]] = row[col]

outdata = ""
for fn in os.listdir(viddir):
    if not ".mp4" in fn:
        continue
    if fn not in classification_dict:
        fp = os.path.join(viddir, fn)
        os.remove(fp)
        continue
    outdata += fn + "," + classification_dict[fn] + "\n"

outfile = os.path.join(viddir,"classifications.csv")
with open(outfile,"w") as fd:
    fd.write(outdata)



