#!/usr/bin/env python3
import sys
import os
import cv2

DATA_DIR = 'images_new/'

def main():
    vid_path = sys.argv[1]
    base_path = vid_path.replace('_out.mp4', '')
    stamp_path = base_path + '_stamps.txt'
    vidcap = cv2.VideoCapture(vid_path)
    tstamp_f = open(stamp_path, 'r')
    out_dir = DATA_DIR + base_path
    os.makedirs(out_dir, exist_ok=True)
    success,image = vidcap.read()
    tstamp_f.readline() # discard first line
    tstamp = int(tstamp_f.readline().split('.')[0])
    count = 0
    while success:
        cv2.imwrite(out_dir+"/frame{:04d}_{:05d}.jpg".format(count, tstamp), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        tstamp = int(tstamp_f.readline().split('.')[0])
        count += 1


if __name__ == '__main__':
    main()
