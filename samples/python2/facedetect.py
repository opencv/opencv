import numpy as np
import cv2
from video import create_capture

if __name__ == '__main__':
    import sys, getopt
    
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = '0'
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml.xml")


    cam = create_capture(video_src)

    while True:
        ret, img = cam.read()
        cv2.imshow('facedetect', img)

        if cv2.waitKey(5) == 27:
            break

