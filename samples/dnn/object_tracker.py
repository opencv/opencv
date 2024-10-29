#!/usr/bin/env python
'''
Tracker demo

For usage download models by following links
For DaSiamRPN:
    kernel_r1:       https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
    kernel_cls1:     https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0
    dasiamrpn_model: https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
For NanoTrack:
    nanotrack_backbone: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx
    nanotrack_headneck: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx
For VitTrack:
    vitTracker: https://github.com/opencv/opencv_zoo/raw/fef72f8fa7c52eaf116d3df358d24e6e959ada0e/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
'''

import sys
import cv2 as cv
import argparse
from common import *

def help():
    print(
        '''
        Firstly, download required models using the links provided in description. For vit tracker download model using `python download_models.py vit`
        Use this script for Object Tracking using OpenCV.

        To run:
            nano:
                Example: python object_tracker.py nano
            vit:
                Example: python object_tracker.py vit
            dasiamrpn:
                Example: python object_tracker.py dasiamrpn
        '''
    )

def createTracker():
    if args.alias == 'dasiamrpn':
        print("Using Dasiamrpn Tracker.")
        params = cv.TrackerDaSiamRPN_Params()
        params.model = findModel(args.model, args.sha1)
        params.kernel_cls1 = findModel(args.dasiamrpn_kernel_cls_model, args.dasiamrpn_kernel_cls_sha1)
        params.kernel_r1 = findModel(args.dasiamrpn_kernel_r1_model, args.dasiamrpn_kernel_r1_sha1)
        tracker = cv.TrackerDaSiamRPN_create(params)
    elif args.alias == 'nano':
        print("Using Nano Tracker.")
        params = cv.TrackerNano_Params()
        params.backbone = findModel(args.nanotrack_back_model, args.nanotrack_back_sha1)
        params.neckhead = findModel(args.nanotrack_head_model, args.nanotrack_head_sha1)
        tracker = cv.TrackerNano_create(params)
    elif args.alias == 'vit':
        print("Using Vit Tracker.")
        params = cv.TrackerVit_Params()
        params.net = findModel(args.model, args.sha1)
        tracker = cv.TrackerVit_create(params)
    else:
        help()
        exit(-1)
    return tracker

def main():
    tracker = createTracker()
    videoPath = args.input
    print('Using video: {}'.format(videoPath))
    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    if not cap.isOpened():
        print("Can't open video stream: {}".format(videoPath))
        exit(-1)

    stdSize = 0.6
    stdWeight = 2
    stdImgSize = 512
    imgWidth = -1 # Initialization
    fontSize = 1.5
    fontThickness = 1

    while True:
        ret, image = cap.read()
        if not ret:
            print("Error reading the video")
            return -1
        if imgWidth == -1:
            imgWidth = min(image.shape[:2])
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize)
            fontThickness = max(fontThickness,(stdWeight*imgWidth)//stdImgSize)

        label = "Press space bar to pause video to draw bounding box."
        labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
        cv.rectangle(image, (0, 0), (labelSize[0]+10, labelSize[1]+int(30*fontSize)), (255,255,255), cv.FILLED)
        cv.putText(image, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.putText(image, "Press space bar after selecting.", (10, int(50*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.imshow('TRACKING', image)

        key = cv.waitKey(100) & 0xFF
        if key == ord(' '):
            bbox = cv.selectROI('TRACKING', image)
            print('ROI: {}'.format(bbox))
            if bbox:
                break

        if key == ord('q') or key == 27:
            return
    try:
        tracker.init(image, bbox)
    except Exception as e:
        print('Unable to initialize tracker with requested bounding box. Is there any object?')
        print(e)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if imgWidth == -1:
            imgWidth = min(frame.shape[:2])
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize)
            fontThickness = max(fontThickness,(stdWeight*imgWidth)//stdImgSize)

        ok, newbox = tracker.update(frame)
        if ok:
            cv.rectangle(frame, newbox, (200, 0, 0), thickness=2)

        label="Tracking"
        labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
        cv.rectangle(frame, (0, 0), (labelSize[0]+10, labelSize[1]+10), (255,255,255), cv.FILLED)
        cv.putText(frame, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.imshow("TRACKING", frame)
        if cv.waitKey(100) & 0xFF in [ord('q'), 27]:
            break

if __name__ == '__main__':
    print(__doc__)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument("--input", type=str, help="Path to video source")
    args, _ = parser.parse_known_args()

    model_name = sys.argv[1]
    add_preproc_args(args.zoo, parser, 'object_tracker')
    if model_name == "dasiamrpn":
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="dasiamrpn_kernel_r1_")
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="dasiamrpn_kernel_cls_")
    elif model_name == "nano":
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="nanotrack_back_")
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="nanotrack_head_")
    elif model_name != "vit":
        print("Pass the valid alias. Choices are { nano, vit, dasiamrpn }")
        exit(0)
    parser = argparse.ArgumentParser(parents=[parser],
                                    description='Object tracking using OpenCV.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    main()
    cv.destroyAllWindows()
