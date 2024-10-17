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
                Example: python object_tracker.py nano --nanotrack_backbone=<path to nanotrack_backbone onnx model> --nanotrack_head=<path to nanotrack_head onnx model>
            vit:
                Example: python object_tracker.py vit --model=<path to vitTracker onnx model>
            dasiamrpn:
                Example: python object_tracker.py dasiamrpn --dasiamrpn_model=<path to dasiamrpn_model onnx model> --kernel_r1=<path to dasiamrpn_kernel_r1 onnx model> --kernel_cls1=<path to dasiamrpn_kernel_cls1 onnx model>
        '''
    )

def createTracker():
    if args.alias == 'dasiamrpn' or args.dasiamrpn_model is not None:
        if args.dasiamrpn_model is None or args.kernel_r1 is None or args.kernel_cls1 is None:
            print("Pass model files using --dasiamrpn_model , --kernel_cls1 and --kernel_r1 arguments for using dasiamrpn tracker. \nDownload dasiamrpn_model using link: https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0")
            print("And, download kernel_r1 using link: https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0")
            print("And, download kernel_cls1 using link: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0")
            exit(-1)
        print("Using Dasiamrpn Tracker.")
        params = cv.TrackerDaSiamRPN_Params()
        params.model = findModel(args.dasiamrpn_model, "")
        params.kernel_cls1 = findModel(args.kernel_cls1, "")
        params.kernel_r1 = findModel(args.kernel_r1, "")
        tracker = cv.TrackerDaSiamRPN_create(params)
    elif args.alias == 'nano' or args.nanotrack_head is not None:
        if args.nanotrack_backbone is None or args.nanotrack_head is None:
            print("Pass model files using --nanotrack_head and --nanotrack_backbone arguments for using nano tracker. \nDownload nanotrack_head using link: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx")
            print("And, download nanotrack_backbone using link: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx")
            exit(-1)
        print("Using Nano Tracker.")
        params = cv.TrackerNano_Params()
        params.backbone = findModel(args.nanotrack_backbone, "")
        params.neckhead = findModel(args.nanotrack_head, "")
        tracker = cv.TrackerNano_create(params)
    elif args.alias == 'vit' or args.model is not None:
        print("Using Vit Tracker.")
        sha1 = ""
        if hasattr(args, "sha1"):
            sha1 = args.sha1
        params = cv.TrackerVit_Params()
        params.net = findModel(args.model, sha1)
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
    parser.add_argument("alias", type=str, nargs='?', help="alias i.e. (vit, nano, or dasiamrpn)")
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument("--input", type=str, help="Path to video source")
    parser.add_argument("--dasiamrpn_model", type=str, help="Path to onnx model of DaSiamRPN net")
    parser.add_argument("--kernel_r1", type=str, help="Path to onnx model of DaSiamRPN kernel_r1")
    parser.add_argument("--kernel_cls1", type=str, help="Path to onnx model of DaSiamRPN kernel_cls1")
    parser.add_argument("--nanotrack_backbone", type=str, help="Path to onnx model of NanoTrack backBone")
    parser.add_argument("--nanotrack_head", type=str, help="Path to onnx model of NanoTrack headNeck")
    args, _ = parser.parse_known_args()
    if args.alias == "vit":
        add_preproc_args(args.zoo, parser, 'object_tracker', alias="vit")
        parser = argparse.ArgumentParser(parents=[parser],
                                        description='Object tracking using OpenCV.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        parser.add_argument("--model", type=str, help="Path to onnx model of  vittrack")
        parser = argparse.ArgumentParser(parents=[parser], add_help=True)
    args = parser.parse_args()
    main()
    cv.destroyAllWindows()
