#!/usr/bin/env python
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

def load_parser(model_name):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument("--input", type=str, help="Path to video source")
    args, _ = parser.parse_known_args()

    add_preproc_args(args.zoo, parser, 'object_tracker', alias=model_name)
    if model_name == "dasiamrpn":
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="dasiamrpn_", alias="dasiamrpn")
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="dasiamrpn_kernel_r1_", alias="dasiamrpn")
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="dasiamrpn_kernel_cls_", alias="dasiamrpn")
    elif model_name == "nano":
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="nanotrack_back_", alias="nano")
        add_preproc_args(args.zoo, parser, 'object_tracker', prefix="nanotrack_head_", alias="nano")
    elif model_name != "vit":
        print("Pass the valid alias. Choices are { nano, vit, dasiamrpn }")
        exit(0)
    parser = argparse.ArgumentParser(parents=[parser],
                                    description='''
    Firstly, download required models using `python download_models.py {modelName}`
    Use this script for Object Tracking using OpenCV.
    To run:
        nano:
            Example: python object_tracker.py nano
        vit:
            Example: python object_tracker.py vit
        dasiamrpn:
            Example: python object_tracker.py dasiamrpn
    ''',
                                    formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args()

def createTracker(model_name, args):
    if model_name == 'dasiamrpn':
        print("Using Dasiamrpn Tracker.")
        params = cv.TrackerDaSiamRPN_Params()
        params.model = findModel(args.dasiamrpn_model, args.dasiamrpn_sha1)
        params.kernel_cls1 = findModel(args.dasiamrpn_kernel_cls_model, args.dasiamrpn_kernel_cls_sha1)
        params.kernel_r1 = findModel(args.dasiamrpn_kernel_r1_model, args.dasiamrpn_kernel_r1_sha1)
        tracker = cv.TrackerDaSiamRPN_create(params)
    elif model_name == 'nano':
        print("Using Nano Tracker.")
        params = cv.TrackerNano_Params()
        params.backbone = findModel(args.nanotrack_back_model, args.nanotrack_back_sha1)
        params.neckhead = findModel(args.nanotrack_head_model, args.nanotrack_head_sha1)
        tracker = cv.TrackerNano_create(params)
    elif model_name == 'vit':
        print("Using Vit Tracker.")
        params = cv.TrackerVit_Params()
        params.net = findModel(args.model, args.sha1)
        tracker = cv.TrackerVit_create(params)
    else:
        help()
        exit(-1)
    return tracker

def main(model_name, args):
    tracker = createTracker(model_name, args)
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
    alpha = 0.5
    windowName = "TRACKING"
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)

    while True:
        ret, image = cap.read()
        if not ret:
            print("Video completed!!")
            return -1
        if imgWidth == -1:
            imgWidth = min(image.shape[:2])
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize)
            fontThickness = max(fontThickness,(stdWeight*imgWidth)//stdImgSize)
        org_img = image.copy()
        label = "Press space bar to pause video to draw bounding box."
        labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
        cv.rectangle(image, (0, 0), (labelSize[0]+10, labelSize[1]+int(40*fontSize)), (255,255,255), cv.FILLED)
        cv.addWeighted(image, alpha, org_img, 1 - alpha, 0, image)
        cv.putText(image, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.putText(image, "Press space bar after selecting.", (10, int(55*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.imshow(windowName, image)

        key = cv.waitKey(30) & 0xFF
        if key == ord(' '):
            bbox = cv.selectROI(windowName, image)
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

    tick_meter = cv.TickMeter()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if imgWidth == -1:
            imgWidth = min(frame.shape[:2])
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize)
            fontThickness = max(fontThickness,(stdWeight*imgWidth)//stdImgSize)
        tick_meter.reset()
        tick_meter.start()
        ok, newbox = tracker.update(frame)
        tick_meter.stop()
        score = tracker.getTrackingScore()
        render_image = frame.copy()

        key = cv.waitKey(30) & 0xFF
        label="Press space bar to select new target"
        labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
        h, w = frame.shape[:2]
        cv.rectangle(render_image, (0, 0), (labelSize[0]+10, labelSize[1]+int(100*fontSize)), (255,255,255), cv.FILLED)
        cv.rectangle(render_image, (0, int(h-45*fontSize)), (w, h), (255,255,255), cv.FILLED)
        cv.addWeighted(render_image, alpha, frame, 1 - alpha, 0, render_image)
        cv.putText(render_image, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.putText(render_image, "For switching between trackers: press 'v' for ViT, 'n' for Nano, and 'd' for DaSiamRPN.", (10, h-10), cv.FONT_HERSHEY_SIMPLEX, 0.8*fontSize, (0, 0, 0), fontThickness)

        if ok:
            if key == ord(' '):
                cv.putText(render_image, "Select the new target", (10, int(55*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
                bbox = cv.selectROI(windowName, render_image)
                print('ROI:', bbox)
                if bbox:
                    tracker.init(frame, bbox)
            elif key == ord('v'):
                model_name = "vit"
                args = load_parser(model_name)
                tracker = createTracker(model_name, args)
                tracker.init(frame, newbox)
            elif key == ord('n'):
                model_name = "nano"
                args = load_parser(model_name)
                tracker = createTracker(model_name, args)
                tracker.init(frame, newbox)
            elif key == ord('d'):
                model_name = "dasiamrpn"
                args = load_parser(model_name)
                tracker = createTracker(model_name, args)
                tracker.init(frame, newbox)
            elif key == ord('q') or key == 27:
                return

            cv.rectangle(render_image, newbox, (200, 0, 0), thickness=2)
        time_label = f"Inference time: {tick_meter.getTimeMilli():.2f} ms"
        score_label = f"Tracking score: {score:.2f}"
        algo_label = f"Algorithm: {model_name}"
        cv.putText(render_image, time_label, (10, int(55*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.putText(render_image, score_label, (10, int(85*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.putText(render_image, algo_label, (10, int(115*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)

        cv.imshow(windowName, render_image)
        if key in [ord('q'), 27]:
            break

if __name__ == '__main__':
    if len(sys.argv) < 2:
        help()
        exit(-1)

    model_name = sys.argv[1]
    args = load_parser(model_name)

    main(model_name, args)
    cv.destroyAllWindows()
