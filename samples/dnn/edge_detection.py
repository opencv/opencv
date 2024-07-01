'''
This sample demonstrates edge detection with dexined and canny edge detection techniques.
For switching between deep learning based model(dexined) and canny edge detector, press 'd' (for dexined) or 'c' (for canny) respectively in case of video. For image pass the argument --method for switching between dexined and canny.
'''

import cv2 as cv
import argparse
import numpy as np
from common import *

def get_args_parser(func_args):
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)

    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16,
               cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN,
               cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required=False)
    parser.add_argument('--method', help='choose method: dexined or canny', default='canny', required=False)
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA" % backends)

    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: NCS2 VPU, '
                             '%d: HDDL VPU, '
                             '%d: Vulkan, '
                             '%d: CUDA, '
                             '%d: CUDA fp16 (half-float preprocess)' % targets)

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'edge_detection')
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='Use this script to run edge detection using OpenCV. '
                                     "This sample demonstrates edge detection with Dexined and Canny edge detection techniques. "
                                        "In case of video input, for switching between deep learning based model (Dexined) and Canny edge detector, press 'd' (for Dexined) or 'c' (for Canny) respectively. Pass as argument in case of image input.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args(func_args)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def post_processing(output, shape):
    h, w = shape
    preds = []
    for p in output:
        img = sigmoid(p)
        img = np.squeeze(img)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        img = cv.resize(img, (w, h))
        preds.append(img)
    fuse = preds[-1]
    ave = np.array(preds, dtype=np.float32)
    ave = np.uint8(np.mean(ave, axis=0))
    return fuse, ave

def canny_detection_thresh1(position, user_data):
    user_data["thrs1"] = position
    apply_canny(user_data)

def canny_detection_thresh2(position, user_data):
    user_data["thrs2"] = position
    apply_canny(user_data)

def apply_canny(user_data):
    blurred = cv.GaussianBlur(user_data["gray"], (user_data["blur"] * 2 + 1, user_data["blur"] * 2 + 1), 0)
    out = cv.Canny(blurred, user_data["thrs1"], user_data["thrs2"])
    cv.imshow('Output', out)

def blur_change(position, user_data):
    user_data["blur"] = position
    apply_canny(user_data)

def setupCannyWindow(image, user_data):
    cv.destroyWindow('Output')
    cv.namedWindow('Output', cv.WINDOW_NORMAL)
    cv.moveWindow('Output', 200, 50)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    user_data["gray"] = gray
    cv.createTrackbar('thrs1', 'Output', 0, 255, lambda value: canny_detection_thresh1(value, user_data))
    cv.setTrackbarPos('thrs1', 'Output', 100)
    cv.createTrackbar('thrs2', 'Output', 0, 255, lambda value: canny_detection_thresh2(value, user_data))
    cv.setTrackbarPos('thrs2', 'Output', 200)
    cv.createTrackbar('blur', 'Output', 1, 20, lambda value: blur_change(value, user_data))
    cv.setTrackbarPos('blur', 'Output', 1)

def loadModel(args):
    net = cv.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)
    return net

def main(func_args=None):
    args = get_args_parser(func_args)

    user_data = {"gray": None, "thrs1": 100, "thrs2": 200, "blur": 1}
    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    if not cap.isOpened():
        print("Failed to open the input video")
        exit(-1)
    cv.namedWindow('Input', cv.WINDOW_NORMAL)
    cv.namedWindow('Output', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('Output', 200, 50)

    method = args.method

    if not hasattr(args, 'model'):
        print("[WARN] Model file not provided, cannot use dexined")
        method = 'canny'
    else:
        args.model = findFile(args.model)
    if method == 'canny':
        dummy = np.zeros((512, 512, 3), dtype="uint8")
        setupCannyWindow(dummy, user_data)

    net = None
    if method == "dexined":
        net = loadModel(args)
    while cv.waitKey(1) < 0:
        hasFrame, image = cap.read()
        if not hasFrame:
            print("Press any key to exit")
            cv.waitKey(0)
            break
        if method == "canny":
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            user_data["gray"] = gray
            canny_detection_thresh1(user_data["thrs1"], user_data)
        elif method == "dexined":
            inp = cv.dnn.blobFromImage(image, args.scale, (args.width, args.height), args.mean, swapRB=args.rgb, crop=False)
            net.setInput(inp)
            out = net.forward()
            out = post_processing(out, image.shape[:2])
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv.putText(out[1], label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv.imshow("Output", out[1])
        cv.imshow("Input", image)
        key = cv.waitKey(30)
        if key == ord('d') or key == ord('D'):
            if hasattr(args, 'model'):
                method = "dexined"
                if net is None:
                    net = loadModel(args)
                cv.destroyWindow('Output')
                cv.namedWindow('Output', cv.WINDOW_AUTOSIZE)
                cv.moveWindow('Output', 200, 50)
            else:
                print("[ERROR] Provide model file using --model to use dexined")
        elif key == ord('c') or key == ord('C'):
            method = "canny"
            setupCannyWindow(image, user_data)
        elif key == 27 or key == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()