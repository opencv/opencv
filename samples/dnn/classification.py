import argparse

import cv2 as cv
import numpy as np
from common import *


def get_args_parser(func_args):
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
               cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input',
                        help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'torch', 'darknet'],
                        help='Optional name of an origin framework of the model. '
                             'Detect it automatically if it does not set.')
    parser.add_argument('--std', nargs='*', type=float,
                        help='Preprocess input image by dividing on a standard deviation.')
    parser.add_argument('--crop', type=bool, default=False,
                        help='Preprocess input image by dividing on a standard deviation.')
    parser.add_argument('--initial_width', type=int,
                        help='Preprocess input image by initial resizing to a specific width.')
    parser.add_argument('--initial_height', type=int,
                        help='Preprocess input image by initial resizing to a specific height.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Halide language (http://halide-lang.org/), "
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
                             '%d: CUDA fp16 (half-float preprocess)'% targets)

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'classification')
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='Use this script to run classification deep learning networks using OpenCV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args(func_args)


def main(func_args=None):
    args = get_args_parser(func_args)
    args.model = findFile(args.model)
    args.config = findFile(args.config)
    args.classes = findFile(args.classes)

    # Load names of classes
    classes = None
    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

    # Load a network
    net = cv.dnn.readNet(args.model, args.config, args.framework)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)

    winName = 'Deep learning image classification in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    cap = cv.VideoCapture(args.input if args.input else 0)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # Create a 4D blob from a frame.
        inpWidth = args.width if args.width else frame.shape[1]
        inpHeight = args.height if args.height else frame.shape[0]

        if args.initial_width and args.initial_height:
            frame = cv.resize(frame, (args.initial_width, args.initial_height))

        blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=args.crop)
        if args.std:
            blob[0] /= np.asarray(args.std, dtype=np.float32).reshape(3, 1, 1)

        # Run a model
        net.setInput(blob)
        out = net.forward()

        # Get a class with a highest score.
        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]

        # Put efficiency information.
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Print predicted class.
        label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
        cv.putText(frame, label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv.imshow(winName, frame)


if __name__ == "__main__":
    main()
