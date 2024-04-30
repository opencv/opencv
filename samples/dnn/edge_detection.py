# Script is based on https://github.com/axinc-ai/ailia-models/blob/master/line_segment_detection/dexined/dexined.py
# To download the onnx model, see: https://storage.googleapis.com/ailia-models/dexined/model.onnx
# python edge_detection.py --model model.onnx --input chicky_512.png --method dexined
import cv2 as cv
import argparse
import numpy as np

def parse_args():
     # Define supported computation backends and target devices for OpenCV DNN
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                    cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)

    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
                cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    # Define supported edge detection methods
    methods = ("dexined", "canny")

    # Set up argument parser
    parser = argparse.ArgumentParser(
            description='This sample shows how to use the '
                        'Dense Extreme Inception Network for Edge Detection (Dexened Edge Detection) (https://arxiv.org/abs/2112.02250)'
                        'model in Python. Find the pre-trained model at https://storage.googleapis.com/ailia-models/dexined/model.onnx')

    parser.add_argument('--input', help='Path to image', default="chicky_512.png", required= False)

    parser.add_argument('--model', help='Path to onnx model', required=True)

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
                                '%d: CUDA fp16 (half-float preprocess)'% targets)

    parser.add_argument('--image_size', help='Resize input image to a size', default=512, type=int)

    parser.add_argument('--method', choices=methods, default= "dexined", type=str,
                        help = 'Choose one of the method : '
                        '%s: dexined, '
                        '%s: canny'% methods)

    args = parser.parse_args()

    return args

def preprocess(img):
    # Resize and normalize the image

    IMAGE_SIZE = args.image_size
    img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    img = img.astype(np.float32)
    img[:, :, 0] = (img[:, :, 0] - 103.939)
    img[:, :, 1] = (img[:, :, 1] - 116.779)
    img[:, :, 2] = (img[:, :, 2] - 123.68)

    return img

def sigmoid(x):
    # Apply the sigmoid function
    return 1.0 / (1.0 + np.exp(-x))

def post_processing(output, shape):
    # Process network output to generate edge maps
    h, w = shape

    preds = []
    epsilon = 1e-12
    for p in output:
        img = sigmoid(p)
        img = np.squeeze(img)
        img = (img - np.min(img)) * 255 / \
              ((np.max(img) - np.min(img)) + epsilon)
        img = img.astype(np.uint8)
        img = cv.resize(img, (w, h))
        preds.append(img)

    fuse = preds[-1]

    ave = np.array(preds, dtype=np.float32)
    ave = np.uint8(np.mean(ave, axis=0))

    return fuse, ave

def canny_detection_thresh1(position, user_data):
    # Threshold adjustment callback for Canny edge detection
    out = cv.Canny(user_data["gray"],position,user_data["thrs2"])
    user_data["thrs1"] =  position

    cv.imshow('Output', out)

def canny_detection_thresh2(position, user_data):
    # Threshold adjustment callback for Canny edge detection
    out = cv.Canny(user_data["gray"],user_data["thrs1"],position)
    user_data["thrs2"] =  position
    cv.imshow('Output', out)


if __name__ == '__main__':
    args = parse_args()
    method = args.method
    user_data = {"gray": None, "thrs1": 100, "thrs2": 200}

    cv.namedWindow('Output', cv.WINDOW_NORMAL)
    cv.namedWindow('Input', cv.WINDOW_NORMAL)

    if method == "canny":
        # Load and process the image for Canny edge detection
        img = cv.imread(cv.samples.findFile(args.input))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        user_data["gray"] = gray

        cv.createTrackbar('thrs1', 'Output', 0, 255, lambda value: canny_detection_thresh1(value, user_data))
        cv.setTrackbarPos('thrs1', 'Output', 100)
        cv.createTrackbar('thrs2', 'Output', 0, 255, lambda value: canny_detection_thresh2(value, user_data))
        cv.setTrackbarPos('thrs2', 'Output', 200)
        cv.imshow("Input", img)
        cv.waitKey(0)

    elif method == "dexined":
        # Load the model
        session = cv.dnn.readNetFromONNX(args.model)
        session.setPreferableBackend(args.backend)
        session.setPreferableTarget(args.target)
        # Load and process the image using DexiNed model
        orig_img = cv.imread(cv.samples.findFile(args.input))
        # Prepocess the image
        img = preprocess(orig_img)

        inp = cv.dnn.blobFromImage(img, swapRB=False, crop=False)

        session.setInput(inp)

        out = session.forward()
        # Post processing on the model output
        out = post_processing(out, (orig_img.shape[1], orig_img.shape[0]))

        cv.imshow("Input", orig_img)
        cv.imshow("Output", out[1])
        cv.waitKey(0)