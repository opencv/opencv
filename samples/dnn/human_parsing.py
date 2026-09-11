#!/usr/bin/env python
'''
Human parsing (segmenting human body parts) with JPPNet, based on
https://github.com/Engineering-Course/LIP_JPPNet

Download the ONNX model from
https://huggingface.co/opencv/opencv_contribution/tree/main/human_parsing_jppnet
or convert the original TensorFlow graph yourself with the convert_to_onnx.py
script published next to it.

The graph is frozen at 384x384 and always takes two pictures at once: the image
and its horizontal mirror. The two predictions are averaged, with the left/right
body part channels of the mirrored half swapped back first.
'''

import argparse
import os.path
import numpy as np
import cv2 as cv


backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
           cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)


def preprocess(image):
    """
    Create 4-dimensional blob from image and flip image
    :param image: input image
    """
    resized = cv.resize(image, (384, 384))
    image_rev = np.flip(resized, axis=1)
    input = cv.dnn.blobFromImages([resized, image_rev], mean=(104.00698793, 116.66876762, 122.67891434))
    return input


def run_net(input, model_path, backend, target):
    """
    Read network and infer model
    :param model_path: path to JPPNet model
    :param backend: computation backend
    :param target: computation device
    """
    net = cv.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    net.setInput(input)
    out = net.forward()
    return out


def postprocess(out, input_shape):
    """
    Create a grayscale human segmentation
    :param out: network output
    :param input_shape: input image width and height
    """
    # LIP classes
    # 0 Background
    # 1 Hat
    # 2 Hair
    # 3 Glove
    # 4 Sunglasses
    # 5 UpperClothes
    # 6 Dress
    # 7 Coat
    # 8 Socks
    # 9 Pants
    # 10 Jumpsuits
    # 11 Scarf
    # 12 Skirt
    # 13 Face
    # 14 LeftArm
    # 15 RightArm
    # 16 LeftLeg
    # 17 RightLeg
    # 18 LeftShoe
    # 19 RightShoe
    head_output, tail_output = np.split(out, indices_or_sections=[1], axis=0)
    head_output = head_output.squeeze(0)
    tail_output = tail_output.squeeze(0)

    head_output = np.stack([cv.resize(img, dsize=input_shape) for img in head_output[:, ...]])
    tail_output = np.stack([cv.resize(img, dsize=input_shape) for img in tail_output[:, ...]])

    tail_list = np.split(tail_output, indices_or_sections=list(range(1, 20)), axis=0)
    tail_list = [arr.squeeze(0) for arr in tail_list]
    tail_list_rev = [tail_list[i] for i in range(14)]
    tail_list_rev.extend([tail_list[15], tail_list[14], tail_list[17], tail_list[16], tail_list[19], tail_list[18]])
    tail_output_rev = np.stack(tail_list_rev, axis=0)
    tail_output_rev = np.flip(tail_output_rev, axis=2)
    raw_output_all = np.mean(np.stack([head_output, tail_output_rev], axis=0), axis=0, keepdims=True)
    raw_output_all = np.argmax(raw_output_all, axis=1)
    raw_output_all = raw_output_all.transpose(1, 2, 0)
    return raw_output_all


def decode_labels(gray_image):
    """
    Colorize image according to labels
    :param gray_image: grayscale human segmentation result
    """
    height, width, _ = gray_image.shape
    colors = [(0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0),
              (0, 0, 85), (0, 119, 221), (85, 85, 0), (0, 85, 85), (85, 51, 0), (52, 86, 128),
              (0, 128, 0), (0, 0, 255), (51, 170, 221), (0, 255, 255),(85, 255, 170),
              (170, 255, 85), (255, 255, 0), (255, 170, 0)]

    segm = np.stack([colors[idx] for idx in gray_image.flatten()])
    segm = segm.reshape(height, width, 3).astype(np.uint8)
    segm = cv.cvtColor(segm, cv.COLOR_BGR2RGB)
    return segm


def parse_human(image, model_path, backend=cv.dnn.DNN_BACKEND_OPENCV, target=cv.dnn.DNN_TARGET_CPU):
    """
    Prepare input for execution, run net and postprocess output to parse human.
    :param image: input image
    :param model_path: path to JPPNet model
    :param backend: name of computation backend
    :param target: name of computation target
    """
    input = preprocess(image)
    output = run_net(input, model_path, backend, target)
    grayscale_out = postprocess(output, (image.shape[1], image.shape[0]))
    segmentation = decode_labels(grayscale_out)
    return segmentation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to run human parsing using JPPNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', required=True, help='Path to input image.')
    parser.add_argument('--model', '-m', default='human_parsing_jppnet_2026sep.onnx', help='Path to onnx model.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA"% backends)
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

    if not os.path.isfile(args.model):
        raise OSError("Model not exist")

    image = cv.imread(args.input)
    output = parse_human(image, args.model, args.backend, args.target)
    winName = 'Deep learning human parsing in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    cv.imshow(winName, output)
    cv.waitKey()
