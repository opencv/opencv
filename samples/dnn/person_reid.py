#!/usr/bin/env python
'''
This sample detects the query person in the given video file.

Authors of samples and Youtu ReID baseline:
        Xing Sun <winfredsun@tencent.com>
        Feng Zheng <zhengf@sustech.edu.cn>
        Xinyang Jiang <sevjiang@tencent.com>
        Fufu Yu <fufuyu@tencent.com>
        Enwei Zhang <miyozhang@tencent.com>

Copyright (C) 2020-2021, Tencent.
Copyright (C) 2020-2021, SUSTech.

How to use:
    sample command to run:

        python person_reid.py --query=/path/to/query_image(optional) --video=/path/to/video/footage --model=path/to/youtu_reid_baseline_medium.onnx --yolo=/path/to/yolov8.onnx

    You can download a baseline ReID model from:
        https://github.com/ReID-Team/ReID_extra_testdata

'''
import argparse
import os.path
import numpy as np
import cv2 as cv

backends = (cv.dnn.DNN_BACKEND_DEFAULT,
    cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
    cv.dnn.DNN_BACKEND_OPENCV,
    cv.dnn.DNN_BACKEND_VKCOM,
    cv.dnn.DNN_BACKEND_CUDA)

targets = (cv.dnn.DNN_TARGET_CPU,
    cv.dnn.DNN_TARGET_OPENCL,
    cv.dnn.DNN_TARGET_OPENCL_FP16,
    cv.dnn.DNN_TARGET_MYRIAD,
    cv.dnn.DNN_TARGET_HDDL,
    cv.dnn.DNN_TARGET_VULKAN,
    cv.dnn.DNN_TARGET_CUDA,
    cv.dnn.DNN_TARGET_CUDA_FP16)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
drawing = False
ix, iy = -1, -1
rect = None
img_dict = {} # Dictionary to store bounding boxes for corresponding cropped image

def preprocess(images, height, width):
    """
    Create 4-dimensional blob from image
    :param image: input image
    :param height: the height of the resized input image
    :param width: the width of the resized input image
    """
    img_list = []
    for image in images:
        image = cv.resize(image, (width, height))
        img_list.append(image[:, :, ::-1])

    images = np.array(img_list)
    images = (images / 255.0 - MEAN) / STD

    input = cv.dnn.blobFromImages(images.astype(np.float32), ddepth = cv.CV_32F)
    return input

def yolo_detector(frame, net):
    global img_dict
    height, width, _ = frame.shape

    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame

    scale = length/640
    # Create blob from the frame with correct scale factor and size for the model

    blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    net.setInput(blob)
    outputs = net.forward()

    outputs = np.array([cv.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (_, maxScore, _, (x, maxClassIndex)) = cv.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)


    # Apply Non-Maximum Suppression
    indexes = cv.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    images = []
    for i in indexes:
        x, y, w, h = boxes[i]
        x = round(x*scale)
        y = round(y*scale)
        w = round(w*scale)
        h = round(h*scale)

        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        crop_img = frame[y:y+h, x:x+w]
        images.append(crop_img)
        img_dict[crop_img.tobytes()] = (x, y, w, h)
    return images

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param[0].copy()
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv.imshow('TRACKING', img_copy)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)
        cv.rectangle(param[0], (ix, iy), (x, y), (0, 255, 0), 2)
        cv.imshow('TRACKING', param[0])

def extract_frames(query_image_path, video_path, model_path, yolo_path, resize_h=384, resize_w=128, backend=cv.dnn.DNN_BACKEND_OPENCV, target=cv.dnn.DNN_TARGET_CPU, batch_size=32):
    cap = cv.VideoCapture(video_path)
    net = cv.dnn.readNet(yolo_path)
    query_images = []

    if query_image_path:
        query_images = [cv.imread(query_image_path)]
    else:
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading the video")
            return
        cv.putText(first_frame, "Draw Bounding Box on Target", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv.imshow('TRACKING', first_frame)
        cv.setMouseCallback('TRACKING', draw_rectangle, [first_frame])

        while True:
            if rect:
                break
            if cv.waitKey(1) & 0xFF == ord('q'):
                return

        x1, y1, x2, y2 = rect
        query_image = first_frame[y1:y2, x1:x2]
        query_images = [query_image]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        images = yolo_detector(frame, net)
        query_feat = extract_feature(query_images, model_path, resize_h, resize_w, backend, target, batch_size)
        gallery_feat = extract_feature(images, model_path, resize_h, resize_w, backend, target, batch_size)

        topk_idx = topk(query_feat, gallery_feat)

        top_img = images[topk_idx]
        x, y, w, h = img_dict[top_img.tobytes()]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, "Target", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv.putText(frame, "Tracking", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv.imshow("TRACKING", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return

def extract_feature(images, model_path, resize_h = 384, resize_w = 128, backend=cv.dnn.DNN_BACKEND_OPENCV, target=cv.dnn.DNN_TARGET_CPU, batch_size = 32):
    """
    Extract features from images
    :param images: the input images
    :param model_path: path to ReID model
    :param batch_size: the batch size for each network inference iteration
    :param resize_h: the height of the input image
    :param resize_w: the width of the input image
    :param backend: name of computation backend
    :param target: name of computation target
    """
    feat_list = []

    for i in range(0, len(images), batch_size):
        batch = images[i : min(i + batch_size, len(images))]
        inputs = preprocess(batch, resize_h, resize_w)

        feat = run_net(inputs, model_path, backend, target)

        feat_list.append(feat)

    feats = np.concatenate(feat_list, axis = 0)
    return feats

def run_net(inputs, model_path, backend=cv.dnn.DNN_BACKEND_OPENCV, target=cv.dnn.DNN_TARGET_CPU):
    """
    Forword propagation for a batch of images.
    :param inputs: input batch of images
    :param model_path: path to ReID model
    :param backend: name of computation backend
    :param target: name of computation target
    """
    net = cv.dnn.readNet(model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    net.setInput(inputs)
    out = net.forward()
    out = np.reshape(out, (out.shape[0], out.shape[1]))
    return out

def normalize(nparray, order=2, axis=0):
    """
    Normalize a N-D numpy array along the specified axis.
    :param nparry: the array of vectors to be normalized
    :param order: order of the norm
    :param axis: the axis of x along which to compute the vector norms
    """
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)

def similarity(array1, array2):
    """
    Compute the euclidean or cosine distance of all pairs.
    :param  array1: numpy array with shape [m1, n]
    :param  array2: numpy array with shape [m2, n]
    Returns:
      numpy array with shape [m1, m2]
    """
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist

def topk(query_feat, gallery_feat):
    """
    Return the index of the top gallery image most similar to the query image
    :param query_feat: array of feature vectors of query images
    :param gallery_feat: array of feature vectors of gallery images
    """
    sim = similarity(query_feat, gallery_feat)
    index = np.argmax(sim, axis=1)[0]
    return index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to run human parsing using JPPNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--query', '-q', help='Path to target image. Skip this argument to select target in the video frame.')
    parser.add_argument('--video', '-g', required=True, help='Path to video file.')
    parser.add_argument('--yolo', required=True, help='Path to yolov8.onnx.')
    parser.add_argument('--resize_h', default = 256, help='The height of the input for model inference.')
    parser.add_argument('--resize_w', default = 128, help='The width of the input for model inference')
    parser.add_argument('--model', '-m', default='reid.onnx', help='Path to pb model.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA backend"% backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: NCS2 VPU, '
                             '%d: HDDL VPU, '
                             '%d: Vulkan, '
                             '%d: CUDA, '
                             '%d: CUDA FP16'
                             % targets)
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model):
        raise OSError("Model not exist")

    extract_frames(args.query, args.video, args.model, args.yolo, args.resize_h, args.resize_w, args.backend, args.target)