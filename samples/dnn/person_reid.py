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

        python person_reid.py --query=/path/to/query_image --video=/path/to/video/footage --model=path/to/youtu_reid_baseline_medium.onnx --yolo=/path/to/yolov3.weights --cfg=/path/to/yolov3.cfg

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

img_dict = {} # Dictionary to store bounding boxes for corresponding cropped image

def yolo_detector(frame, net, output_layers):
    global img_dict
    height, width, channels = frame.shape

    # Create blob from the frame
    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Filter to detect only 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to reduce overlapping bounding boxes
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    images = []
    for i in indexes:
        x, y, w, h = boxes[i]

        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        crop_img = frame[y:y+h, x:x+w]
        images.append(crop_img)
        img_dict[crop_img.tobytes()] = (x, y, w, h)
    return images

def extract_frames(query_image_path, video_path, model_path, yolo_path, cfg_path, resize_h = 384, resize_w = 128, backend=cv.dnn.DNN_BACKEND_OPENCV, target=cv.dnn.DNN_TARGET_CPU, batch_size = 32):
    cap = cv.VideoCapture(video_path)

    net = cv.dnn.readNet(yolo_path, cfg_path)
    layer_names = net.getLayerNames()

    # Handle different formats of layer outputs from different OpenCV versions
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    frames = []
    query_images = [cv.imread(query_image_path)]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        images = yolo_detector(frame, net, output_layers)
        query_feat = extract_feature(query_images, model_path, resize_h, resize_w, backend, target, batch_size)
        gallery_feat = extract_feature(images, model_path, resize_h, resize_w, backend, target, batch_size)

        topk_idx = topk(query_feat, gallery_feat)

        query_img = query_images[0]
        top_img = images[topk_idx]
        x, y, w, h = img_dict[top_img.tobytes()]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, "Target", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv.imshow("Image", frame)
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
    parser.add_argument('--query', '-q', required=True, help='Path to target image.')
    parser.add_argument('--video', '-g', required=True, help='Path to video file.')
    parser.add_argument('--yolo', required=True, help='Path to yolov3.weights.')
    parser.add_argument('--cfg', required=True, help='Path to yolov3.cfg.')
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

    extract_frames(args.query, args.video, args.model, args.yolo, args.cfg, args.resize_h, args.resize_w, args.backend, args.target)