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
Copyright (C) 2024, Bigvision LLC.

How to use:
    sample command to run:
        `python person_reid.py`

    You can download ReID model using
        `python download_models.py reid`
    and yolo model using:
        `python download_models.py yolov8`

    Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.
'''
import argparse
import os.path
import numpy as np
import cv2 as cv
from common import *

def help():
    print(
        '''
        Use this script for Person Re-identification using OpenCV.

        Firstly, download required models i.e. reid and yolov8 using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.

        To run:
        Example: python person_reid.py reid

        Re-identification model path can also be specified using --model argument and detection model can be specified using --yolo_model argument.
        '''
    )

def get_args_parser():
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--query', '-q', help='Path to target image. Skip this argument to select target in the video frame.')
    parser.add_argument('--input', '-i', default=0, help='Path to video file.', required=False)
    parser.add_argument('--backend', default="default", type=str, choices=backends,
            help="Choose one of computation backends: "
            "default: automatically (by default), "
            "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
            "opencv: OpenCV implementation, "
            "vkcom: VKCOM, "
            "cuda: CUDA, "
            "webnn: WebNN")
    parser.add_argument('--target', default="cpu", type=str, choices=targets,
            help="Choose one of target computation devices: "
            "cpu: CPU target (by default), "
            "opencl: OpenCL, "
            "opencl_fp16: OpenCL fp16 (half-float precision), "
            "ncs2_vpu: NCS2 VPU, "
            "hddl_vpu: HDDL VPU, "
            "vulkan: Vulkan, "
            "cuda: CUDA, "
            "cuda_fp16: CUDA fp16 (half-float preprocess)")
    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'person_reid', prefix="", alias="reid")
    add_preproc_args(args.zoo, parser, 'person_reid', prefix="yolo_", alias="reid")
    parser = argparse.ArgumentParser(parents=[parser],
                                        description='Person Re-identification using OpenCV.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args()

img_dict = {} # Dictionary to store bounding boxes for corresponding cropped image

def yolo_detector(frame, net):
    global img_dict
    height, width, _ = frame.shape

    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame

    scale = length/args.yolo_width
    # Create blob from the frame with correct scale factor and size for the model

    blob = cv.dnn.blobFromImage(image, scalefactor=args.yolo_scale, size=(args.yolo_width, args.yolo_height), swapRB=args.yolo_rgb)
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

def extract_feature(images, net):
    """
    Extract features from images
    :param images: the input images
    :param net: the model network
    """
    feat_list = []
    # net = reid_net.copy()
    for img in images:
        blob = cv.dnn.blobFromImage(img, scalefactor=args.scale, size=(args.width, args.height), mean=args.mean, swapRB=args.rgb, crop=False, ddepth=cv.CV_32F)

        for j in range(blob.shape[1]):
            blob[:, j, :, :] /= args.std[j]

        net.setInput(blob)
        feat = net.forward()
        feat = np.reshape(feat, (feat.shape[0], feat.shape[1]))
        feat_list.append(feat)

    feats = np.concatenate(feat_list, axis = 0)
    return feats

def find_matching(query_feat, gallery_feat):
    """
    Return the index of the gallery image most similar to the query image
    :param query_feat: array of feature vectors of query images
    :param gallery_feat: array of feature vectors of gallery images
    """
    cv.normalize(query_feat, query_feat, 1.0, 0.0, cv.NORM_L2)
    cv.normalize(gallery_feat, gallery_feat, 1.0, 0.0, cv.NORM_L2)

    sim = query_feat.dot(gallery_feat.T)
    index = np.argmax(sim, axis=1)[0]
    return index

def main():
    if hasattr(args, 'help'):
        help()
        exit(1)

    args.model = findModel(args.model, args.sha1)

    if args.yolo_model is None:
        print("[ERROR] Please pass path to yolov8.onnx model file using --yolo_model.")
        exit(1)
    else:
        args.yolo_model = findModel(args.yolo_model, args.yolo_sha1)

    engine = cv.dnn.ENGINE_AUTO

    if args.backend != "default" or args.target != "cpu":
        engine = cv.dnn.ENGINE_CLASSIC
    yolo_net = cv.dnn.readNetFromONNX(args.yolo_model, engine)
    reid_net = cv.dnn.readNetFromONNX(args.model, engine)
    reid_net.setPreferableBackend(get_backend_id(args.backend))
    reid_net.setPreferableTarget(get_target_id(args.target))
    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    query_images = []

    stdSize = 0.6
    stdWeight = 2
    stdImgSize = 512
    imgWidth = -1 # Initialization
    fontSize = 1.5
    fontThickness = 1

    if args.query:
        query_images = [cv.imread(findFile(args.query))]
    else:
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
                rect = cv.selectROI("TRACKING", image)
                if rect:
                    x, y, w, h = rect
                    query_image = image[y:y + h, x:x + w]
                    query_images = [query_image]
                    break

            if key == ord('q') or key == 27:
                return

    query_feat = extract_feature(query_images, reid_net)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if imgWidth == -1:
            imgWidth = min(frame.shape[:2])
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize)
            fontThickness = max(fontThickness,(stdWeight*imgWidth)//stdImgSize)

        images = yolo_detector(frame, yolo_net)
        gallery_feat = extract_feature(images, reid_net)

        match_idx = find_matching(query_feat, gallery_feat)

        match_img = images[match_idx]
        x, y, w, h = img_dict[match_img.tobytes()]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, "Target", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), fontThickness)

        label="Tracking"
        labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
        cv.rectangle(frame, (0, 0), (labelSize[0]+10, labelSize[1]+10), (255,255,255), cv.FILLED)
        cv.putText(frame, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.imshow("TRACKING", frame)
        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv.destroyAllWindows()
    return

if __name__ == '__main__':
    args = get_args_parser()
    main()
