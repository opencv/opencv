#!/usr/bin/env python
'''
Follow several people at once with cv.MultiTracker.

A detector finds people in each frame and the tracker decides which box belongs to which person,
so every box keeps its number for as long as that person is on screen.

Copyright (C) 2026, BigVision LLC.

How to use:
    Download the models once:
        python download_models.py multi_object_tracker

    Then run:
        python multi_object_tracker.py --input=vtest.avi
        python multi_object_tracker.py --input=vtest.avi --reid

    --reid also loads a re-identification network, which helps ids survive when people cross.

    Set OPENCV_DOWNLOAD_CACHE_DIR to where the models were downloaded, and
    OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.
'''
import argparse
import os.path
import numpy as np
import cv2 as cv

from common import *

def get_args_parser():
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', '-i', default='vtest.avi', help='Path to video file or camera index.')
    parser.add_argument('--reid', action='store_true',
                        help='Also use appearance, so ids survive when people cross.')
    parser.add_argument('--conf', type=float, default=0.25, help='Detector confidence threshold.')
    parser.add_argument('--nms', type=float, default=0.45, help='Detector NMS threshold.')
    parser.add_argument('--backend', default="default", type=str, choices=backends,
                        help="Choose one of computation backends.")
    parser.add_argument('--target', default="cpu", type=str, choices=targets,
                        help="Choose one of target computation devices.")
    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'multi_object_tracker', prefix="", alias="multi_object_tracker")
    add_preproc_args(args.zoo, parser, 'multi_object_tracker', prefix="yolo_", alias="multi_object_tracker")
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='Multi-object tracking using OpenCV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args()


def color_for_id(track_id):
    '''One colour per id, so a person keeps the same colour while they are tracked.'''
    rng = np.random.RandomState(track_id * 7919 % 2**31)
    return tuple(int(c) for c in rng.randint(64, 256, 3))


def detect(frame, net, size, scale, swap_rb, conf_threshold, nms_threshold):
    '''
    yolov8 emits one [1, 84, 8400] tensor: four box numbers then one score per class, for every
    candidate. The frame is padded to a square first so nothing is distorted, which is why the
    boxes are scaled back by the padded length afterwards.
    '''
    height, width = frame.shape[:2]
    length = max(height, width)
    square = np.zeros((length, length, 3), dtype=frame.dtype)
    square[:height, :width] = frame
    back_scale = length / float(size)

    blob = cv.dnn.blobFromImage(square, scale, (size, size), swapRB=swap_rb, crop=False)
    net.setInput(blob)
    out = net.forward()
    candidates = out.reshape(out.shape[1], -1).T        # [8400, 84]

    boxes, scores = [], []
    for row in candidates:
        class_scores = row[4:]
        best = int(np.argmax(class_scores))
        if class_scores[best] < conf_threshold or best != 0:   # class 0 is "person" in COCO
            continue
        cx, cy, w, h = row[:4]
        boxes.append([float(cx - w / 2), float(cy - h / 2), float(w), float(h)])
        scores.append(float(class_scores[best]))

    if not boxes:
        return [], [], []

    keep = np.array(cv.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)).flatten()
    out_boxes = [tuple(v * back_scale for v in boxes[i]) for i in keep]
    return out_boxes, [scores[i] for i in keep], [0] * len(keep)


def describe(frame, reid, boxes):
    '''One L2-normalised descriptor per detection, which is what the appearance overload wants.'''
    if len(boxes) == 0:
        return None
    height, width = frame.shape[:2]
    rows = []
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x + w)), min(height, int(y + h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            x0, y0, x1, y1 = 0, 0, min(2, width), min(2, height)
        feature = reid.predict(frame[y0:y1, x0:x1])[0].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        rows.append(feature)
    return np.array(rows, dtype=np.float32)


def main():
    args = get_args_parser()

    yolo = cv.dnn.readNetFromONNX(findModel(args.yolo_model, args.yolo_sha1))
    yolo.setPreferableBackend(get_backend_id(args.backend))
    yolo.setPreferableTarget(get_target_id(args.target))

    reid = None
    if args.reid:
        reid = cv.dnn.Model(cv.dnn.readNetFromONNX(findModel(args.model, args.sha1)))
        # (pixel - mean) * scale, per channel, with the std folded into the scale. swapRB swaps
        # channel 0 and 2 of both, so each is given in the model's own RGB order.
        reid.setInputMean(args.mean)
        reid.setInputScale([args.scale / s for s in args.std])
        reid.setInputSize(args.width, args.height)
        reid.setInputSwapRB(args.rgb)
        reid.setPreferableBackend(get_backend_id(args.backend))
        reid.setPreferableTarget(get_target_id(args.target))

    params = cv.MultiTracker_Params()
    if args.reid:
        params.embeddingWeight = 0.4    # appearance helps, but motion still leads
    tracker = cv.MultiTracker.create(params)

    source = args.input if not str(args.input).isdigit() else int(args.input)
    if isinstance(source, str):
        source = findFile(source)
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        print('Could not open the input', args.input)
        return

    win_name = 'MULTI OBJECT TRACKING'
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    font_face = cv.FontFace("sans")
    std_size, std_weight, std_img_size = 15, 400, 512

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        font_size = max(10, int(std_size * frame.shape[1] / std_img_size))
        font_weight = max(200, int(std_weight * frame.shape[1] / std_img_size))

        boxes, scores, classes = detect(frame, yolo, args.yolo_width, args.yolo_scale,
                                        args.yolo_rgb, args.conf, args.nms)

        if args.reid:
            features = describe(frame, reid, boxes)
            ids, out_boxes, out_classes = tracker.update(boxes, scores, classes, features)
        else:
            ids, out_boxes, out_classes = tracker.update(boxes, scores, classes)

        for track_id, box in zip(ids, out_boxes):
            color = color_for_id(int(track_id))
            x, y, w, h = (int(v) for v in box)
            cv.rectangle(frame, (x, y, w, h), color, 2)

            label = str(track_id)
            badge_w = font_size * len(label) + 10
            cv.rectangle(frame, (x, max(0, y - font_size - 6), badge_w, font_size + 6), color, cv.FILLED)
            # FontFace overload: (img, text, org, color, fface, size, weight)
            cv.putText(frame, label, (x + 4, max(font_size, y - 4)), (0, 0, 0),
                       font_face, font_size, font_weight)

        status = 'tracking %d  (detections %d)%s' % (len(ids), len(boxes),
                                                     '  [re-id on]' if args.reid else '')
        cv.putText(frame, status, (10, font_size + 5), (0, 255, 0),
                   font_face, font_size, font_weight)

        cv.imshow(win_name, frame)
        if cv.waitKey(30) >= 0:
            break


if __name__ == '__main__':
    main()
