import argparse

import numpy as np
import cv2 as cv

# e.g. python3 face_detector.py --image ./image.jpg --face_detection_model ./face_detection_yunet_2021dec.onnx
parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2021dec.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
args = parser.parse_args()

def visualize(image, confidences, boxes, landmarks, fps, thickness=2):
    fps_string = "{:.2g}".format(fps)
    print(fps_string)
    cv.putText(image, fps_string, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    num_faces = len(boxes)
    for i in range(num_faces):
        confidence = confidences[i]
        box = boxes[i]
        landmark = landmarks[i]

        result = "face{} : ( {}, {} ), {} x {}, {:.2g}".format(i, box[0], box[1], box[2], box[3], confidence)
        print(result)

        # Draw Bounding Box
        color = (0, 255, 0)
        cv.rectangle(image, box, color, thickness)

        # Draw Landmarks
        radius = 2
        for point in landmark:
            cv.circle(image, point, radius, color, thickness)

if __name__ == '__main__':
    scale = args.scale
    score_threshold = args.score_threshold
    nms_threshold = args.nms_threshold

    ## [Create_FaceDetectionModel_YN]
    face_detection_model_path = args.face_detection_model
    face_detector = cv.dnn_FaceDetectionModel_YN(face_detection_model_path)
    ## [Create_FaceDetectionModel_YN]

    tm = cv.TickMeter()

    if args.image is not None:
        image_path = args.image
        image = cv.imread(cv.samples.findFile(image_path))
        if image is None:
            raise Exception("can't read image: {}".format(image_path))

        image = cv.resize(image, dsize=None, fx=scale, fy=scale)

        tm.reset()
        tm.start()

        ## [Face_Detection]
        # Detect faces from image.
        confidences, boxes = face_detector.detect(image, score_threshold, nms_threshold)
        ## [Face_Detection]

        ## [Face_Landmarks]
        # Get face landmarks of each faces.
        # If the face detection model have the feature to detect face landmarks, you can get face landmarks using getLandmarks().
        # The number and order of face landmarks varies by model.
        # In the case of YuNet, you can get positions of Right-Eye, Left-Eye, Nose, Right-Mouth Corner, and Right-Mouth Corner.
        landmarks = face_detector.getLandmarks()
        ## [Face_Landmarks]

        tm.stop()

        fps = tm.getFPS()
        visualize(image, confidences, boxes, landmarks, fps)

        cv.imshow("face detector", image)
        cv.waitKey(0)

    elif args.video is not None:
        video_path = args.video
        if len(video_path) == 1 and video_path.isdigit():
            capture = cv.VideoCapture(int(video_path))
        else:
            capture = cv.VideoCapture(cv.samples.findFile(video_path))

        if not capture.isOpened():
            raise Exception("can't open video: {}".format(video_path))

        while (True):
            result, frame = capture.read()
            if not result:
                print("can't grab frame!")
                break

            frame = cv.resize(frame, dsize=None, fx=scale, fy=scale)

            tm.start()

            confidences, boxes = face_detector.detect(frame, score_threshold, nms_threshold)

            landmarks = face_detector.getLandmarks()

            tm.stop()

            fps = tm.getFPS()
            visualize(frame, confidences, boxes, landmarks, fps)

            cv.imshow("face detector", frame)
            key = cv.waitKey(1)
            if key != -1:
                break
