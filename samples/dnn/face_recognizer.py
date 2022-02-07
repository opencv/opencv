import argparse

import numpy as np
import cv2 as cv

COSINE_THRESHOLD = 0.363
L2NORM_THRESHOLD = 1.128

# e.g. std::cout << "e.g. python3 face_recognizer.py --image1 ./image1.jpg --image2 ./image2.jpg --face_detection_model ./face_detection_yunet_2021dec.onnx -face_recognition_model ./face_recognition_sface_2021dec.onnx" << std::endl;
parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2021dec.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
args = parser.parse_args()

def visualize(image1, box1, image2, box2, score, distance_type=cv.dnn.FaceRecognitionModel_FR_COSINE, thickness=2):
    # Green is same identity, Red is different identities.
    if distance_type == 0:
        same_identity = (score >= COSINE_THRESHOLD)
    else:
        same_identity = (score <= L2NORM_THRESHOLD)
    color = (0, 255, 0) if same_identity else (255, 0, 0)

    # Draw Bounding Box
    cv.rectangle(image1, box1, color, thickness)
    cv.rectangle(image2, box2, color, thickness)

if __name__ == '__main__':
    if (args.image1 is None) or (args.image2 is None):
        raise Exception("can't specify image paths")

    scale = args.scale
    score_threshold = args.score_threshold
    nms_threshold = args.nms_threshold

    face_detection_model_path = args.face_detection_model
    print(face_detection_model_path)
    face_detector = cv.dnn_FaceDetectionModel_YN(face_detection_model_path)

    ## [Create_FaceRecognitionModel_SF]
    # Create FaceRecognitionModel_SF
    face_recognition_model_path = args.face_recognition_model
    face_recognizer = cv.dnn_FaceRecognitionModel_SF(face_recognition_model_path)
    ## [Create_FaceRecognitionModel_SF]

    image1_path = args.image1
    image1 = cv.imread(cv.samples.findFile(image1_path))
    if image1 is None:
        raise Exception("can't read image1: {}".format(image1_path))

    image1 = cv.resize(image1, dsize=None, fx=scale, fy=scale)

    _, boxes1 = face_detector.detect(image1, score_threshold, nms_threshold)
    landmarks1 = face_detector.getLandmarks()
    if len(landmarks1) == 0:
        raise Exception("can't detect face from image1")

    image2_path = args.image2
    image2 = cv.imread(cv.samples.findFile(image2_path))
    if image2 is None:
        raise Exception("can't read image2: {}".format(image2_path))

    image2= cv.resize(image2, dsize=None, fx=scale, fy=scale)

    _, boxes2 = face_detector.detect(image2, score_threshold, nms_threshold)
    landmarks2 = face_detector.getLandmarks()
    if len(landmarks2) == 0:
        raise Exception("can't detect face from image2")

    ## [Align_Crop]
    # Aligning and Cropping facial image through the first face of faces detected.
    # In the case of SFace, It use five landmarks that lined up in a specific order.
    # (Right-Eye, Left-Eye, Nose, Right-Mouth Corner, Right-Mouth Corner)
    aligned_face1 = face_recognizer.alignCrop(image1, landmarks1[0])
    aligned_face2 = face_recognizer.alignCrop(image2, landmarks2[0])
    ## [Align_Crop]

    ## [Extract_Feature]
    # Run feature extraction with given aligned_face.
    face_feature1 = face_recognizer.feature(aligned_face1)
    face_feature2 = face_recognizer.feature(aligned_face2)
    ## [Extract_Feature]

    ## [Match_Features]
    # Match two features using each distance types.
    # * DisType::FR_COSINE : Cosine similarity. Higher value means higher similarity. (max 1.0)
    # * DisType::FR_NORM_L2 : L2-Norm distance. Lower value means higher similarity. (min 0.0)
    cos_score = face_recognizer.match(face_feature1, face_feature2, cv.dnn.FaceRecognitionModel_FR_COSINE)
    l2norm_score = face_recognizer.match(face_feature1, face_feature2, cv.dnn.FaceRecognitionModel_NORM_L2)
    ## [Match_Features]

    ## [Check_Identity]
    # Check identity using Cosine similarity.
    if (cos_score >= COSINE_THRESHOLD):
        print("They have the same identity;", end="")
    else:
        print("They have different identities;", end="")
    print(" Cosine Similarity: {}, threshold: {}. (higher value means higher similarity, max 1.0)".format(cos_score, COSINE_THRESHOLD));

    # Check identity using L2-Norm distance.
    if (l2norm_score <= L2NORM_THRESHOLD):
        print("They have the same identity;", end="")
    else:
        print("They have different identities;", end="")
    print(" L2-Norm Distance: {}, threshold: {}. (lower value means higher similarity, min 0.0)".format(l2norm_score, L2NORM_THRESHOLD))
    ## [Check_Identity]

    visualize(image1, boxes1[0], image2, boxes2[0], cos_score)

    cv.imshow("face recognizer (image1)", image1)
    cv.imshow("face recognizer (image2)", image2)
    cv.waitKey(0);
