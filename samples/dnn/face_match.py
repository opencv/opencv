import argparse

import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--input1', '-i1', type=str, help='Path to the input image1.')
parser.add_argument('--input2', '-i2', type=str, help='Path to the input image2.')
parser.add_argument('--face_detection_model', '-fd', type=str, help='Path to the face detection model. Download the model at https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.')
parser.add_argument('--face_recognition_model', '-fr', type=str, help='Path to the face recognition model. Download the model at https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view.')
args = parser.parse_args()

# Read the input image
img1 = cv.imread(args.input1)
img2 = cv.imread(args.input2)

# Instantiate face detector and recognizer
detector = cv.FaceDetectorYN.create(
    args.face_detection_model,
    "",
    (img1.shape[1], img1.shape[0])
)
recognizer = cv.FaceRecognizerSF.create(
    args.face_recognition_model,
    ""
)

# Detect face
detector.setInputSize((img1.shape[1], img1.shape[0]))
face1 = detector.detect(img1)
detector.setInputSize((img2.shape[1], img2.shape[0]))
face2 = detector.detect(img2)
assert face1[1].shape[0] > 0, 'Cannot find a face in {}'.format(args.input1)
assert face2[1].shape[0] > 0, 'Cannot find a face in {}'.format(args.input2)

# Align faces
face1_align = recognizer.alignCrop(img1, face1[1][0])
face2_align = recognizer.alignCrop(img2, face2[1][0])

# Extract features
face1_feature = recognizer.faceFeature(face1_align)
face2_feature = recognizer.faceFeature(face2_align)

# Calculate distance (0: cosine, 1: L2)
cosine_similarity_threshold = 0.363
cosine_score = recognizer.faceMatch(face1_feature, face2_feature, 0)
msg = 'different identities'
if cosine_score >= cosine_similarity_threshold:
    msg = 'the same identity'
print('They have {}. Cosine Similarity: {}, threshold: {} (higher value means higher similarity, max 1.0).'.format(msg, cosine_score, cosine_similarity_threshold))

l2_similarity_threshold = 1.128
l2_score = recognizer.faceMatch(face1_feature, face2_feature, 1)
msg = 'different identities'
if l2_score <= l2_similarity_threshold:
    msg = 'the same identity'
print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg, l2_score, l2_similarity_threshold))