'''
    Text detection model: https://github.com/argman/EAST
    Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

    CRNN Text recognition model taken from here: https://github.com/meijieru/crnn.pytorch
    How to convert from pb to onnx:
    Using classes from here: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py

    More converted onnx text recognition models can be downloaded directly here:
    Download link: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
    And these models taken from here:https://github.com/clovaai/deep-text-recognition-benchmark

    import torch
    from models.crnn import CRNN

    model = CRNN(32, 1, 37, 256)
    model.load_state_dict(torch.load('crnn.pth'))
    dummy_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)
'''


# Import required modules
import numpy as np
import cv2
import math
import argparse

# ############ Add argument parser for command line arguments ############
# parser = argparse.ArgumentParser(
#     description="Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
#                 "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)"
#                 "The OCR model can be obtained from converting the pretrained CRNN model to .onnx format from the github repository https://github.com/meijieru/crnn.pytorch"
#                 "Or you can download trained OCR model directly from https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing")
# parser.add_argument('--input',
#                     help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# parser.add_argument('--model', '-m', required=True,
#                     help='Path to a binary .pb file contains trained detector network.')
# parser.add_argument('--ocr', default="crnn.onnx",
#                     help="Path to a binary .pb or .onnx file contains trained recognition network", )
# parser.add_argument('--width', type=int, default=320,
#                     help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
# parser.add_argument('--height', type=int, default=320,
#                     help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
# parser.add_argument('--thr', type=float, default=0.5,
#                     help='Confidence threshold.')
# parser.add_argument('--nms', type=float, default=0.4,
#                     help='Non-maximum suppression threshold.')
# args = parser.parse_args()


def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

def approximate_to_rectangle(points):
    contour = np.array(points).reshape((-1, 1, 2)).astype(np.float32)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(np.intp(box), dtype="float32")  # Updated dtype to avoid deprecation warning
    return box


def main():
    vocPath = "/media/pincambv/hugedrive1/opencv5_gursimar/opencv/samples/data/alphabet_36.txt"
    recModelPath = "/media/pincambv/hugedrive1/Opencv_required_models/ResNet_CTC.onnx"
    detModelPath = "/media/pincambv/hugedrive1/Opencv_required_models/DB_IC15_resnet50.onnx"
    thr = 0.5
    nms = 0.4
    binThresh = 0.3
    polyThresh = 0.5
    maxCandidates = 200
    unclipRatio = 2.0
    inputFile = "/media/pincambv/hugedrive1/opencv5_gursimar/opencv/samples/data/right.jpg"
    model = "DB"
    imreadRGB=False
    width = 736
    height = 736
    mean = [122.67891434, 116.66876762, 104.00698793]

    frame = cv2.imread(inputFile)

    if(model == "DB"):
        detector = cv2.dnn_TextDetectionModel_DB(detModelPath)

        # Set various parameters for the detector
        detector.setBinaryThreshold(binThresh)
        detector.setPolygonThreshold(polyThresh)
        detector.setUnclipRatio(unclipRatio)
        detector.setMaxCandidates(maxCandidates)

        # Set input parameters specific to the DB model
        detector.setInputParams(scale=1.0 / 255, size=(width, height), mean=mean)

        # Perform text detection
        detResults = detector.detect(frame)

    # Open the vocabulary file and read lines into a list
    with open(vocPath, 'r') as voc_file:
        vocabulary = [line.strip() for line in voc_file]

    # Initialize the text recognition model with the specified model path
    recognizer = cv2.dnn_TextRecognitionModel(recModelPath)

    # Set the vocabulary for the model
    recognizer.setVocabulary(vocabulary)

    # Set the decoding method to 'CTC-greedy'
    recognizer.setDecodeType("CTC-greedy")

    recScale = 1.0 / 127.5
    recMean = (127.5, 127.5, 127.5)
    recInputSize = (100, 32)
    recognizer.setInputParams(scale=recScale, size=recInputSize, mean=recMean)

    if len(detResults) > 0:
        recInput = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if not imreadRGB else frame.copy()
        contours = []

        for i, (detection, probability) in enumerate(zip(detResults[0], detResults[1])):
            quadrangle = detection

            if isinstance(quadrangle, np.ndarray):
                quadrangle = approximate_to_rectangle(quadrangle)
                if quadrangle is None or len(quadrangle) != 4:
                    print("Skipping a quadrangle with incorrect points or transformation failed.")
                    continue

                contours.append(np.array(quadrangle, dtype=np.int32))
                cropped = fourPointsTransform(recInput, quadrangle)
                recognitionResult = recognizer.recognize(cropped)
                print(f"{i}: '{recognitionResult}'")

                try:
                    text_origin = (int(quadrangle[3][0]), int(quadrangle[3][1]))
                    cv2.putText(frame, recognitionResult, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except Exception as e:
                    print("Failed to write text on the frame:", e)
            else:
                print("Skipping a detection with invalid format:", quadrangle)

        cv2.polylines(frame, contours, True, (0, 255, 0), 2)
    else:
        print("No Text Detected.")

    cv2.imshow("Text Detection and Recognition", frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
