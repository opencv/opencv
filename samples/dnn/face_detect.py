import argparse

import numpy as np
import cv2 as cv

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2021dec.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':

    ## [initialize_FaceDetectorYN]
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    ## [initialize_FaceDetectorYN]

    tm = cv.TickMeter()

    # If input is an image
    if args.image1 is not None:
        img1 = cv.imread(cv.samples.findFile(args.image1))
        img1Width = int(img1.shape[1]*args.scale)
        img1Height = int(img1.shape[0]*args.scale)

        img1 = cv.resize(img1, (img1Width, img1Height))
        tm.start()

        ## [inference]
        # Set input size before inference
        detector.setInputSize((img1Width, img1Height))

        faces1 = detector.detect(img1)
        ## [inference]

        tm.stop()
        assert faces1[1] is not None, 'Cannot find a face in {}'.format(args.image1)

        # Draw results on the input image
        visualize(img1, faces1, tm.getFPS())

        # Save results if save is true
        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', img1)

        # Visualize results in a new window
        cv.imshow("image1", img1)

        if args.image2 is not None:
            img2 = cv.imread(cv.samples.findFile(args.image2))

            tm.reset()
            tm.start()
            detector.setInputSize((img2.shape[1], img2.shape[0]))
            faces2 = detector.detect(img2)
            tm.stop()
            assert faces2[1] is not None, 'Cannot find a face in {}'.format(args.image2)
            visualize(img2, faces2, tm.getFPS())
            cv.imshow("image2", img2)

            ## [initialize_FaceRecognizerSF]
            recognizer = cv.FaceRecognizerSF.create(
            args.face_recognition_model,"")
            ## [initialize_FaceRecognizerSF]

            ## [facerecognizer]
            # Align faces
            face1_align = recognizer.alignCrop(img1, faces1[1][0])
            face2_align = recognizer.alignCrop(img2, faces2[1][0])

            # Extract features
            face1_feature = recognizer.feature(face1_align)
            face2_feature = recognizer.feature(face2_align)
            ## [facerecognizer]

            cosine_similarity_threshold = 0.363
            l2_similarity_threshold = 1.128

            ## [match]
            cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
            l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)
            ## [match]

            msg = 'different identities'
            if cosine_score >= cosine_similarity_threshold:
                msg = 'the same identity'
            print('They have {}. Cosine Similarity: {}, threshold: {} (higher value means higher similarity, max 1.0).'.format(msg, cosine_score, cosine_similarity_threshold))

            msg = 'different identities'
            if l2_score <= l2_similarity_threshold:
                msg = 'the same identity'
            print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg, l2_score, l2_similarity_threshold))
        cv.waitKey(0)
    else: # Omit input to call default camera
        if args.video is not None:
            deviceId = args.video
        else:
            deviceId = 0
        cap = cv.VideoCapture(deviceId)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
        detector.setInputSize([frameWidth, frameHeight])

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame = cv.resize(frame, (frameWidth, frameHeight))

            # Inference
            tm.start()
            faces = detector.detect(frame) # faces is a tuple
            tm.stop()

            # Draw results on the input image
            visualize(frame, faces, tm.getFPS())

            # Visualize results
            cv.imshow('Live', frame)
    cv.destroyAllWindows()
