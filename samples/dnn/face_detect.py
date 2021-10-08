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
parser.add_argument('--input', '-i', type=str, help='Path to the input image.')
parser.add_argument('--model', '-m', type=str, default='yunet.onnx', help='Path to the model. Download the model at https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(input, faces, thickness=2):
    output = input.copy()
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
            cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
            cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
            cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
            cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
            cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
    return output

if __name__ == '__main__':

    # Instantiate FaceDetectorYN
    detector = cv.FaceDetectorYN.create(
        args.model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Set input size before inference
        detector.setInputSize((image.shape[1], image.shape[0]))

        # Inference
        faces = detector.detect(image)

        # Draw results on the input image
        result = visualize(image, faces)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg\n')
            cv.imwrite('result.jpg', result)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, result)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frameWidth, frameHeight])

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            faces = detector.detect(frame) # faces is a tuple
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, faces)

            cv.putText(frame, 'FPS: {}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # Visualize results in a new Window
            cv.imshow('Live', frame)

            tm.reset()