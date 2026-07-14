import os
import numpy as np
import cv2 as cv
import argparse
from common import findFile

parser = argparse.ArgumentParser(description='Use this script to run action recognition using 3D ResNet34',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', '-i', help='Path to input video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', required=True, help='Path to model.')
parser.add_argument('--classes', default=findFile('action_recongnition_kinetics.txt'), help='Path to classes list.')

# To get net download original repository https://github.com/kenshohara/video-classification-3d-cnn-pytorch
# For correct ONNX export modify file: video-classification-3d-cnn-pytorch/models/resnet.py
# change
# - def downsample_basic_block(x, planes, stride):
# -     out = F.avg_pool3d(x, kernel_size=1, stride=stride)
# -     zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
# -                              out.size(2), out.size(3),
# -                              out.size(4)).zero_()
# -     if isinstance(out.data, torch.cuda.FloatTensor):
# -         zero_pads = zero_pads.cuda()
# -
# -     out = Variable(torch.cat([out.data, zero_pads], dim=1))
# -     return out

# To
# + def downsample_basic_block(x, planes, stride):
# +     out = F.avg_pool3d(x, kernel_size=1, stride=stride)
# +     out = F.pad(out, (0, 0, 0, 0, 0, 0, 0, int(planes - out.size(1)), 0, 0), "constant", 0)
# +     return out

# To ONNX export use torch.onnx.export(model, inputs, model_name)

def get_class_names(path):
    class_names = []
    with open(path) as f:
        for row in f:
            class_names.append(row[:-1])
    return class_names

def classify_video(video_path, net_path):
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112
    mean = (114.7748, 107.7354, 99.4750)
    class_names = get_class_names(args.classes)

    net = cv.dnn.readNet(net_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    winName = 'Deep learning image classification in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    cap = cv.VideoCapture(video_path)
    while cv.waitKey(1) < 0:
        frames = []
        for _ in range(SAMPLE_DURATION):
            hasFrame, frame = cap.read()
            if not hasFrame:
                exit(0)
            frames.append(frame)

        inputs = cv.dnn.blobFromImages(frames, 1, (SAMPLE_SIZE, SAMPLE_SIZE), mean, True, crop=True)
        inputs = np.transpose(inputs, (1, 0, 2, 3))
        inputs = np.expand_dims(inputs, axis=0)
        net.setInput(inputs)
        outputs = net.forward()
        class_pred = np.argmax(outputs)
        label = class_names[class_pred]

        for frame in frames:
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(frame, (0, 10 - labelSize[1]),
                                (labelSize[0], 10 + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (0, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv.imshow(winName, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    classify_video(args.input if args.input else 0, args.model)
