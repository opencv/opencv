import os
import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Use this script to run action classification using 3D ResNet34')
parser.add_argument('--input', '-i', help='Path to input video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', help='Path to model.')
parser.add_argument('--classes', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'dnn', 'action_classification.txt'), help='Path to classes list.')

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

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Scale(object):
    def __init__(self, size, interpolation=cv.INTER_LINEAR):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            h, w = img.shape[0], img.shape[1]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return cv.resize(img, (ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return cv.resize(img, (ow, oh), self.interpolation)
        else:
            return cv.resize(img, self.size, interpolation=self.interpolation)

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1 + th, x1:x1 + tw]

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = np.transpose(tensor.astype(np.float32), (2, 0, 1))
        for ch in range(0, tensor.shape[0]):
            tensor[ch] -= self.mean[ch]
            tensor[ch] /= self.std[ch]
        return tensor

class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)
        return out

def classify_video(video_path, net_path):
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112
    video = []
    frame_indices = []

    class_names = get_class_names(args.classes)
    mean = (114.7748, 107.7354, 99.4750)
    temporal_transform = LoopPadding(SAMPLE_DURATION)
    spatial_transform = Compose([
                                Scale(SAMPLE_SIZE),
                                CenterCrop(SAMPLE_SIZE),
                                Normalize(mean, [1, 1, 1]),
                                ])

    net = cv.dnn.readNet(net_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    left = 0
    top = 10
    winName = 'Deep learning action classification in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    cap = cv.VideoCapture(video_path)
    i = 0
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if hasFrame:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            video.append(frame)
            frame_indices.append(i)
            i += 1
            if len(video) == SAMPLE_DURATION:
                frame_indices = temporal_transform(frame_indices)
                clip = [spatial_transform(img) for img in video]
                inputs = np.stack(clip, axis=0)
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np.transpose(inputs, (0, 2, 1, 3, 4))

                net.setInput(inputs)
                outputs = net.forward()
                class_pred = np.argmax(outputs)
                label = class_names[class_pred]

                for frame in video:
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv.rectangle(frame, (left, top - labelSize[1]),
                                        (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    cv.imshow(winName, frame)
                video = []
                frame_indices = []
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    classify_video(args.input if args.input else 0, args.model)
