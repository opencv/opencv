import os
import copy
import shutil
import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--input', help='Path to input video file.')
parser.add_argument('--model', help='Path to model.')
parser.add_argument('--classes', help='Path to classes list.')
parser.add_argument('--temporal_unit', default=5, help='Averages the scores over temporal_unit x sample_duration clips.')
parser.add_argument('--sample_duration', default=16, help='Number of frames for action detection.')

# To get net download original repository https://github.com/kenshohara/video-classification-3d-cnn-pytorch.git
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


def parse_video(video_path):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
    os.mkdir('tmp')
    i = 0
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if hasFrame:
            cv.imwrite('tmp/image_{:05}.jpg'.format(i), frame)
            i += 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    return fps

def get_class_names(path):
    class_names = []
    with open(path) as f:
        for row in f:
            class_names.append(row[:-1])
    return class_names

def make_dataset(video_path, sample_duration):
    dataset = []
    n_frames = len(os.listdir(video_path))
    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }
    for i in range(1, (n_frames - sample_duration + 1), sample_duration):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = np.array([i, i + sample_duration - 1])
        dataset.append(sample_i)
    return dataset


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

def video_loader(video_dir_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            img = cv.imread(image_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            video.append(img)
    return video

class Video(object):
    def __init__(self, video_path,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16):
        self.data = make_dataset(video_path, sample_duration)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, index):
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        clip = video_loader(path, frame_indices)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = np.stack(clip, axis=0)
        target = self.data[index]['segment']
        return clip, target
    
    def __len__(self):
        return len(self.data)


def prepare_input(video_dir, sample_duration=16, sample_size=112):
    mean = (114.7748, 107.7354, 99.4750)
    temporal_transform = LoopPadding(sample_duration)
    spatial_transform = Compose([
                                Scale(sample_size),
                                CenterCrop(sample_size),
                                Normalize(mean, [1, 1, 1]),
                                ])
    data = Video(video_dir, spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                sample_duration=sample_duration)

    data_loader = [elem for elem in data]
    return data_loader


def classify_video(video_dir, net_path, classes_path, sample_duration=16, sample_size=112):
    data_loader = prepare_input(video_dir, sample_duration, sample_size)
    class_names = get_class_names(classes_path)
    video_outputs = []
    video_segments = []

    net = cv.dnn.readNet(net_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    for i, (inputs, segments) in enumerate(data_loader):
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, (0, 2, 1, 3, 4))    
        net.setInput(inputs)
        outputs = net.forward()
        video_outputs.append(outputs)
        video_segments.append(segments)

    video_outputs = np.concatenate(video_outputs)

    max_indices = [np.argmax(elem) for elem in video_outputs]
    results = []
    for i, out in enumerate(video_outputs):
        clip_results = {
            'segment': video_segments[i].tolist()
            }
        clip_results['label'] = class_names[max_indices[i]]
        clip_results['scores'] = out.tolist()
        results.append(clip_results)
    return results

def postprocess(clips, temporal_unit):
    class_names = get_class_names(args.classes)
    unit_classes = []
    unit_segments = []
    unit = int(temporal_unit) if temporal_unit != 0 else len(clips)

    for i in range(0, len(clips), unit):
        n_elements = min(unit, len(clips) - i)
        scores = np.array(clips[i]['scores'])
        for j in range(i, min(i + unit, len(clips))):
            scores += np.array(clips[i]['scores'])
        scores /= n_elements
        unit_classes.append(class_names[np.argmax(scores)])
        unit_segments.append([clips[i]['segment'][0],
                                clips[i + n_elements - 1]['segment'][1]])

    for i in range(len(unit_classes)):
        for j in range(unit_segments[i][0], unit_segments[i][1] + 1):
            frame = cv.imread('tmp/image_{:05}.jpg'.format(j))
            label = unit_classes[i]
            left = 0
            top = 10
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(frame, (left, top - labelSize[1]),
                                (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv.imwrite('tmp/image_{:05}.jpg'.format(j), frame)


def generate_video(name='action_recognition_opencv.avi', fps=30.0):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    if os.path.exists('tmp/image_00001.jpg'):
        img = cv.imread('tmp/image_00001.jpg')
        h, w = img.shape[0], img.shape[1]
    video = cv.VideoWriter(name, fourcc, fps, (w, h))
    for i in range(1, len(os.listdir('tmp'))):
        img = cv.imread('tmp/image_{:05}.jpg'.format(i))
        video.write(img)
    video.release()
    shutil.rmtree('tmp')


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    fps = parse_video(args.input if args.input else 0)
    predictions = classify_video('tmp', args.model, args.classes, args.sample_duration)
    postprocess(predictions, args.temporal_unit)
    generate_video(fps=fps)