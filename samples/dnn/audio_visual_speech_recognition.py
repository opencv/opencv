import argparse

import numpy as np
import cv2 as cv

'''
AVSpeechRecognition

    How to obtain the model required for this sample:
    Option 1: Download the model from https://drive.google.com/file/d/1xuwk5ZQagKFoTXev27zvlSg8dAmSJoo7/view?usp=sharing
    Option 2: Convert the model using pretrained torch model and base repo.
        Use the code here: https://gist.github.com/spazewalker/b8ab3eabc96ffcb30218cbb6f6ea09b3

For preporocessing the video, YUNet face detection is also required. YUNet can be downloaded from:
https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx
'''

class AVSpeechRecognition:
    '''
    Audio Video Speech Recognition based on AVHubert (arXiv:2201.02184 [eess.AS])
    '''
    def __init__(self, source, detector_path, model_path, margin, video_width, video_height,
                score_threshold, nms_threshold, top_k, backend, show_video=False, labels=None):
        '''
        params:
            source: video source
            detector_path: face detection model path
            model_path: speech recognition model path
            margin: margin for temporal window
            video_width: video width
            video_height: video height
            score_threshold: score threshold for face detection
            nms_threshold: nms threshold for face detection
            top_k: top k faces for face detection
            backend: backend for model inference
            show_video: show video or not
            labels: labels for speech recognition
        '''
        self.labels = open(labels).read().strip().split('\n') if labels else None
        source = source if source else 0
        self.cap = cv.VideoCapture(source)
        self.samplingRate = 16000
        self.fps = 30
        self.source = source
        self.realtime = True if source == 0 else False
        self.params = np.asarray([cv.CAP_PROP_AUDIO_STREAM, 0,
                cv.CAP_PROP_VIDEO_STREAM, 0,
                cv.CAP_PROP_AUDIO_DATA_DEPTH, cv.CV_32F,
                cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND, self.samplingRate
                ])
        self.margin = margin
        self.height = video_height
        self.width = video_width
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, video_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, video_height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)

        self.model = cv.dnn.readNetFromONNX(model_path)
        self.detector = cv.FaceDetectorYN.create(detector_path, "", (video_width, video_height), score_threshold, nms_threshold, top_k)
        self.model.setPreferableBackend(backend)
        self.model.enableFusion(False)

        self.landmarks_queue = []
        self.frames_queue = []
        self.audio_queue = []

        self.show_video = show_video

    def warp_image(self, frame, smoothed_landmarks):
        '''
        warps frame to make lips horizontal and fixed at center
        params:
            frame: input frame
            smoothed_landmarks: smoothed landmarks
        return:
            trans_frame: warped frame
            trans_landmarks: warped landmarks
        '''
        rotateby = np.arctan((smoothed_landmarks[6][1]-smoothed_landmarks[5][1])/(smoothed_landmarks[6][0]-smoothed_landmarks[5][0]))*180/np.pi
        image_center = tuple((smoothed_landmarks[0]+smoothed_landmarks[1])/2)
        rot_mat = cv.getRotationMatrix2D(image_center, rotateby, 2)
        trans_frame = cv.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv.INTER_LINEAR)
        trans_landmarks = np.hstack((smoothed_landmarks, np.ones(shape=(7,1))))@rot_mat.T
        return trans_frame, trans_landmarks

    def cut_patch(self, img, landmarks, height, width, threshold=5):
        '''
        cuts mouth roi from image based on the mouth landmarks
        params:
            img: input image
            landmarks: mouth landmarks
            height: height of patch
            width: width of patch
            threshold: threshold for cutting (default:5)
        return:
            cutted_img: cutted image
        '''
        center_x, center_y = np.mean(landmarks, axis=0)
        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')

        if center_y + height > img.shape[0]:
            center_y = img.shape[0] - height
        if center_y + height > img.shape[0] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[1]:
            center_x = img.shape[1] - width
        if center_x + width > img.shape[1] + threshold:
            raise Exception('too much bias in width')

        cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                            int(round(center_x) - round(width)): int(round(center_x) + round(width))])
        return cutted_img

    def preprocess(self, frame, audio):
        '''
        preprocesses frame to get landmarks and mouth rois
        params:
            frame: input frame
            audio: input audio
        return:
            cropped: mouth roi
            normlized_audio: normalized audio
        '''
        landmarks = self.detector.detect(frame)[-1]
        cropped = None
        if landmarks is not None:
            landmarks = landmarks[:,:-1].reshape(landmarks.shape[0],7,2)
            if len(landmarks) == 0:
                return None, None
            self.landmarks_queue.append(landmarks)
            if len(self.landmarks_queue) < self.margin:
                return None, None
            smoothed_landmarks = np.mean(self.landmarks_queue, axis=0)[0]
            trans_frame, trans_landmarks = self.warp_image(frame, smoothed_landmarks)
            cropped = self.cut_patch(trans_frame, trans_landmarks[-2:], 96//2,96//2)
        if audio is not None:
            signal_std = np.std(audio)
            signal_mean = np.mean(audio)
            normalized_audio = (audio - signal_mean) / signal_std
        return cropped, normalized_audio

    def predict(self, frames_queue, audio_queue):
        '''
        predicts word using Audio Video Speech Recognition model.
        params:
            frames_queue: queue of frames
            audio_queue: queue of audio
        return:
            pred: predicted word or index
        '''
        video = np.expand_dims(np.array(frames_queue[-self.margin:]), axis=(0,1))
        audio = np.expand_dims(np.array(audio_queue[-self.margin*self.samplingRate//self.fps+1:], dtype=np.float32),axis=(0,1))
        self.model.setInput(video, 'video_input')
        self.model.setInput(audio, 'audio_input')
        out = self.model.forward()
        pred = out.reshape(-1).argmax()
        pred = self.labels[pred] if self.labels else pred
        return pred

    def run(self):
        '''
        Read the video and process it.
        '''
        self.cap.open(self.source, cv.CAP_MSMF, self.params)
        if not self.cap.isOpened():
            print('Cannot open video source')
            exit(1)
        audioBaseIndex = int(self.cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
        audioChannels = int(self.cap.get(cv.CAP_PROP_AUDIO_TOTAL_CHANNELS))

        if self.cap.isOpened():
            while self.cap.grab():
                ret, frame = self.cap.retrieve()
                frame = cv.resize(frame, (self.width, self.height))
                audioFrame = np.asarray([])
                audioFrame = self.cap.retrieve(audioFrame, audioBaseIndex)
                audioFrame = audioFrame[1][0] if audioFrame is not None else None

                if self.realtime:
                    img, aud = self.preprocess(frame, audioFrame)
                    if img is not None and aud is not None:
                        img = cv.resize(img, (96,96))
                        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        self.frames_queue.append(img)
                        for audio in aud:
                            self.audio_queue.append(audio)
                        if len(self.frames_queue) < self.margin:
                            continue
                        pred = self.predict(self.frames_queue, self.audio_queue)
                        print(pred)
                        if self.show_video:
                            cv.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][5].astype(np.int32), 2, (0,0,255), -1)
                            cv.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][6].astype(np.int32), 2, (0,0,255), -1)
                            cv.imshow('Audio-visual speech recognition in OpenCV', frame)
                            if cv.waitKey(1) & 0xFF == ord('q'):
                                break
                else:
                    self.frames_queue.append(frame)
                    self.audio_queue.append(audioFrame)
        if not self.realtime:
            image_queue = []
            audio_queue = []
            for i in range(len(self.frames_queue)):
                img, aud = self.preprocess(self.frames_queue[i], self.audio_queue[i])
                if img is not None and aud is not None:
                    img = cv.resize(img, (96,96))
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    image_queue.append(img)
                    for audio in aud:
                        audio_queue.append(audio)
                    if len(image_queue) / self.margin > 1:
                        pred = self.predict(image_queue, audio_queue)
                        print(pred)

def parse_args():
    parser = argparse.ArgumentParser(description='Audio Visual Speech Recognition')
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_OPENCV)

    parser.add_argument('--input', type=str,
                        help='Path to input video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--model', type=str, default='AVSpeechRecog.onnx',
                        help='Path to onnx Model file to use for AVSpeech Recognition')
    parser.add_argument('--labels', type=str,
                        help='Path to txt file containing labels. Skip this argument to print class id instead.')
    parser.add_argument('--detector_model', type=str, default='face_detection_yunet_2022mar.onnx',
                        help='Path to YUNet Model onnx file to use for face detection.')
    parser.add_argument('--margin', type=int, default=20,
                        help='Margin for cutting the video')
    parser.add_argument('--video_width', type=int, default=640,
                        help='Preprocess frame by resizing to a specific width.')
    parser.add_argument('--video_height', type=int, default=480,
                        help='Preprocess frame by resizing to a specific Height.')
    parser.add_argument('--score_threshold', type=int, default=0.9,
                        help='score threshold for face detection')
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                        help='NMS threshold for face detection')
    parser.add_argument('--top_k', type=int, default=5000,
                        help='top k for face detection')
    parser.add_argument('--show_video', action='store_true',
                        help='pass --show_video to show video. skip this argument to not show video.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help='Select a computation backend: '
                        "%d: automatically (by default) "
                        "%d: OpenCV Implementation " % backends)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    recognizer = AVSpeechRecognition(args.input, model_path=args.model, detector_path=args.detector_model,
                            margin=args.margin, video_width=args.video_width, video_height=args.video_height,
                            score_threshold=args.score_threshold, nms_threshold=args.nms_threshold, labels=args.labels,
                            top_k=args.top_k, backend=args.backend, show_video=args.show_video)
    recognizer.run()

if __name__ == '__main__':
    main()
