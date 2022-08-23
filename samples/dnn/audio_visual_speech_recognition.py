from collections import deque
import cv2
import numpy as np
import argparse

'''
AVSpeechRecognition

    How to obtain the model required for this sample:
    Option 1: Download the model from https://drive.google.com/file/d/1xuwk5ZQagKFoTXev27zvlSg8dAmSJoo7/view?usp=sharing
    Option 2: Convert the model using pretrained torch model and base repo.
        Use the colab notebook here: https://colab.research.google.com/drive/1awBCZ5O6uAT32cHvufNWad5m6q26TuqQ?usp=sharing
'''

class AVSpeechRecognition:
    '''
    Audio Video Speech Recognition based on AVHubert (arXiv:2201.02184 [eess.AS])
    '''
    def __init__(self, source, type='camera', detector_path='face_detection_yunet_2022mar.onnx', model_path = 'AVSpeechRecog.onnx', margin=20, video_width=640, video_height=480, score_threshold=0.9, nms_threshold=0.3, top_k=5000, show_video=False):
        '''
        params:
            source: video source (e.g. '0', 'video.mp4')
            detector_path: face detection model path (default:'face_detection_yunet_2022mar.onnx')
            margin: margin for temporal window (default:5)
            video_width: video width (default:640)
            video_height: video height (default:480)
            score_threshold: score threshold for face detection (default:0.9)
            nms_threshold: nms threshold for face detection (default:0.3)
            top_k: top k faces for face detection (default:5000)
        '''
        if type not in ['file', 'camera']:
            raise Exception('type must be file or camera')

        self.cap = cv2.VideoCapture(source)
        samplingRate = 16000
        self.input_type = type
        self.source=source
        self.params = np.asarray([cv2.CAP_PROP_AUDIO_STREAM, 0,
                cv2.CAP_PROP_VIDEO_STREAM, 0,
                cv2.CAP_PROP_AUDIO_DATA_DEPTH, cv2.CV_32F,
                cv2.CAP_PROP_AUDIO_SAMPLES_PER_SECOND, samplingRate
                ])
        self.margin = margin
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
        self.detector = cv2.FaceDetectorYN.create(detector_path, "", (video_width, video_height), score_threshold, nms_threshold, top_k)
        self.landmarks_queue = deque(maxlen=margin)
        self.frames_queue = deque(maxlen=margin)
        self.audio_queue = deque(maxlen=margin)
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.show_video = show_video

    def warp_image(self, frame, smoothed_landmarks):
        '''
        warps frame to make lips horizontal and fixed at center
        params:
            frame: input frame
            smoothed_landmarks: smoothed landmarks
        return:
            warped_frame: warped frame
            warped_landmarks: warped landmarks
        '''
        # TODO: fix warping
        rotateby = np.arctan((smoothed_landmarks[6][1]-smoothed_landmarks[5][1])/(smoothed_landmarks[6][0]-smoothed_landmarks[5][0]))*180/np.pi
        image_center = tuple((smoothed_landmarks[0]+smoothed_landmarks[1])/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rotateby, 1)
        trans_frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
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

    def preprocess(self, frame):
        '''
        preprocesses frame to get landmarks and mouth rois
        params:
            frame: input frame
        return:
            cropped: mouth roi
            smoothed_landmarks: smoothed/averaged landmarks
        '''
        landmarks = self.detector.detect(frame)[-1]
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
            return cropped, smoothed_landmarks
        return None, None

    def predict(self):
        '''
        predicts word using Audio Video Speech Recognition model.
        return:
            pred: predicted word
        '''
        # video_input = cv2.resize(video_input, (96,96))
        # video_input = video_input.astype(np.float32)
        # video_input = video_input.transpose(2,0,1)
        # video_input = video_input.reshape(1,3,96,96)
        # video_input = video_input/255.0
        # self.model.setInput(np.array(self.audio_queue), 'audio_input')
        self.model.setInput(np.random.randn(1,1,12800), 'audio_input')
        video = np.expand_dims(np.expand_dims(np.array(self.frames_queue),0),0)
        self.model.setInput(video, 'video_input')
        out = self.model.forward()
        pred = out[0].argmax()
        return pred

    def run(self):
        '''
        Read the video and process it.
        '''
        while True:
            # Read frame along with audio and process it
            # self.cap.open(self.source, cv2.CAP_ANY, self.params)
            # audioBaseIndex = int(self.cap.get(cv2.CAP_PROP_AUDIO_BASE_INDEX))
            # cvTickFreq = cv2.getTickFrequency()
            # sysTimeCurr = cv2.getTickCount()
            # sysTimePrev = sysTimeCurr
            # while ((sysTimeCurr - sysTimePrev) / cvTickFreq < 10):
            #     if (self.cap.grab()):
            #         frame = np.asarray([])
            #         # Get the video and audio data
            #         ret, frame = self.cap.retrieve(frame, audioBaseIndex)
            #         if not ret:
            #             break
            #         # preprocess frame and get landmarks and mouth roi
            #         cropped, landmarks = self.preprocess(frame)
            #         if self.show_video:
            #             cv2.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][5].astype(np.int32), 5, (0,0,255), -1)
            #             cv2.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][6].astype(np.int32), 5, (0,0,255), -1)
            #             cv2.imshow('frame', frame)
            #         if cropped is not None:
            #             if self.show_video:
            #                 cv2.imshow('cropped', cropped)
            #             cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            #             # add to queue
            #             # self.audio_queue.append(inputAudio)
            #             # TODO: extract audio here as well
            #             self.frames_queue.append(cropped)
            #             self.landmarks_queue.append(landmarks)
            #             # predict word
            #             if len(self.frames_queue) == self.margin:
            #                 pred = self.predict()
            #                 print(self.labels[pred])
            #             print(pred)
            #     else:
            #         print("Error: Grab error")
            #         break
            ret, frame = self.cap.read()
            if not ret:
                break
            cutted_img, _ = self.preprocess(frame)
            if cutted_img is not None:
                cv2.imshow('cutted_img', cutted_img)
                cutted_img = cv2.cvtColor(cutted_img, cv2.COLOR_BGR2GRAY)
                cutted_img = cv2.resize(cutted_img, (96,96))
                self.frames_queue.append(cutted_img)
                if len(self.frames_queue) == self.margin:
                    pred = self.predict()
                    print(pred)
                cv2.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][5].astype(np.int32), 5, (0,0,255), -1)
                cv2.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][6].astype(np.int32), 5, (0,0,255), -1)
            # else:
            #     self.landmarks_queue.clear()
            #     self.frames_queue.clear()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return 0

def main():
    '''
    main function
    '''
    parser = argparse.ArgumentParser(description='Video to text')
    parser.add_argument('--source', type=str, default='0', help='Source of the video')
    parser.add_argument('--model', type=str, default='AVSpeechRecog.onnx', help='Model to use')
    parser.add_argument('--detector_model', type=str, default='face_detection_yunet_2022mar.onnx', help='Model to use')
    parser.add_argument('--margin', type=int, default=20, help='Margin for cutting the video')
    parser.add_argument('--video_width', type=int, default=640, help='Video width for cutting the video')
    parser.add_argument('--video_height', type=int, default=480, help='Video height for cutting the video')
    parser.add_argument('--score_threshold', type=int, default=0.9, help='score threshold for face detection')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='NMS threshold for face detection')
    parser.add_argument('--top_k', type=int, default=5000, help='top k for face detection')
    parser.add_argument('--show_video', type=bool, default=False, help='Show video or Not')
    args = parser.parse_args()

    # source = args.source=='0' and 0 or args.source
    recognizer = AVSpeechRecognition(0, model_path=args.model, detector_path=args.detector_model,
                            margin=args.margin, video_width=args.video_width, video_height=args.video_height,
                            score_threshold=args.score_threshold, nms_threshold=args.nms_threshold,
                            top_k=args.top_k, show_video=args.show_video)
    recognizer.run()

if __name__ == '__main__':
    main()
