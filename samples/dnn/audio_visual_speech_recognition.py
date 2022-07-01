from collections import deque
import cv2
import numpy as np
# from fairseq import checkpoint_utils, options, tasks, utils
# from fairseq.dataclass.configs import GenerationConfig

class AVSpeechRecognition:
    '''
    Audio Video Speech Recognition based on AVHubert (arXiv:2201.02184 [eess.AS])
    '''
    def __init__(self, source, detector_path='face_detection_yunet_2022mar.onnx', margin=5, video_width=640, video_height=480, score_threshold=0.9, nms_threshold=0.3, top_k=5000):
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
        self.cap = cv2.VideoCapture(source)
        self.margin = margin
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
        self.detector = cv2.FaceDetectorYN.create(detector_path, "", (video_width, video_height), score_threshold, nms_threshold, top_k)
        self.landmarks_queue = deque(maxlen=margin)
        self.frames_queue = deque(maxlen=margin)

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
            self.frames_queue.append(frame)
            if len(self.landmarks_queue) < self.margin:
                return None, None
            smoothed_landmarks = np.mean(self.landmarks_queue, axis=0)[0]
            trans_frame, trans_landmarks = self.warp_image(frame, smoothed_landmarks)
            cropped = self.cut_patch(trans_frame, trans_landmarks[-2:], 96//2,96//2)
            return cropped, smoothed_landmarks
        return None, None

    # def predict(sample):
    #     gen_cfg = GenerationConfig(beam=20)
    #     models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    #     models = [model.eval().cuda() for model in models]
    #     saved_cfg.task.modalities = modalities
    #     saved_cfg.task.data = data_dir
    #     saved_cfg.task.label_dir = data_dir
    #     task = tasks.setup_task(saved_cfg.task)
    #     task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
    #     generator = task.build_generator(models, gen_cfg)

    #     def decode_fn(x):
    #         dictionary = task.target_dictionary
    #         symbols_ignore = generator.symbols_to_strip_from_output
    #         symbols_ignore.add(dictionary.pad())
    #         return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

    #     itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
    #     sample = next(itr)
    #     sample = utils.move_to_cuda(sample)
    #     hypos = task.inference_step(generator, models, sample)
    #     ref = decode_fn(sample['target'][0].int().cpu())
    #     hypo = hypos[0][0]['tokens'].int().cpu()
    #     hypo = decode_fn(hypo)
    #     return hypo, ref

    def run(self):
        '''
        Read the video and process it.
        '''
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cutted_img, _ = self.preprocess(frame)
            if cutted_img is not None:
                cv2.imshow('cutted_img', cutted_img)
            cv2.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][5].astype(np.int32), 5, (0,0,255), -1)
            cv2.circle(frame, np.mean(self.landmarks_queue, axis=0)[0][6].astype(np.int32), 5, (0,0,255), -1)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return 0

if __name__ == '__main__':
    source = 0
    recognizer = AVSpeechRecognition(source)
    recognizer.run()
