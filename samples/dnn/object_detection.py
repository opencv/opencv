import cv2 as cv
import argparse
import numpy as np
import sys
import copy
import time
from threading import Thread
import queue

from common import *
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph

def help():
    print(
        '''
        Firstly, download required models using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"\n

        To run:
            python object_detection.py model_name(e.g yolov8) --input=path/to/your/input/image/or/video (don't pass --input to use device camera)

        Sample command:
            python object_detection.py yolov8 --input=path/to/image
        Model path can also be specified using --model argument
        '''
    )

backends = ("default", "openvino", "opencv", "vkcom", "cuda")
targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                    help='An optional path to file with preprocessing parameters.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--out_tf_graph', default='graph.pbtxt',
                    help='For models from TensorFlow Object Detection API, you may '
                         'pass a .config file which was used for training through --config '
                         'argument. This way an additional .pbtxt file with TensorFlow graph will be created.')
parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'darknet', 'dldt', 'onnx'],
                    help='Optional name of an origin framework of the model. '
                         'Detect it automatically if it does not set.')
parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold')
parser.add_argument('--backend', default="default", type=str, choices=backends,
                    help="Choose one of computation backends: "
                    "default: automatically (by default), "
                    "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                    "opencv: OpenCV implementation, "
                    "vkcom: VKCOM, "
                    "cuda: CUDA, "
                    "webnn: WebNN")
parser.add_argument('--target', default="cpu", type=str, choices=targets,
                    help="Choose one of target computation devices: "
                    "cpu: CPU target (by default), "
                    "opencl: OpenCL, "
                    "opencl_fp16: OpenCL fp16 (half-float precision), "
                    "ncs2_vpu: NCS2 VPU, "
                    "hddl_vpu: HDDL VPU, "
                    "vulkan: Vulkan, "
                    "cuda: CUDA, "
                    "cuda_fp16: CUDA fp16 (half-float preprocess)")
parser.add_argument('--async', type=int, default=0,
                    dest='use_threads',
                    help='Choose 0 for synchronous mode and 1 for asynchronous mode')
args, _ = parser.parse_known_args()
add_preproc_args(args.zoo, parser, 'object_detection')
parser = argparse.ArgumentParser(parents=[parser],
                                 description='Use this script to run object detection deep learning networks using OpenCV.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()

if args.alias is None or hasattr(args, 'help'):
    help()
    exit(1)

args.model = findModel(args.model, args.sha1)
if args.config is not None:
    args.config = findFile(args.config)
args.labels = findFile(args.labels)

# If config specified, try to load it as TensorFlow Object Detection API's pipeline.
config = readTextMessage(args.config)
if 'model' in config:
    print('TensorFlow Object Detection API config detected')
    if 'ssd' in config['model'][0]:
        print('Preparing text graph representation for SSD model: ' + args.out_tf_graph)
        createSSDGraph(args.model, args.config, args.out_tf_graph)
        args.config = args.out_tf_graph
    elif 'faster_rcnn' in config['model'][0]:
        print('Preparing text graph representation for Faster-RCNN model: ' + args.out_tf_graph)
        createFasterRCNNGraph(args.model, args.config, args.out_tf_graph)
        args.config = args.out_tf_graph


# Load names of classes
labels = None
if args.labels:
    with open(args.labels, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')

# Load a network
net = cv.dnn.readNet(args.model, args.config, args.framework)
net.setPreferableBackend(get_backend_id(args.backend))
net.setPreferableTarget(get_target_id(args.target))
outNames = net.getUnconnectedOutLayersNames()

confThreshold = args.thr
nmsThreshold = args.nms
stdSize = 0.8
stdWeight = 2
stdImgSize = 512
asyncN = 0

def get_color(class_id):
    r = min((class_id >> 0 & 1) * 128 + (class_id >> 3 & 1) * 64 + (class_id >> 6 & 1) * 32 + 80, 255)
    g = min((class_id >> 1 & 1) * 128 + (class_id >> 4 & 1) * 64 + (class_id >> 7 & 1) * 32 + 40, 255)
    b = min((class_id >> 2 & 1) * 128 + (class_id >> 5 & 1) * 64 + (class_id >> 8 & 1) * 32 + 40, 255)
    return (int(b), int(g), int(r))

def get_text_color(bg_color):
    luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1
                    if width <= 2 or height <= 2:
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region' or args.postprocessing == 'yolov8':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        if args.postprocessing == 'yolov8':
            box_scale_w = frameWidth / args.width
            box_scale_h = frameHeight / args.height
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        for out in outs:
            if args.postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)

            for detection in out:
                scores = detection[4:]
                if args.background_label_id >= 0:
                    scores = np.delete(scores, args.background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or (lastLayer.type == 'Region' or args.postprocessing == 'yolov8') and args.backend != cv.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    return boxes, classIds, confidences, indices

def drawPred(classIds, confidences, boxes, indices, fontSize, fontThickness):
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        right = box[0] + box[2]
        bottom = box[1] + box[3]
        bg_color = get_color(classIds[i])
        cv.rectangle(frame, (left, top), (right, bottom), bg_color, fontThickness)

        label = '%.2f' % confidences[i]

        # Print a label of class.
        if labels:
            assert(classIds[i] < len(labels))
            label = '%s: %s' % (labels[classIds[i]], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (int(left-fontThickness/2), top - labelSize[1]), (left + labelSize[0], top + baseLine), bg_color, cv.FILLED)
        cv.putText(frame, label, (left, top-fontThickness), cv.FONT_HERSHEY_SIMPLEX, fontSize, get_text_color(bg_color), fontThickness)

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)

def callback(pos):
    global confThreshold
    confThreshold = pos / 100.0

cv.createTrackbar('Confidence threshold, %', winName, int(confThreshold * 100), 99, callback)

cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input) if args.input else 0)

class QueueFPS(queue.Queue):
    def __init__(self):
        queue.Queue.__init__(self)
        self.startTime = 0
        self.counter = 0

    def put(self, v):
        queue.Queue.put(self, v)
        self.counter += 1
        if self.counter == 1:
            self.startTime = time.time()

    def getFPS(self):
        return self.counter / (time.time() - self.startTime)


process = True

#
# Frames capturing thread
#
framesQueue = QueueFPS()
def framesThreadBody():
    global framesQueue, process

    while process:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        framesQueue.put(frame)


#
# Frames processing thread
#
processedFramesQueue = queue.Queue()
predictionsQueue = QueueFPS()
def processingThreadBody():
    global processedFramesQueue, predictionsQueue, args, process, asyncN

    futureOutputs = []
    while process:
        # Get a next frame
        frame = None
        try:
            frame = framesQueue.get_nowait()

            if asyncN:
                if len(futureOutputs) == asyncN:
                    frame = None  # Skip the frame
            else:
                framesQueue.queue.clear()  # Skip the rest of frames
        except queue.Empty:
            pass


        if not frame is None:
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]

            # Create a 4D blob from a frame.
            inpWidth = args.width if args.width else frameWidth
            inpHeight = args.height if args.height else frameHeight
            blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=args.rgb, ddepth=cv.CV_32F)
            processedFramesQueue.put(frame)

            # Run a model
            net.setInput(blob, scalefactor=args.scale, mean=args.mean)
            if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                frame = cv.resize(frame, (inpWidth, inpHeight))
                net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

            if asyncN:
                futureOutputs.append(net.forwardAsync())
            else:
                outs = net.forward(outNames)
                predictionsQueue.put(copy.deepcopy(outs))

        while futureOutputs and futureOutputs[0].wait_for(0):
            out = futureOutputs[0].get()
            predictionsQueue.put(copy.deepcopy([out]))

            del futureOutputs[0]

if args.use_threads:
    framesThread = Thread(target=framesThreadBody)
    framesThread.start()

    processingThread = Thread(target=processingThreadBody)
    processingThread.start()

    #
    # Postprocessing and rendering loop
    #
    while cv.waitKey(1) < 0:
        try:
            # Request prediction first because they put after frames
            outs = predictionsQueue.get_nowait()
            frame = processedFramesQueue.get_nowait()
            imgWidth = max(frame.shape[:2])
            fontSize = (stdSize*imgWidth)/stdImgSize
            fontThickness = max(1,(stdWeight*imgWidth)//stdImgSize)

            boxes, classIds, confidences, indices = postprocess(frame, outs)
            drawPred(classIds, confidences, boxes, indices, fontSize, fontThickness)
            fontSize = fontSize/2
            # Put efficiency information.
            if predictionsQueue.counter > 1:
                label = 'Camera: %.2f FPS' % (framesQueue.getFPS())
                cv.rectangle(frame, (0, 0), (int(260*fontSize), int(80*fontSize)), (255,255,255), cv.FILLED)
                cv.putText(frame, label, (0, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)

                label = 'Network: %.2f FPS' % (predictionsQueue.getFPS())
                cv.putText(frame, label, (0, int(2*25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)

                label = 'Skipped frames: %d' % (framesQueue.counter - predictionsQueue.counter)
                cv.putText(frame, label, (0, int(3*25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)

            cv.imshow(winName, frame)
        except queue.Empty:
            pass


    process = False
    framesThread.join()
    processingThread.join()

else:
    # Non-threaded processing if --async is 0
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        inpWidth = args.width if args.width else frameWidth
        inpHeight = args.height if args.height else frameHeight
        blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=args.rgb, ddepth=cv.CV_32F)

        net.setInput(blob, scalefactor=args.scale, mean=args.mean)
        outs = net.forward(outNames)

        boxes, classIds, confidences, indices = postprocess(frame, outs)
        drawPred(classIds, confidences, boxes, indices, (stdSize*max(frame.shape[:2]))/stdImgSize, (stdWeight*max(frame.shape[:2]))//stdImgSize)

        cv.imshow(winName, frame)