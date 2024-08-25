import cv2
import argparse
 
 
backends = (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_BACKEND_HALIDE, cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_BACKEND_VKCOM, cv2.dnn.DNN_BACKEND_CUDA)
targets = (cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD,
            cv2.dnn.DNN_TARGET_VULKAN, cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16)
 
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--net', type=str, default="vitTracker.onnx", help="Path to the VitTracker model file.")
parser.add_argument('--tracking_score_threshold', type=float,  help="Tracking score threshold. If a bbox of score >= 0.3, it is considered as found ")
parser.add_argument('--backend', choices=backends, default=cv2.dnn.DNN_BACKEND_DEFAULT, type=int,
                help="Choose one of computation backends: "
                        "%d: automatically (by default), "
                        "%d: Halide language (http://halide-lang.org/), "
                        "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "%d: OpenCV implementation, "
                        "%d: VKCOM, "
                        "%d: CUDA"% backends)
parser.add_argument("--target", choices=targets, default=cv2.dnn.DNN_TARGET_CPU, type=int,
                help="Choose one of target computation devices: "
                        '%d: CPU target (by default), '
                        '%d: OpenCL, '
                        '%d: OpenCL fp16 (half-float precision), '
                        '%d: VPU, '
                        '%d: VULKAN, '
                        '%d: CUDA, '
                        '%d: CUDA fp16 (half-float preprocess)'% targets)
#Parse command line arguments.
args = parser.parse_args()
 
try:
    tracker_params = cv2.TrackerVit_Params()
    tracker_params.net = args.net
    tracker_params.backend = args.backend
    tracker_params.target = args.target
    tracker_params.tracking_score_threshold = 0.5
    tracker = cv2.TrackerVit_create(tracker_params)
except Exception as e:
    print(f"Exception: {e}")
    print(f"Can't load the network using file: {args.net}")
    exit(2)
 
 
 
cap = cv2.VideoCapture(args.input if args.input else 0)
 
ret, image = cap.read()
if not ret or image is None:
    print("Can't capture frame!")
    cap.release()
    exit(2)
 
image_select = image.copy()
cv2.putText(image_select, "Select initial bounding box you want to track.", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
cv2.putText(image_select, "And Press the ENTER key.", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
 
select_rect = cv2.selectROI("vitTracker", image_select)
print(f"ROI={select_rect}")
if select_rect == (0, 0, 0, 0):
    print("Invalid ROI!")
    cap.release()
    exit(2)
 
tracker.init(image, select_rect)
 
tick_meter = cv2.TickMeter()
 
while True:
    ret, image = cap.read()
    if not ret or image is None:
        print("Can't capture frame. End of video stream?")
        break
 
    tick_meter.start()
    ok, rect = tracker.update(image)
    tick_meter.stop()
 
    if ok:
        cv2.rectangle(image, rect, (0, 255, 0), 2)
 
        time_label = f"Inference time: {tick_meter.getTimeMilli():.2f} ms"
        score_label = f"Tracking score: {tracker.getTrackingScore():.2f}"
        cv2.putText(image, time_label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(image, score_label, (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        print("Target lost")
        cv2.putText(image, "Target lost", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
 
    cv2.imshow("vitTracker", image)
 
    tick_meter.reset()
 
    if cv2.waitKey(1) & 0xFF == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
 
print("Exit")