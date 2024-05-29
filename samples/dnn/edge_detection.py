'''
    This sample demonstrates edge detection with Dexined and Canny edge detection techniques.
    For switching between deep learning based model (Dexined) and Canny edge detector, press 'd' (for Dexined) or 'c' (for Canny) respectively.
    Script is based on https://github.com/axinc-ai/ailia-models/blob/master/line_segment_detection/dexined/dexined.py

    To download the ONNX model, see:
    https://storage.googleapis.com/ailia-models/dexined/model.onnx

    OpenCV ONNX importer does not process dynamic shape. These need to be substituted with values using:
    python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param w --dim_value 512 model.onnx model.sim1.onnx
    python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param h --dim_value 512 model.sim1.onnx model.sim.onnx
'''
import cv2 as cv
import argparse
import numpy as np

def parse_args():
     # Define supported computation backends and target devices for OpenCV DNN
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                    cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)

    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
                cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(description="This sample demonstrates edge detection with Dexined and Canny edge detection techniques.\n"
    "For switching between deep learning based model (Dexined) and Canny edge detector, press 'd' (for Dexined) or 'c' (for Canny) respectively.")

    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required= False)

    parser.add_argument('--model', help='Path to onnx model', required=False)

    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                            help="Choose one of computation backends: "
                                "%d: automatically (by default), "
                                "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                                "%d: OpenCV implementation, "
                                "%d: VKCOM, "
                                "%d: CUDA" % backends)

    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                            help='Choose one of target computation devices: '
                                '%d: CPU target (by default), '
                                '%d: OpenCL, '
                                '%d: OpenCL fp16 (half-float precision), '
                                '%d: NCS2 VPU, '
                                '%d: HDDL VPU, '
                                '%d: Vulkan, '
                                '%d: CUDA, '
                                '%d: CUDA fp16 (half-float preprocess)'% targets)

    parser.add_argument('--image_size', help='Resize input image to a size', default=512, type=int)

    args = parser.parse_args()

    return args

def preprocess(img):
    # Resize and normalize the image
    IMAGE_SIZE = args.image_size
    img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    img = img.astype(np.float32)
    img[:, :, 0] = (img[:, :, 0] - 103.939)
    img[:, :, 1] = (img[:, :, 1] - 116.779)
    img[:, :, 2] = (img[:, :, 2] - 123.68)

    return img

def sigmoid(x):
    # Apply the sigmoid function
    return 1.0 / (1.0 + np.exp(-x))

def post_processing(output, shape):
    # Process network output to generate edge maps
    h, w = shape

    preds = []
    for p in output:
        img = sigmoid(p)
        img = np.squeeze(img)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        img = cv.resize(img, (w, h))
        preds.append(img)

    fuse = preds[-1]

    ave = np.array(preds, dtype=np.float32)
    ave = np.uint8(np.mean(ave, axis=0))

    return fuse, ave

def canny_detection_thresh1(position, user_data):
    # Threshold adjustment callback for Canny edge detection
    out = cv.Canny(user_data["gray"],position,user_data["thrs2"])
    user_data["thrs1"] =  position

    cv.imshow('Output', out)

def canny_detection_thresh2(position, user_data):
    # Threshold adjustment callback for Canny edge detection
    out = cv.Canny(user_data["gray"],user_data["thrs1"],position)
    user_data["thrs2"] =  position
    cv.imshow('Output', out)

def setupCannyWindow(image, user_data):
    cv.destroyWindow('Output')
    cv.namedWindow('Output', cv.WINDOW_NORMAL)
    cv.moveWindow('Output', 200, 50)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    user_data["gray"] = gray

    cv.createTrackbar('thrs1', 'Output', 0, 255, lambda value: canny_detection_thresh1(value, user_data))
    cv.setTrackbarPos('thrs1', 'Output', 100)
    cv.createTrackbar('thrs2', 'Output', 0, 255, lambda value: canny_detection_thresh2(value, user_data))
    cv.setTrackbarPos('thrs2', 'Output', 200)

# Load the model
def loadModel(args):
    net = cv.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)
    return net

if __name__ == '__main__':
    args = parse_args()
    user_data = {"gray": None, "thrs1": 100, "thrs2": 200}
    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    if not cap.isOpened():
        print("Failed to open the input video")
        exit(-1)
    cv.namedWindow('Input', cv.WINDOW_NORMAL)
    cv.namedWindow('Output', cv.WINDOW_NORMAL)
    cv.moveWindow('Output', 200, 50)

    image_size = args.image_size

    if args.model:
        method = "dexined"
    else:
        print("[WARN] Model file not provided, cannot use dexined")
        method = "canny"
        dummy = np.zeros((image_size, image_size, 3), dtype = "uint8")
        setupCannyWindow(dummy, user_data)

    net = None
    if method == "dexined":
        net = loadModel(args)
    while cv.waitKey(1) < 0:
        hasFrame, image = cap.read()
        if not hasFrame:
            print("Press any key to exit")
            cv.waitKey(0)
            break
        if method == "canny":
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            user_data["gray"] = gray
            canny_detection_thresh1(user_data["thrs1"], user_data)
        elif method == "dexined":
            # Preprocess the image
            img =  preprocess(image.copy())
            inp = cv.dnn.blobFromImage(img, swapRB=False, crop=False)
            net.setInput(inp)
            out = net.forward()
            # Post processing on the model output
            out = post_processing(out, image.shape[:2])
            # Put efficiency information.
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv.putText(out[1], label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv.imshow("Output", out[1])
        cv.imshow("Input", image)
        key = cv.waitKey(30)
        if key == ord('d') or key == ord('D'):
            if args.model:
                method = "dexined"
            else:
                print("[WARN] Provide model file using --model to use dexined")
            if net is None:
                net = loadModel(args)
            cv.destroyWindow('Output')
            cv.namedWindow('Output', cv.WINDOW_NORMAL)
            cv.moveWindow('Output', 200, 50)
        elif key == ord('c') or key == ord('C'):
            method = "canny"
            setupCannyWindow(image, user_data)
        elif key == 27 or key == ord('q'):
            break
    cv.destroyAllWindows()