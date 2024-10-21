import cv2 as cv
import argparse
import numpy as np

from common import *

def help():
    print(
        '''
        Firstly, download required models using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"\n

        To run:
            python segmentation.py model_name(e.g. u2netp) --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)

        Model path can also be specified using --model argument
        '''
    )

def get_args_parser(func_args):
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--colors', help='Optional path to a text file with colors for an every class. '
                                        'An every color is represented with three values from 0 to 255 in BGR channels order.')
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

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'segmentation')
    parser = argparse.ArgumentParser(parents=[parser],
                                    description='Use this script to run semantic segmentation deep learning networks using OpenCV.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args(func_args)

def showLegend(labels, colors, legend):
    if not labels is None and legend is None:
        blockHeight = 30
        assert(len(labels) == len(colors))

        legend = np.zeros((blockHeight * len(colors), 200, 3), np.uint8)
        for i in range(len(labels)):
            block = legend[i * blockHeight:(i + 1) * blockHeight]
            block[:,:] = colors[i]
            cv.putText(block, labels[i], (0, blockHeight//2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        cv.namedWindow('Legend', cv.WINDOW_AUTOSIZE)
        cv.imshow('Legend', legend)
        labels = None

def main(func_args=None):
    args = get_args_parser(func_args)
    if args.alias is None or hasattr(args, 'help'):
        help()
        exit(1)

    args.model = findModel(args.model, args.sha1)
    if args.labels is not None:
        args.labels = findFile(args.labels)

    np.random.seed(324)

    # Load names of labels
    labels = None
    if args.labels:
        with open(args.labels, 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')

    # Load colors
    colors = None
    if args.colors:
        with open(args.colors, 'rt') as f:
            colors = [np.array(color.split(' '), np.uint8) for color in f.read().rstrip('\n').split('\n')]

    # Load a network
    engine = cv.dnn.ENGINE_AUTO
    net = cv.dnn.readNetFromONNX(args.model, engine)
    net.setPreferableBackend(get_backend_id(args.backend))
    net.setPreferableTarget(get_target_id(args.target))

    winName = 'Deep learning semantic segmentation in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)

    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    legend = None
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        cv.imshow("Original Image", frame)
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Create a 4D blob from a frame.
        inpWidth = args.width if args.width else frameWidth
        inpHeight = args.height if args.height else frameHeight

        blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=False)
        net.setInput(blob)

        if args.alias == 'u2netp':
            output = net.forward(net.getUnconnectedOutLayersNames())
            pred = output[0][0, 0, :, :]
            mask = (pred * 255).astype(np.uint8)
            mask = cv.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_AREA)
            # Create overlays for foreground and background
            foreground_overlay = np.zeros_like(frame, dtype=np.uint8)
            # Set foreground (object) to red and background to blue
            foreground_overlay[:, :, 2] = mask  # Red foreground
            # Blend the overlays with the original frame
            frame = cv.addWeighted(frame, 0.25, foreground_overlay, 0.75, 0)
        else:
            score = net.forward()

            numClasses = score.shape[1]
            height = score.shape[2]
            width = score.shape[3]
            # Draw segmentation
            if not colors:
                # Generate colors
                colors = [np.array([0, 0, 0], np.uint8)]
                for i in range(1, numClasses):
                    colors.append((colors[i - 1] + np.random.randint(0, 256, [3], np.uint8)) / 2)
            classIds = np.argmax(score[0], axis=0)
            segm = np.stack([colors[idx] for idx in classIds.flatten()])
            segm = segm.reshape(height, width, 3)

            segm = cv.resize(segm, (frameWidth, frameHeight), interpolation=cv.INTER_NEAREST)
            frame = (0.1 * frame + 0.9 * segm).astype(np.uint8)

            showLegend(labels, colors, legend)

        # Put efficiency information.
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv.imshow(winName, frame)

if __name__ == "__main__":
    main()