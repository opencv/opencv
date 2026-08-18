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

        For promptable segmentation (sam) pass a foreground point with --point=x,y (defaults to the image centre):
            python segmentation.py sam --input=path/to/your/input/image/or/video --point=320,240 (don't give --input flag if want to use device camera)
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
    parser.add_argument('--point', help="Foreground point prompt as 'x,y' in input image coordinates, "
                                        "used by promptable models (sam). Defaults to the image centre.")
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
    args, _ = parser.parse_known_args()
    if args.alias == 'sam':
        add_preproc_args(args.zoo, parser, 'segmentation', prefix='decoder_', alias='sam')
    parser = argparse.ArgumentParser(parents=[parser],
                                    description='Use this script to run semantic segmentation deep learning networks using OpenCV.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args(func_args)

# SAM input: longest edge scaled to target, per-channel normalized, zero-padded to a square.
# Not expressible with blobFromImage; newH/newW report the unpadded extent for mask cropping.
def samPreprocess(frame, target):
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    h, w = frame.shape[:2]
    s = target / max(h, w)
    newH, newW = int(h * s + 0.5), int(w * s + 0.5)

    img = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2RGB), (newW, newH), interpolation=cv.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - mean) / std

    blob = np.zeros((1, 3, target, target), np.float32)
    blob[0, :, :newH, :newW] = img.transpose(2, 0, 1)
    return blob, newH, newW

def showLegend(labels, colors, legend):
    if not labels is None and legend is None:
        blockHeight = 30
        assert(len(labels) == len(colors))

        legend = np.zeros((blockHeight * len(colors), 200, 3), np.uint8)
        for i in range(len(labels)):
            block = legend[i * blockHeight:(i + 1) * blockHeight]
            block[:,:] = colors[i]
            cv.putText(block, labels[i], (0, blockHeight//2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.namedWindow('Legend', cv.WINDOW_AUTOSIZE)
        cv.imshow('Legend', legend)
        labels = None

def main(func_args=None):
    args = get_args_parser(func_args)
    if args.alias is None or hasattr(args, 'help'):
        help()
        exit(1)

    cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_INFO)
    args.model = findModel(args.model, args.sha1)
    if args.labels is not None:
        args.labels = findFile(args.labels)

    np.random.seed(324)

    stdSize = 0.8
    stdWeight = 2
    stdImgSize = 512
    imgWidth = -1 # Initialization
    fontSize = 1.5
    fontThickness = 1

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
    engine = cv.dnn.ENGINE_OPENCV
    net = cv.dnn.readNetFromONNX(args.model, engine)
    net.setPreferableBackend(get_backend_id(args.backend))
    net.setPreferableTarget(get_target_id(args.target))
    if hasattr(cv.dnn, 'DNN_PROFILE_SUMMARY'):
        net.setProfilingMode(cv.dnn.DNN_PROFILE_SUMMARY)

    # Promptable models split into an image encoder (the primary model) and a prompt/mask decoder.
    decoder = None
    if args.alias == 'sam':
        decoder = cv.dnn.readNetFromONNX(findModel(args.decoder_model, args.decoder_sha1), engine)
        decoder.setPreferableBackend(get_backend_id(args.backend))
        decoder.setPreferableTarget(get_target_id(args.target))

    point = None
    if args.point:
        point = tuple(int(v) for v in args.point.split(','))

    winName = 'Deep learning semantic segmentation in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)

    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    if not cap.isOpened():
        print("Failed to open the input video")
        exit(-1)

    legend = None
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        if imgWidth == -1:
            imgWidth = max(frame.shape[:2])
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize)
            fontThickness = max(fontThickness,(stdWeight*imgWidth)//stdImgSize)

        cv.imshow("Original Image", frame)
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Create a 4D blob from a frame.
        inpWidth = args.width if args.width else frameWidth
        inpHeight = args.height if args.height else frameHeight

        if args.alias != 'sam':  # SAM builds its own padded blob and uses named inputs
            blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=False)
            net.setInput(blob)

        t0 = cv.getTickCount()
        if args.alias == 'sam':
            blob, newH, newW = samPreprocess(frame, inpWidth)
            net.setInput(blob, 'pixel_values')
            emb, pos = net.forward(['image_embeddings', 'image_positional_embeddings'])
            net.printPerfProfile()

            # The prompt is given in input image coordinates, so scale it into the padded frame.
            pt = point if point else (frameWidth // 2, frameHeight // 2)
            s = inpWidth / max(frameHeight, frameWidth)
            inputPoints = np.array([[[[pt[0] * s, pt[1] * s]]]], np.float32)
            inputLabels = np.ones((1, 1, 1), np.int64)  # 1 = foreground point

            decoder.setInput(inputPoints, 'input_points')
            decoder.setInput(inputLabels, 'input_labels')
            decoder.setInput(emb, 'image_embeddings')
            decoder.setInput(pos, 'image_positional_embeddings')
            iouScores, predMasks = decoder.forward(['iou_scores', 'pred_masks'])

            # The decoder proposes several masks per prompt; keep the highest scoring one.
            best = int(np.argmax(iouScores.reshape(-1)))
            lowRes = predMasks.reshape(-1, predMasks.shape[-2], predMasks.shape[-1])[best]

            # Mask logits cover the padded square: upsample, crop the unpadded extent, then
            # resize to the frame. A logit above zero belongs to the object.
            padded = cv.resize(lowRes, (inpWidth, inpHeight), interpolation=cv.INTER_LINEAR)
            logits = cv.resize(padded[:newH, :newW], (frameWidth, frameHeight), interpolation=cv.INTER_LINEAR)

            overlay = np.zeros_like(frame)
            overlay[logits > 0] = (0, 0, 255)
            frame = cv.addWeighted(frame, 0.6, overlay, 0.4, 0)
            cv.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), cv.FILLED)
        elif args.alias == 'u2netp':
            output = net.forward(net.getUnconnectedOutLayersNames())
            net.printPerfProfile()
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
            net.printPerfProfile()

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

        label = 'Inference time: %.2f ms' % ((cv.getTickCount() - t0) * 1000.0 / cv.getTickFrequency())
        labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
        cv.rectangle(frame, (0, 0), (labelSize[0]+10, labelSize[1]), (255,255,255), cv.FILLED)
        cv.putText(frame, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)

        cv.imshow(winName, frame)

if __name__ == "__main__":
    main()
