import argparse
import time
import numpy as np
import cv2 as cv


# ------------------------Service operations------------------------
def weight_path(model_path):
    """ Get path of weights based on path to IR

    Params:
    model_path: the string contains path to IR file

    Return:
    Path to weights file
    """
    assert model_path.endswith('.xml'), "Wrong topology path was provided"
    return model_path[:-3] + 'bin'


def build_argparser():
    """ Parse arguments from command line

    Return:
    Pack of arguments from command line
    """
    parser = argparse.ArgumentParser(description='This is an OpenCV-based version of Gaze Estimation example')

    parser.add_argument('--input',
                        help='Path to the input video file or camera device number')
    parser.add_argument('--out',
                        help='Path to the output video file')
    parser.add_argument('--facem',
                        default='face-detection-retail-0005.xml',
                        help='Path to OpenVINO face detection model (.xml)')
    parser.add_argument('--faced',
                        default='CPU',
                        help='Target device for the face detection' +
                        '(e.g. CPU, GPU, VPU, ...)')
    parser.add_argument('--headm',
                        default='head-pose-estimation-adas-0001.xml',
                        help='Path to OpenVINO head pose estimation model (.xml)')
    parser.add_argument('--headd',
                        default='CPU',
                        help='Target device for the head pose estimation inference ' +
                        '(e.g. CPU, GPU, VPU, ...)')
    parser.add_argument('--landm',
                        default='facial-landmarks-35-adas-0002.xml',
                        help='Path to OpenVINO landmarks detector model (.xml)')
    parser.add_argument('--landd',
                        default='CPU',
                        help='Target device for the landmarks detector (e.g. CPU, GPU, VPU, ...)')
    parser.add_argument('--gazem',
                        default='gaze-estimation-adas-0002.xml',
                        help='Path to OpenVINO gaze vector estimaiton model (.xml)')
    parser.add_argument('--gazed',
                        default='CPU',
                        help='Target device for the gaze vector estimation inference ' +
                        '(e.g. CPU, GPU, VPU, ...)')
    parser.add_argument('--eyem',
                        default='open-closed-eye-0001.xml',
                        help='Path to OpenVINO open closed eye model (.xml)')
    parser.add_argument('--eyed',
                        default='CPU',
                        help='Target device for the eyes state inference (e.g. CPU, GPU, VPU, ...)')
    return parser


# ------------------------Support functions for custom kernels------------------------
def intersection(surface, rect):
    """ Remove zone of out of bound from ROI

    Params:
    surface: image bounds is rect representation (top left coordinates and width and height)
    rect: region of interest is also has rect representation

    Return:
    Modified ROI with correct bounds
    """
    l_x = max(surface[0], rect[0])
    l_y = max(surface[1], rect[1])
    width = min(surface[0] + surface[2], rect[0] + rect[2]) - l_x
    height = min(surface[1] + surface[3], rect[1] + rect[3]) - l_y
    if width < 0 or height < 0:
        return (0, 0, 0, 0)
    return (l_x, l_y, width, height)


def process_landmarks(r_x, r_y, r_w, r_h, landmarks):
    """ Create points from result of inference of facial-landmarks network and size of input image

    Params:
    r_x: x coordinate of top left corner of input image
    r_y: y coordinate of top left corner of input image
    r_w: width of input image
    r_h: height of input image
    landmarks: result of inference of facial-landmarks network

    Return:
    Array of landmarks points for one face
    """
    lmrks = landmarks[0]
    raw_x = lmrks[::2] * r_w + r_x
    raw_y = lmrks[1::2] * r_h + r_y
    return np.array([[int(x), int(y)] for x, y in zip(raw_x, raw_y)])


def eye_box(p_1, p_2, scale=1.8):
    """ Get bounding box of eye

    Params:
    p_1: point of left edge of eye
    p_2: point of right edge of eye
    scale: change size of box with this value

    Return:
    Bounding box of eye and its midpoint
    """

    size = np.linalg.norm(p_1 - p_2)
    midpoint = (p_1 + p_2) / 2
    width = scale * size
    height = width
    p_x = midpoint[0] - (width / 2)
    p_y = midpoint[1] - (height / 2)
    return (int(p_x), int(p_y), int(width), int(height)), list(map(int, midpoint))


# ------------------------Custom graph operations------------------------
@cv.gapi.op('custom.GProcessPoses',
            in_types=[cv.GArray.GMat, cv.GArray.GMat, cv.GArray.GMat],
            out_types=[cv.GArray.GMat])
class GProcessPoses:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1, arr_desc2):
        return cv.empty_array_desc()


@cv.gapi.op('custom.GParseEyes',
            in_types=[cv.GArray.GMat, cv.GArray.Rect, cv.GOpaque.Size],
            out_types=[cv.GArray.Rect, cv.GArray.Rect, cv.GArray.Point, cv.GArray.Point])
class GParseEyes:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1, arr_desc2):
        return cv.empty_array_desc(), cv.empty_array_desc(), \
               cv.empty_array_desc(), cv.empty_array_desc()


@cv.gapi.op('custom.GGetStates',
            in_types=[cv.GArray.GMat, cv.GArray.GMat],
            out_types=[cv.GArray.Int, cv.GArray.Int])
class GGetStates:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1):
        return cv.empty_array_desc(), cv.empty_array_desc()


# ------------------------Custom kernels------------------------
@cv.gapi.kernel(GProcessPoses)
class GProcessPosesImpl:
    """ Custom kernel. Processed poses of heads
    """
    @staticmethod
    def run(in_ys, in_ps, in_rs):
        """ Сustom kernel executable code

        Params:
        in_ys: yaw angle of head
        in_ps: pitch angle of head
        in_rs: roll angle of head

        Return:
        Arrays with heads poses
        """
        return [np.array([ys[0], ps[0], rs[0]]).T for ys, ps, rs in zip(in_ys, in_ps, in_rs)]


@cv.gapi.kernel(GParseEyes)
class GParseEyesImpl:
    """ Custom kernel. Get information about eyes
    """
    @staticmethod
    def run(in_landm_per_face, in_face_rcs, frame_size):
        """ Сustom kernel executable code

        Params:
        in_landm_per_face: landmarks from inference of facial-landmarks network for each face
        in_face_rcs: bounding boxes for each face
        frame_size: size of input image

        Return:
        Arrays of ROI for left and right eyes, array of midpoints and
        array of landmarks points
        """
        left_eyes = []
        right_eyes = []
        midpoints = []
        lmarks = []
        surface = (0, 0, *frame_size)
        for landm_face, rect in zip(in_landm_per_face, in_face_rcs):
            points = process_landmarks(*rect, landm_face)
            lmarks.extend(points)

            rect, midpoint_l = eye_box(points[0], points[1])
            left_eyes.append(intersection(surface, rect))

            rect, midpoint_r = eye_box(points[2], points[3])
            right_eyes.append(intersection(surface, rect))

            midpoints.append(midpoint_l)
            midpoints.append(midpoint_r)
        return left_eyes, right_eyes, midpoints, lmarks


@cv.gapi.kernel(GGetStates)
class GGetStatesImpl:
    """ Custom kernel. Get state of eye - open or closed
    """
    @staticmethod
    def run(eyesl, eyesr):
        """ Сustom kernel executable code

        Params:
        eyesl: result of inference of open-closed-eye network for left eye
        eyesr: result of inference of open-closed-eye network for right eye

        Return:
        States of left eyes and states of right eyes
        """
        out_l_st = [int(st) for eye_l in eyesl for st in (eye_l[:, 0] < eye_l[:, 1]).ravel()]
        out_r_st = [int(st) for eye_r in eyesr for st in (eye_r[:, 0] < eye_r[:, 1]).ravel()]
        return out_l_st, out_r_st


if __name__ == '__main__':
    ARGUMENTS = build_argparser().parse_args()

    # ------------------------Demo's graph------------------------
    g_in = cv.GMat()

    # Detect faces
    face_inputs = cv.GInferInputs()
    face_inputs.setInput('data', g_in)
    face_outputs = cv.gapi.infer('face-detection', face_inputs)
    faces = face_outputs.at('detection_out')

    # Parse faces
    sz = cv.gapi.streaming.size(g_in)
    faces_rc = cv.gapi.parseSSD(faces, sz, 0.5, False, False)

    # Detect poses
    head_inputs = cv.GInferInputs()
    head_inputs.setInput('data', g_in)
    face_outputs = cv.gapi.infer('head-pose', faces_rc, head_inputs)
    angles_y = face_outputs.at('angle_y_fc')
    angles_p = face_outputs.at('angle_p_fc')
    angles_r = face_outputs.at('angle_r_fc')

    # Parse poses
    heads_pos = GProcessPoses.on(angles_y, angles_p, angles_r)

    # Detect landmarks
    landmark_inputs = cv.GInferInputs()
    landmark_inputs.setInput('data', g_in)
    landmark_outputs = cv.gapi.infer('facial-landmarks', faces_rc,
                                     landmark_inputs)
    landmark = landmark_outputs.at('align_fc3')

    # Parse landmarks
    left_eyes, right_eyes, mids, lmarks = GParseEyes.on(landmark, faces_rc, sz)

    # Detect eyes
    eyes_inputs = cv.GInferInputs()
    eyes_inputs.setInput('input.1', g_in)
    eyesl_outputs = cv.gapi.infer('open-closed-eye', left_eyes, eyes_inputs)
    eyesr_outputs = cv.gapi.infer('open-closed-eye', right_eyes, eyes_inputs)
    eyesl = eyesl_outputs.at('19')
    eyesr = eyesr_outputs.at('19')

    # Process eyes states
    l_eye_st, r_eye_st = GGetStates.on(eyesl, eyesr)

    # Gaze estimation
    gaze_inputs = cv.GInferListInputs()
    gaze_inputs.setInput('left_eye_image', left_eyes)
    gaze_inputs.setInput('right_eye_image', right_eyes)
    gaze_inputs.setInput('head_pose_angles', heads_pos)
    gaze_outputs = cv.gapi.infer2('gaze-estimation', g_in, gaze_inputs)
    gaze_vectors = gaze_outputs.at('gaze_vector')

    out = cv.gapi.copy(g_in)
    # ------------------------End of graph------------------------

    comp = cv.GComputation(cv.GIn(g_in), cv.GOut(out,
                                                 faces_rc,
                                                 left_eyes,
                                                 right_eyes,
                                                 gaze_vectors,
                                                 angles_y,
                                                 angles_p,
                                                 angles_r,
                                                 l_eye_st,
                                                 r_eye_st,
                                                 mids,
                                                 lmarks))

    # Networks
    face_net = cv.gapi.ie.params('face-detection', ARGUMENTS.facem,
                                 weight_path(ARGUMENTS.facem), ARGUMENTS.faced)
    head_pose_net = cv.gapi.ie.params('head-pose', ARGUMENTS.headm,
                                      weight_path(ARGUMENTS.headm), ARGUMENTS.headd)
    landmarks_net = cv.gapi.ie.params('facial-landmarks', ARGUMENTS.landm,
                                      weight_path(ARGUMENTS.landm), ARGUMENTS.landd)
    gaze_net = cv.gapi.ie.params('gaze-estimation', ARGUMENTS.gazem,
                                 weight_path(ARGUMENTS.gazem), ARGUMENTS.gazed)
    eye_net = cv.gapi.ie.params('open-closed-eye', ARGUMENTS.eyem,
                                weight_path(ARGUMENTS.eyem), ARGUMENTS.eyed)

    nets = cv.gapi.networks(face_net, head_pose_net, landmarks_net, gaze_net, eye_net)

    # Kernels pack
    kernels = cv.gapi.kernels(GParseEyesImpl, GProcessPosesImpl, GGetStatesImpl)

    # ------------------------Execution part------------------------
    ccomp = comp.compileStreaming(args=cv.gapi.compile_args(kernels, nets))
    if ARGUMENTS.input.isdigit():
        source = cv.gapi.wip.make_capture_src(int(ARGUMENTS.input))
    else:
        source = cv.gapi.wip.make_capture_src(ARGUMENTS.input)

    ccomp.setSource(cv.gin(source))
    ccomp.start()

    frames = 0
    fps = 0
    print('Processing')
    START_TIME = time.time()

    while True:
        start_time_cycle = time.time()
        has_frame, (oimg,
                    outr,
                    l_eyes,
                    r_eyes,
                    outg,
                    out_y,
                    out_p,
                    out_r,
                    out_st_l,
                    out_st_r,
                    out_mids,
                    outl) = ccomp.pull()

        if not has_frame:
            break

        # Draw
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        WHITE = (255, 255, 255)
        BLUE = (255, 0, 0)
        PINK = (255, 0, 255)
        YELLOW = (0, 255, 255)

        M_PI_180 = np.pi / 180
        M_PI_2 = np.pi / 2
        M_PI = np.pi

        FACES_SIZE = len(outr)

        for i, out_rect in enumerate(outr):
            # Face box
            cv.rectangle(oimg, out_rect, WHITE, 1)
            rx, ry, rwidth, rheight = out_rect

            # Landmarks
            lm_radius = int(0.01 * rwidth + 1)
            lmsize = int(len(outl) / FACES_SIZE)
            for j in range(lmsize):
                cv.circle(oimg, outl[j + i * lmsize], lm_radius, YELLOW, -1)

            # Headposes
            yaw = out_y[i]
            pitch = out_p[i]
            roll = out_r[i]
            sin_y = np.sin(yaw[:] * M_PI_180)
            sin_p = np.sin(pitch[:] * M_PI_180)
            sin_r = np.sin(roll[:] * M_PI_180)

            cos_y = np.cos(yaw[:] * M_PI_180)
            cos_p = np.cos(pitch[:] * M_PI_180)
            cos_r = np.cos(roll[:] * M_PI_180)

            axis_length = 0.4 * rwidth
            x_center = int(rx + rwidth / 2)
            y_center = int(ry + rheight / 2)

            # center to right
            cv.line(oimg, [x_center, y_center],
                    [int(x_center + axis_length * (cos_r * cos_y + sin_y * sin_p * sin_r)),
                     int(y_center + axis_length * cos_p * sin_r)],
                    RED, 2)

            # center to top
            cv.line(oimg, [x_center, y_center],
                    [int(x_center + axis_length * (cos_r * sin_y * sin_p + cos_y * sin_r)),
                     int(y_center - axis_length * cos_p * cos_r)],
                    GREEN, 2)

            # center to forward
            cv.line(oimg, [x_center, y_center],
                    [int(x_center + axis_length * sin_y * cos_p),
                     int(y_center + axis_length * sin_p)],
                    PINK, 2)

            scale_box = 0.002 * rwidth
            cv.putText(oimg, "head pose: (y=%0.0f, p=%0.0f, r=%0.0f)" %
                       (np.round(yaw), np.round(pitch), np.round(roll)),
                       [int(rx), int(ry + rheight + 5 * rwidth / 100)],
                       cv.FONT_HERSHEY_PLAIN, scale_box * 2, WHITE, 1)

            # Eyes boxes
            color_l = GREEN if out_st_l[i] else RED
            cv.rectangle(oimg, l_eyes[i], color_l, 1)
            color_r = GREEN if out_st_r[i] else RED
            cv.rectangle(oimg, r_eyes[i], color_r, 1)

            # Gaze vectors
            norm_gazes = np.linalg.norm(outg[i][0])
            gaze_vector = outg[i][0] / norm_gazes

            arrow_length = 0.4 * rwidth
            gaze_arrow = [arrow_length * gaze_vector[0], -arrow_length * gaze_vector[1]]
            left_arrow = [int(a+b) for a, b in zip(out_mids[0 + i * 2], gaze_arrow)]
            right_arrow = [int(a+b) for a, b in zip(out_mids[1 + i * 2], gaze_arrow)]
            if out_st_l[i]:
                cv.arrowedLine(oimg, out_mids[0 + i * 2], left_arrow, BLUE, 2)
            if out_st_r[i]:
                cv.arrowedLine(oimg, out_mids[1 + i * 2], right_arrow, BLUE, 2)

            v0, v1, v2 = outg[i][0]

            gaze_angles = [180 / M_PI * (M_PI_2 + np.arctan2(v2, v0)),
                           180 / M_PI * (M_PI_2 - np.arccos(v1 / norm_gazes))]
            cv.putText(oimg, "gaze angles: (h=%0.0f, v=%0.0f)" %
                       (np.round(gaze_angles[0]), np.round(gaze_angles[1])),
                       [int(rx), int(ry + rheight + 12 * rwidth / 100)],
                       cv.FONT_HERSHEY_PLAIN, scale_box * 2, WHITE, 1)

        # Add FPS value to frame
        cv.putText(oimg, "FPS: %0i" % (fps), [int(20), int(40)],
                   cv.FONT_HERSHEY_PLAIN, 2, RED, 2)

        # Show result
        cv.imshow('Gaze Estimation', oimg)
        cv.waitKey(1)

        fps = int(1. / (time.time() - start_time_cycle))
        frames += 1
    EXECUTION_TIME = time.time() - START_TIME
    print('Execution successful')
    print('Mean FPS is ', int(frames / EXECUTION_TIME))
