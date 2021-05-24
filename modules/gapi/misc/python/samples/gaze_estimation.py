import argparse
import time
import numpy as np
import cv2 as cv

# ------------------------Service operations------------------------
def weight_path(model_path):
    assert model_path.endswith('.xml'), "Wrong topology path was provided"
    return model_path[:-3] + 'bin'


def intersection(surface, rect):
    l_x = max(surface[0], rect[0])
    l_y = max(surface[1], rect[1])
    width = min(surface[0] + surface[2], rect[0] + rect[2]) - l_x
    height = min(surface[1] + surface[3], rect[1] + rect[3]) - l_y
    if width < 0 or height < 0:
        return (0, 0, 0, 0)
    return (l_x, l_y, width, height)


def process_landmarks(r_x, r_y, r_w, r_h, landmarks):
    lmrks = landmarks[0]
    raw_x = lmrks[::2] * r_w + r_x
    raw_y = lmrks[1::2] * r_h + r_y
    return np.array([[int(x), int(y)] for x, y in zip(raw_x, raw_y)])


def eye_box(p_1, p_2, scale=1.8):
    size = np.linalg.norm(p_1 - p_2)
    midpoint = (p_1 + p_2) / 2
    width = scale * size
    height = width
    p_x = midpoint[0] - (width / 2)
    p_y = midpoint[1] - (height / 2)
    return (int(p_x), int(p_y), int(width), int(height)), list(map(int, midpoint))


def build_argparser():
    parser = argparse.ArgumentParser(description='This is an OpenCV-based version of ' +
                                     'Gaze Estimation example')

    parser.add_argument('--input',
                        help='Path to the input video file')
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


# ------------------------Kernels------------------------
@cv.gapi.op('custom.ProcessPoses', in_types=[cv.GArray.GMat,
                                             cv.GArray.GMat,
                                             cv.GArray.GMat],
            out_types=[cv.GArray.GMat])
class ProcessPoses:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1, arr_desc2):
        return cv.empty_array_desc()


@cv.gapi.kernel(ProcessPoses)
class GProcessPosesImpl:
    @staticmethod
    def run(in_ys, in_ps, in_rs):
        out_poses = []
        size = len(in_ys)
        for i in range(size):
            out_poses.append(np.array([in_ys[i][0], in_ps[i][0], in_rs[i][0]]).T)
        return out_poses


@cv.gapi.op('custom.ParseEyes', in_types=[cv.GArray.GMat,
                                          cv.GArray.Rect,
                                          cv.GOpaque.Size],
            out_types=[cv.GArray.Rect,
                       cv.GArray.Rect,
                       cv.GArray.Point,
                       cv.GArray.Point])
class ParseEyes:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1, arr_desc2):
        return cv.empty_array_desc(), cv.empty_array_desc(), \
               cv.empty_array_desc(), cv.empty_array_desc()


@cv.gapi.kernel(ParseEyes)
class GParseEyesImpl:
    @staticmethod
    def run(in_landm_per_face, in_face_rcs, frame_size):
        left_eyes = []
        right_eyes = []
        midpoints = []
        lmarks = []
        num_faces = len(in_landm_per_face)
        surface = (0, 0, *frame_size)
        for i in range(num_faces):
            rect = in_face_rcs[i]
            points = process_landmarks(*rect, in_landm_per_face[i])
            for p in points:
                lmarks.append(p)
            size = int(len(in_landm_per_face[i][0]) / 2)

            rect, midpoint_l = eye_box(lmarks[0 + i * size], lmarks[1 + i * size])
            left_eyes.append(intersection(surface, rect))
            rect, midpoint_r = eye_box(lmarks[2 + i * size], lmarks[3 + i * size])
            right_eyes.append(intersection(surface, rect))
            midpoints += [midpoint_l, midpoint_r]
        return left_eyes, right_eyes, midpoints, lmarks


@cv.gapi.op('custom.GetStates', in_types=[cv.GArray.GMat, cv.GArray.GMat],
            out_types=[cv.GArray.Int, cv.GArray.Int])
class GetStates:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1):
        return cv.empty_array_desc(), cv.empty_array_desc()


@cv.gapi.kernel(GetStates)
class GGetStatesImpl:
    @staticmethod
    def run(eyesl, eyesr):
        size = len(eyesl)
        out_l_st = []
        out_r_st = []
        for i in range(size):
            for st in eyesl[i]:
                out_l_st += [1 if st[0] < st[1] else 0]
            for st in eyesr[i]:
                out_r_st += [1 if st[0] < st[1] else 0]
        return out_l_st, out_r_st

# FIXME: the operation should be wrapped soon
@cv.gapi.op('custom.Copy', in_types=[cv.GMat], out_types=[cv.GMat])
class Copy:
    @staticmethod
    def outMeta(desc):
        return desc


@cv.gapi.kernel(Copy)
class GCopyImpl:
    @staticmethod
    def run(input):
        return input


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
    heads_pos = ProcessPoses.on(angles_y, angles_p, angles_r)

    # Detect landmarks
    landmark_inputs = cv.GInferInputs()
    landmark_inputs.setInput('data', g_in)
    landmark_outputs = cv.gapi.infer('facial-landmarks', faces_rc,
                                     landmark_inputs)
    landmark = landmark_outputs.at('align_fc3')

    # Parse landmarks
    left_eyes, right_eyes, mids, lmarks = ParseEyes.on(landmark, faces_rc, sz)

    # Detect eyes
    eyes_inputs = cv.GInferInputs()
    eyes_inputs.setInput('input.1', g_in)
    eyesl_outputs = cv.gapi.infer('open-closed-eye', left_eyes, eyes_inputs)
    eyesr_outputs = cv.gapi.infer('open-closed-eye', right_eyes, eyes_inputs)
    eyesl = eyesl_outputs.at('19')
    eyesr = eyesr_outputs.at('19')

    # Process eyes states
    l_eye_st, r_eye_st = GetStates.on(eyesl, eyesr)

    # Gaze estimation
    gaze_inputs = cv.GInferListInputs()
    gaze_inputs.setInput('left_eye_image', left_eyes)
    gaze_inputs.setInput('right_eye_image', right_eyes)
    gaze_inputs.setInput('head_pose_angles', heads_pos)
    gaze_outputs = cv.gapi.infer2('gaze-estimation', g_in, gaze_inputs)
    gaze_vectors = gaze_outputs.at('gaze_vector')

    out = Copy.on(g_in)
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

    nets = cv.gapi.networks(face_net, head_pose_net, landmarks_net, gaze_net,
                            eye_net)

    # Kernels pack
    kernels = cv.gapi.kernels(GParseEyesImpl, GProcessPosesImpl, GGetStatesImpl,
                              GCopyImpl)

    # ------------------------Execution part------------------------
    ccomp = comp.compileStreaming(args=cv.gapi.compile_args(kernels, nets))
    source = cv.gapi.wip.make_capture_src(ARGUMENTS.input)
    ccomp.setSource(source)
    ccomp.start()

    frames = 0
    fps = 0
    state = True  # pull() result
    print('Processing')
    START_TIME = time.time()

    while state:
        start_time_cycle = time.time()
        state, (oimg,
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

        for i in range(FACES_SIZE):
            # Face box
            cv.rectangle(oimg, outr[i], WHITE, 1)
            rx, ry, rwidth, rheight = outr[i]

            # Landmarks
            lmRadius = int(0.01 * rwidth + 1)
            lmsize = int(len(outl) / FACES_SIZE)
            for j in range(lmsize):
                cv.circle(oimg, outl[j + i * lmsize], lmRadius, YELLOW, -1)

            # Headposes
            yaw = out_y[i]
            pitch = out_p[i]
            roll = out_r[i]
            sinY = np.sin(yaw[:] * M_PI_180)
            sinP = np.sin(pitch[:] * M_PI_180)
            sinR = np.sin(roll[:] * M_PI_180)

            cosY = np.cos(yaw[:] * M_PI_180)
            cosP = np.cos(pitch[:] * M_PI_180)
            cosR = np.cos(roll[:] * M_PI_180)

            axisLength = 0.4 * rwidth
            xCenter = int(rx + rwidth / 2)
            yCenter = int(ry + rheight / 2)

            # center to right
            cv.line(oimg, [xCenter, yCenter],
                    [int(xCenter + axisLength * (cosR * cosY + sinY * sinP * sinR)),
                     int(yCenter + axisLength * cosP * sinR)],
                    RED, 2)

            # center to top
            cv.line(oimg, [xCenter, yCenter],
                    [int(xCenter + axisLength * (cosR * sinY * sinP + cosY * sinR)),
                     int(yCenter - axisLength * cosP * cosR)],
                    GREEN, 2)

            # center to forward
            cv.line(oimg, [xCenter, yCenter],
                    [int(xCenter + axisLength * sinY * cosP),
                     int(yCenter + axisLength * sinP)],
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
            normGazes = np.linalg.norm(outg[i][0])
            gazeVector = outg[i][0] / normGazes

            arrowLength = 0.4 * rwidth
            gazeArrow = [arrowLength * gazeVector[0], -arrowLength * gazeVector[1]]
            left_arrow = [int(a+b) for a, b in zip(out_mids[0 + i * 2], gazeArrow)]
            right_arrow = [int(a+b) for a, b in zip(out_mids[1 + i * 2], gazeArrow)]
            if out_st_l[i]:
                cv.arrowedLine(oimg, out_mids[0 + i * 2], left_arrow, BLUE, 2)
            if out_st_r[i]:
                cv.arrowedLine(oimg, out_mids[1 + i * 2], right_arrow, BLUE, 2)

            v0, v1, v2 = outg[i][0]

            gazeAngles = [180 / M_PI * (M_PI_2 + np.arctan2(v2, v0)),
                          180 / M_PI * (M_PI_2 - np.arccos(v1 / normGazes))]
            cv.putText(oimg, "gaze angles: (h=%0.0f, v=%0.0f)" %
                       (np.round(gazeAngles[0]), np.round(gazeAngles[1])),
                       [int(rx), int(ry + rheight + 12 * rwidth / 100)],
                       cv.FONT_HERSHEY_PLAIN, scale_box * 2, WHITE, 1)

        # Add FPS value to frame
        cv.putText(oimg, "FPS: %0i" % (fps), [int(144), int(94)],
                   cv.FONT_HERSHEY_PLAIN, 2, RED, 2)

        # Show result
        cv.imshow('Gaze Estimation', oimg)

        fps = int(1. / (time.time() - start_time_cycle))
        frames += 1
    ALL_TIME = time.time() - START_TIME
    print('Execution successful')
    print('Mean FPS is ', int(frames / ALL_TIME))
