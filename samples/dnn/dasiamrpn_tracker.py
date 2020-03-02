"""
DaSiamRPN tracker.
Original paper: https://arxiv.org/abs/1808.06048
Link to original repo: https://github.com/foolwood/DaSiamRPN
Link to onnx models: https://drive.google.com/drive/folders/1UuwZsgoOVJfwHdE7rS8dtG4UvYfr8OGi?usp=sharing
"""

import numpy as np
import cv2 as cv
import argparse

#function for cropping image
def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2

    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    r, c, k = im.shape

    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    return im_patch.astype(float)

def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0

    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)], [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

    return anchor

#parameters for another functions
class TrackerConfig(object):
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 271
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    size_checking = True

#initialization of network
def SiamRPN_init(im, target_pos, target_sz, net, kernel_r1, kernel_cls1):
    state = dict()
    p = TrackerConfig()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.size_checking:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            raise AssertionError("Initializing bounding box is too small - try to restart tracker  with larger bounding box")

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    avg_chans = np.mean(im, axis=(0, 1))
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))

    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)    
    z_crop = z_crop.transpose(2, 0, 1).reshape(1, 3, 127, 127).astype(np.float32)

    #"temple" Start
    net.setInput(z_crop)
    z_f = net.forward('63')

    kernel_r1.setInput(z_f)
    r1 = kernel_r1.forward()

    kernel_cls1.setInput(z_f)
    cls1 = kernel_cls1.forward()

    r1 = r1.reshape(20, 256, 4, 4)
    cls1 = cls1.reshape(10, 256 , 4, 4)

    net.setParam(net.getLayerId('65'), 0, r1)
    net.setParam(net.getLayerId('68'), 0, cls1)
    #"temple" End

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))

    window = np.tile(window.flatten(), p.anchor_num)
    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz

    return state

#track function
def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    #region preprocessing
    x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = x_crop.transpose(2, 0, 1).reshape(1, 3, 271, 271).astype(np.float32)

    target_pos, target_sz, score = tracker_eval(net, x_crop, target_pos, target_sz * scale_z, window, scale_z, p)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    # print(state)
    return state

#evaluations with cropped images
def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    net.setInput(x_crop)
    outNames = net.getUnconnectedOutLayersNames()
    outNames = ['66', '68']
    delta, score = net.forward(outNames)

    delta = np.transpose(delta, (1, 2, 3, 0))
    delta = np.ascontiguousarray(delta, dtype = np.float32)
    delta = np.reshape(delta, (4, -1))

    score = np.transpose(score, (1, 2, 3, 0))
    score = np.ascontiguousarray(score, dtype = np.float32)
    score = np.reshape(score, (2, -1))
    score = softmax(score)[1, :]

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    return target_pos, target_sz, score[best_pscore_id]

def softmax(x):
    x_max = x.max(0)
    e_x = np.exp(x - x_max)
    y = e_x / e_x.sum(axis = 0)
    return y

# #function for drawing initial bounding box
# def get_bb(event, x, y, flag, param):
#     global point, cx, cy, w, h
#     if event == cv.EVENT_LBUTTONDOWN:
#         point = [(x, y)]

#     elif event == cv.EVENT_LBUTTONUP:
#         point.append((x, y))
#         cv.rectangle(frame, point[0], point[1], (0, 255, 255), 3)

#         cx = point[0][0] - (point[0][0] - point[1][0]) / 2
#         cy = point[0][1] - (point[0][1]- point[1][1]) / 2
#         w = abs(point[0][0] - point[1][0])
#         h = abs(point[0][1] - point[1][1])

def get_bb(event, x, y, flag, param):
    global point1, point2, cx, cy, w, h, drawing, mark
    if event == cv.EVENT_LBUTTONDOWN:
        if drawing == False:
            drawing = True
            point1 = (x, y)
        else:
            drawing = False
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            point2 = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        cx = point1[0] - (point1[0] - point2[0]) / 2
        cy = point1[1] - (point1[1] - point2[1]) / 2
        w = abs(point1[0] - point2[0])
        h = abs(point1[1] - point2[1])
        mark = False

#Reading paths to onnx files with models from command line when launching tracking.
#--net *absolute path to onnx file with net*
#--kernel_r1 and --kernel_cls1 *absolute paths to onnx files with added convolution layers*
parser = argparse.ArgumentParser(description = "Run tracker")
parser.add_argument("--net", type = str, help = "Full path to onnx model of net")
parser.add_argument("--kernel_r1", type = str, help = "Full path to onnx model of kernel_r1")
parser.add_argument("--kernel_cls1", type = str, help = "Full path to onnx model of kernel_cls1")
args = parser.parse_args()

net = cv.dnn.readNet(args.net)
kernel_r1 = cv.dnn.readNet(args.kernel_r1)
kernel_cls1 = cv.dnn.readNet(args.kernel_cls1)

#read source of video/image sequence
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)

# ret, frame = cap.read()
# cv.namedWindow("DaSiamRPN")
# cv.setMouseCallback("DaSiamRPN", get_bb)
# cv.imshow("DaSiamRPN", frame)
# cv.waitKey(0)

ret, frame = cap.read()
cv.namedWindow("DaSiamRPN")
cv.setMouseCallback("DaSiamRPN", get_bb)
drawing = False
point1 = ()
point2 = ()
mark = True
while mark == True:
    ret, frame = cap.read()
    if point1 and point2:
        cv.rectangle(frame, point1, point2, (0, 255, 255), 3)
    cv.imshow("DaSiamRPN", frame)
    cv.waitKey(1)
    # key = cv.waitKey(1)
    # if key == 27:
    #     break

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
state = SiamRPN_init(frame, target_pos, target_sz, net, kernel_r1, kernel_cls1)

#variables for fps counter
toc = 0
f = 1

#tracking cicle
while (cap.isOpened):

    ret,frame = cap.read()

    f += 1
    tic = cv.getTickCount()

    state = SiamRPN_track(state, frame)

    toc += cv.getTickCount() - tic

    w, h = state['target_sz']
    cx, cy = state['target_pos']

    # cv.rectangle(frame, (int(cx - w // 2), int(cy - h // 2), int(w), int(h)), (0, 255, 255), 3)
    cv.rectangle(frame, (int(cx - w // 2), int(cy - h // 2)), (int(cx - w // 2) + int(w), int(cy - h // 2) + int(h)),(0, 255, 255), 3)

    cv.imshow('DaSiamRPN', frame)

    key = cv.waitKey(1)
    if key == 27:
        print('Tracking Speed {:.1f}fps'.format((f - 1) / (toc / cv.getTickFrequency())))
        break

cap.release()
cv.destroyAllWindows()
