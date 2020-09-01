"""
DaSiamRPN tracker.
Original paper: https://arxiv.org/abs/1808.06048
Link to original repo: https://github.com/foolwood/DaSiamRPN
Links to onnx models:
network:     https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
kernel_r1:   https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
kernel_cls1: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0
"""

import numpy as np
import cv2 as cv
import argparse
import sys

class DaSiamRPNTracker:
    # Initialization of used values, initial bounding box, used network
    def __init__(self, net="dasiamrpn_model.onnx", kernel_r1="dasiamrpn_kernel_r1.onnx", kernel_cls1="dasiamrpn_kernel_cls1.onnx"):
        self.windowing = "cosine"
        self.exemplar_size = 127
        self.instance_size = 271
        self.total_stride = 8
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1
        self.context_amount = 0.5
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8, ]
        self.anchor_num = len(self.ratios) * len(self.scales)
        self.penalty_k = 0.055
        self.window_influence = 0.42
        self.lr = 0.295
        self.score = []
        if self.windowing == "cosine":
            self.window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.windowing == "uniform":
            self.window = np.ones((self.score_size, self.score_size))
        self.window = np.tile(self.window.flatten(), self.anchor_num)
        # Loading network`s and kernel`s models
        self.net = cv.dnn.readNet(net)
        self.kernel_r1 = cv.dnn.readNet(kernel_r1)
        self.kernel_cls1 = cv.dnn.readNet(kernel_cls1)

    def init(self, im, init_bb):
        target_pos, target_sz = np.array([init_bb[0], init_bb[1]]), np.array([init_bb[2], init_bb[3]])
        self.im_h = im.shape[0]
        self.im_w = im.shape[1]
        self.target_pos = target_pos
        self.target_sz = target_sz
        self.avg_chans = np.mean(im, axis=(0, 1))

        # When we trying to generate ONNX model from the pre-trained .pth model
        # we are using only one state of the network. In our case used state
        # with big bounding box, so we were forced to add assertion for
        # too small bounding boxes - current state of the network can not
        # work properly with such small bounding boxes
        if ((self.target_sz[0] * self.target_sz[1]) / float(self.im_h * self.im_w)) < 0.004:
            raise AssertionError(
        "Initializing BB is too small-try to restart tracker with larger BB")

        self.anchor = self.__generate_anchor()
        wc_z = self.target_sz[0] + self.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        z_crop = self.__get_subwindow_tracking(im, self.exemplar_size, s_z)
        z_crop = z_crop.transpose(2, 0, 1).reshape(1, 3, 127, 127).astype(np.float32)
        self.net.setInput(z_crop)
        z_f = self.net.forward('63')
        self.kernel_r1.setInput(z_f)
        r1 = self.kernel_r1.forward()
        self.kernel_cls1.setInput(z_f)
        cls1 = self.kernel_cls1.forward()
        r1 = r1.reshape(20, 256, 4, 4)
        cls1 = cls1.reshape(10, 256 , 4, 4)
        self.net.setParam(self.net.getLayerId('65'), 0, r1)
        self.net.setParam(self.net.getLayerId('68'), 0, cls1)

    # Сreating anchor for tracking bounding box
    def __generate_anchor(self):
        self.anchor = np.zeros((self.anchor_num, 4),  dtype = np.float32)
        size = self.total_stride * self.total_stride
        count = 0

        for ratio in self.ratios:
            ws = int(np.sqrt(size / ratio))
            hs = int(ws * ratio)
            for scale in self.scales:
                wws = ws * scale
                hhs = hs * scale
                self.anchor[count] = [0, 0, wws, hhs]
                count += 1

        score_sz = int(self.score_size)
        self.anchor = np.tile(self.anchor, score_sz * score_sz).reshape((-1, 4))
        ori = - (score_sz / 2) * self.total_stride
        xx, yy = np.meshgrid([ori + self.total_stride * dx for dx in range(score_sz)], [ori + self.total_stride * dy for dy in range(score_sz)])
        xx, yy = np.tile(xx.flatten(), (self.anchor_num, 1)).flatten(), np.tile(yy.flatten(), (self.anchor_num, 1)).flatten()
        self.anchor[:, 0], self.anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return self.anchor

    # Function for updating tracker state
    def update(self, im):
        wc_z = self.target_sz[1] + self.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[0] + self.context_amount * sum(self.target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.exemplar_size / s_z
        d_search = (self.instance_size - self.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = round(s_z + 2 * pad)

        # Region preprocessing part
        x_crop = self.__get_subwindow_tracking(im, self.instance_size, s_x)
        x_crop = x_crop.transpose(2, 0, 1).reshape(1, 3, 271, 271).astype(np.float32)
        self.score = self.__tracker_eval(x_crop, scale_z)
        self.target_pos[0] = max(0, min(self.im_w, self.target_pos[0]))
        self.target_pos[1] = max(0, min(self.im_h, self.target_pos[1]))
        self.target_sz[0] = max(10, min(self.im_w, self.target_sz[0]))
        self.target_sz[1] = max(10, min(self.im_h, self.target_sz[1]))

        cx, cy = self.target_pos
        w, h = self.target_sz
        updated_bb = (cx, cy, w, h)
        return True, updated_bb

    # Function for updating position of the bounding box
    def __tracker_eval(self, x_crop, scale_z):
        target_size = self.target_sz * scale_z
        self.net.setInput(x_crop)
        outNames = self.net.getUnconnectedOutLayersNames()
        outNames = ['66', '68']
        delta, score = self.net.forward(outNames)
        delta = np.transpose(delta, (1, 2, 3, 0))
        delta = np.ascontiguousarray(delta, dtype = np.float32)
        delta = np.reshape(delta, (4, -1))
        score = np.transpose(score, (1, 2, 3, 0))
        score = np.ascontiguousarray(score, dtype = np.float32)
        score = np.reshape(score, (2, -1))
        score = self.__softmax(score)[1, :]
        delta[0, :] = delta[0, :] * self.anchor[:, 2] + self.anchor[:, 0]
        delta[1, :] = delta[1, :] * self.anchor[:, 3] + self.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * self.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * self.anchor[:, 3]

        def __change(r):
            return np.maximum(r, 1./r)

        def __sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def __sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        s_c = __change(__sz(delta[2, :], delta[3, :]) / (__sz_wh(target_size)))
        r_c = __change((target_size[0] / target_size[1]) / (delta[2, :] / delta[3, :]))
        penalty = np.exp(-(r_c * s_c - 1.) * self.penalty_k)
        pscore = penalty * score
        pscore = pscore * (1 - self.window_influence) + self.window * self.window_influence
        best_pscore_id = np.argmax(pscore)
        target = delta[:, best_pscore_id] / scale_z
        target_size /= scale_z
        lr = penalty[best_pscore_id] * score[best_pscore_id] * self.lr
        res_x = target[0] + self.target_pos[0]
        res_y = target[1] + self.target_pos[1]
        res_w = target_size[0] * (1 - lr) + target[2] * lr
        res_h = target_size[1] * (1 - lr) + target[3] * lr
        self.target_pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        return score[best_pscore_id]

    def __softmax(self, x):
        x_max = x.max(0)
        e_x = np.exp(x - x_max)
        y = e_x / e_x.sum(axis = 0)
        return y

    # Reshaping cropped image for using in the model
    def __get_subwindow_tracking(self, im, model_size, original_sz):
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = round(self.target_pos[0] - c)
        context_xmax = context_xmin + original_sz - 1
        context_ymin = round(self.target_pos[1] - c)
        context_ymax = context_ymin + original_sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bot_pad = int(max(0., context_ymax - im_sz[0] + 1))
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad
        r, c, k = im.shape

        if any([top_pad, bot_pad, left_pad, right_pad]):
            te_im = np.zeros((
                r + top_pad + bot_pad, c + left_pad + right_pad, k), np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = self.avg_chans
            if bot_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = self.avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = self.avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = self.avg_chans
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_size, original_sz):
            im_patch_original = cv.resize(im_patch_original, (model_size, model_size))
        return im_patch_original

# Sample for using DaSiamRPN tracker
def main():
    parser = argparse.ArgumentParser(description="Run tracker")
    parser.add_argument("--input", type=str, help="Full path to input (empty for camera)")
    parser.add_argument("--net", type=str, default="dasiamrpn_model.onnx", help="Full path to onnx model of net")
    parser.add_argument("--kernel_r1", type=str, default="dasiamrpn_kernel_r1.onnx", help="Full path to onnx model of kernel_r1")
    parser.add_argument("--kernel_cls1", type=str, default="dasiamrpn_kernel_cls1.onnx", help="Full path to onnx model of kernel_cls1")
    args = parser.parse_args()
    point1 = ()
    point2 = ()
    mark = True
    drawing = False
    cx, cy, w, h = 0.0, 0.0, 0, 0
    # Fucntion for drawing during videostream
    def get_bb(event, x, y, flag, param):
        nonlocal point1, point2, cx, cy, w, h, drawing, mark

        if event == cv.EVENT_LBUTTONDOWN:
            if not drawing:
                drawing = True
                point1 = (x, y)
            else:
                drawing = False

        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                point2 = (x, y)

        elif event == cv.EVENT_LBUTTONUP:
            cx = point1[0] - (point1[0] - point2[0]) / 2
            cy = point1[1] - (point1[1] - point2[1]) / 2
            w = abs(point1[0] - point2[0])
            h = abs(point1[1] - point2[1])
            mark = False

    # Creating window for visualization
    cap = cv.VideoCapture(args.input if args.input else 0)
    cv.namedWindow("DaSiamRPN")
    cv.setMouseCallback("DaSiamRPN", get_bb)

    whitespace_key = 32
    while cv.waitKey(40) != whitespace_key:
        has_frame, frame = cap.read()
        if not has_frame:
            sys.exit(0)
        cv.imshow("DaSiamRPN", frame)

    while mark:
        twin = np.copy(frame)
        if point1 and point2:
            cv.rectangle(twin, point1, point2, (0, 255, 255), 3)
        cv.imshow("DaSiamRPN", twin)
        cv.waitKey(40)

    init_bb = (cx, cy, w, h)
    tracker = DaSiamRPNTracker(args.net, args.kernel_r1, args.kernel_cls1)
    tracker.init(frame, init_bb)

    # Tracking loop
    while cap.isOpened():
        has_frame, frame = cap.read()
        if not has_frame:
            sys.exit(0)
        _, new_bb = tracker.update(frame)
        cx, cy, w, h = new_bb
        cv.rectangle(frame, (int(cx - w // 2), int(cy - h // 2)), (int(cx - w // 2) + int(w), int(cy - h // 2) + int(h)),(0, 255, 255), 3)
        cv.imshow("DaSiamRPN", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
