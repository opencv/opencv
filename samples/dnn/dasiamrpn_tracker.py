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
    #initialization of used values, initial bounding box, used network
    def __init__(self, im, target_pos, target_sz, net, kernel_r1, kernel_cls1):
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
        self.im_h = im.shape[0]
        self.im_w = im.shape[1]
        self.target_pos = target_pos
        self.target_sz = target_sz
        self.avg_chans = np.mean(im, axis=(0, 1))
        self.net = net
        self.score = []

        if ((self.target_sz[0] * self.target_sz[1]) / float(self.im_h * self.im_w)) < 0.004:
             raise AssertionError("Initializing BB is too small-try to restart tracker with larger BB")

        self.anchor = self.__generate_anchor()
        wc_z = self.target_sz[0] + self.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        z_crop = self.__get_subwindow_tracking(im, self.exemplar_size, s_z)
        z_crop = z_crop.transpose(2, 0, 1).reshape(1, 3, 127, 127).astype(np.float32)
        self.net.setInput(z_crop)
        z_f = self.net.forward('63')
        kernel_r1.setInput(z_f)
        r1 = kernel_r1.forward()
        kernel_cls1.setInput(z_f)
        cls1 = kernel_cls1.forward()
        r1 = r1.reshape(20, 256, 4, 4)
        cls1 = cls1.reshape(10, 256 , 4, 4)
        self.net.setParam(self.net.getLayerId('65'), 0, r1)
        self.net.setParam(self.net.getLayerId('68'), 0, cls1)

        if self.windowing == "cosine":
            self.window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.windowing == "uniform":
            self.window = np.ones((self.score_size, self.score_size))
        self.window = np.tile(self.window.flatten(), self.anchor_num)

    #creating anchor for tracking bounding box
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

    #track function
    def track(self, im):
        wc_z = self.target_sz[1] + self.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[0] + self.context_amount * sum(self.target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.exemplar_size / s_z
        d_search = (self.instance_size - self.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = round(s_z + 2 * pad)

        #region preprocessing
        x_crop = self.__get_subwindow_tracking(im, self.instance_size, s_x)
        x_crop = x_crop.transpose(2, 0, 1).reshape(1, 3, 271, 271).astype(np.float32)
        self.score = self.__tracker_eval(x_crop, scale_z)
        self.target_pos[0] = max(0, min(self.im_w, self.target_pos[0]))
        self.target_pos[1] = max(0, min(self.im_h, self.target_pos[1]))
        self.target_sz[0] = max(10, min(self.im_w, self.target_sz[0]))
        self.target_sz[1] = max(10, min(self.im_h, self.target_sz[1]))

    #update bounding box position
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

    #evaluations with cropped image
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
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad
        r, c, k = im.shape

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = self.avg_chans
            if bottom_pad:
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

#function for reading paths, bounding box drawing, showing results
def main():
    parser = argparse.ArgumentParser(description="Run tracker")
    parser.add_argument("--net", type=str, default="dasiamrpn_model.onnx", help="Full path to onnx model of net")
    parser.add_argument("--kernel_r1", type=str, default="dasiamrpn_kernel_r1.onnx", help="Full path to onnx model of kernel_r1")
    parser.add_argument("--kernel_cls1", type=str, default="dasiamrpn_kernel_cls1.onnx", help="Full path to onnx model of kernel_cls1")
    parser.add_argument("--input", type=str, help="Full path to input. Do not use if input is camera")
    args = parser.parse_args()
    point1 = ()
    point2 = ()
    mark = True
    drawing = False
    cx, cy, w, h = 0.0, 0.0, 0, 0

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

    #loading network`s and kernel`s models
    net = cv.dnn.readNet(args.net)
    kernel_r1 = cv.dnn.readNet(args.kernel_r1)
    kernel_cls1 = cv.dnn.readNet(args.kernel_cls1)

    #initializing bounding box
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

    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    tracker = DaSiamRPNTracker(frame, target_pos, target_sz, net, kernel_r1, kernel_cls1)

    #tracking loop
    while cap.isOpened():
        has_frame, frame = cap.read()
        if not has_frame:
            sys.exit(0)
        tracker.track(frame)
        w, h = tracker.target_sz
        cx, cy = tracker.target_pos
        cv.rectangle(frame, (int(cx - w // 2), int(cy - h // 2)), (int(cx - w // 2) + int(w), int(cy - h // 2) + int(h)),(0, 255, 255), 3)
        cv.imshow("DaSiamRPN", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
