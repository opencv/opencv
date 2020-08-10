import argparse
import cv2 as cv
import numpy as np
import os

"""
Link to original paper : https://arxiv.org/abs/1812.11703
Link to original repo  : https://github.com/STVIR/pysot

You can download the pre-trained weights of the Tracker Model from https://drive.google.com/file/d/1PBtRDiWAIaGthMKdzyJL9qYC6ODGnXk5/view?usp=sharing
You can download the backbone model (ResNet50 with target input) from https://drive.google.com/file/d/1jwguUwfvBa-EbDXRDfuyCrvqLPZcFZo0/view?usp=sharing
You can download the backbone model (ResNet50 with search input) from https://drive.google.com/file/d/1L6kxi_WkdH__kfrPC_R-ZsaPD4ky52tI/view?usp=sharing
You can download the neck model 1 (Adjusted Layers 1) from https://drive.google.com/file/d/1690vawz9b76Bwskg3CXtBb1Fd59qBsRw/view?usp=sharing
You can download the neck model 2 (Adjusted Layers 2) from https://drive.google.com/file/d/1E7nX5MzSzVpw6a2Mm5krxp0reLrO1-je/view?usp=sharing
You can download the head model (RPN Head) from https://drive.google.com/file/d/15Sgh1YwdH_fCnbTzhsU-HcFKpSMjPMLY/view?usp=sharing
"""

class ModelBuilder():
    """ This class generates the SiamRPN++ Tracker Model by using Imported ONNX Nets
    """
    def __init__(self, target_net, search_net, rpn_head):
        super(ModelBuilder, self).__init__()
        # Build the target branch
        self.target_net = target_net
        # Build the search branch
        self.search_net = search_net
        # Build RPN_Head
        self.rpn_head = rpn_head

    def template(self, z):
        """ Takes the template of size (1, 1, 127, 127) as an input to generate kernel
        """
        self.target_net.setInput(z)
        outNames = ['output_1', 'output_2', 'output_3']
        self.zfs_1, self.zfs_2, self.zfs_3 = self.target_net.forward(outNames)

    def track(self, x):
        """ Takes the search of size (1, 1, 255, 255) as an input to generate classification score and bounding box regression
        """
        self.search_net.setInput(x)
        outNames = ['output_1', 'output_2', 'output_3']
        xfs_1, xfs_2, xfs_3 = self.search_net.forward(outNames)
        self.rpn_head.setInput(np.stack([self.zfs_1, self.zfs_2, self.zfs_3]), 'input_1')
        self.rpn_head.setInput(np.stack([xfs_1, xfs_2, xfs_3]), 'input_2')
        outNames = ['output_1', 'output_2']
        cls, loc = self.rpn_head.forward(outNames)
        return {'cls': cls, 'loc': loc}

class Anchors:
    """ This class generate anchors.
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size
        self.anchor_num = len(self.scales) * len(self.ratios)
        self.anchors = self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride**2
        count = 0
        for r in self.ratios:
            ws = int(np.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1
        return anchors

class SiamRPNTracker:
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.anchor_stride = 8
        self.anchor_ratios = [0.33, 0.5, 1, 2, 3]
        self.anchor_scales = [8]
        self.track_base_size = 8
        self.track_context_amount = 0.5
        self.track_exemplar_size = 127
        self.track_instance_size = 255
        self.track_lr = 0.4
        self.track_penalty_k = 0.04
        self.track_window_influence = 0.44
        self.score_size = (self.track_instance_size - self.track_exemplar_size) // \
                          self.anchor_stride + 1 + self.track_base_size
        self.anchor_num = len(self.anchor_ratios) * len(self.anchor_scales)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        Args:
            im:         bgr based input image frame
            pos:        position of the center of the frame
            model_sz:   exemplar / target image size
            s_z:        original / search image size
            avg_chans:  channel average
        Return:
            im_patch:   sub_windows for the given image input
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
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
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        return im_patch

    def generate_anchor(self, score_size):
        """
        Args:
            im:         bgr based input image frame
            pos:        position of the center of the frame
            model_sz:   exemplar / target image size
            s_z:        original / search image size
            avg_chans:  channel average
        Return:
            anchor:     anchors for pre-determined values of stride, ratio, and scale
        """
        anchors = Anchors(self.anchor_stride, self.anchor_ratios, self.anchor_scales)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        """
        Args:
            delta:      localisation
            anchor:     anchor of pre-determined anchor size
        Return:
            delta:      prediction of bounding box
        """
        delta_transpose = np.transpose(delta, (1, 2, 3, 0))
        delta_contig = np.ascontiguousarray(delta_transpose)
        delta = delta_contig.reshape(4, -1)
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _softmax(self, x):
        """
        Softmax in the direction of the depth of the layer
        """
        x.astype(dtype=np.float32)
        x_max = x.max(axis=1)[:, np.newaxis]
        e_x = np.exp(x-x_max)
        div = np.sum(e_x, axis=1)[:, np.newaxis]
        y = e_x / div
        return y

    def _convert_score(self, score):
        """
        Args:
            cls:        score
        Return:
            cls:        score for cls
        """
        score_transpose = np.transpose(score, (1, 2, 3, 0))
        score_con = np.ascontiguousarray(score_transpose)
        score_view = score_con.reshape(2, -1)
        score = np.transpose(score_view, (1, 0))
        score = self._softmax(score)
        return score[:,1]

    def _bbox_clip(self, cx, cy, width, height, boundary):
        """
        Adjusting the bounding box
        """
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        Args:
            img(np.ndarray):    bgr based input image frame
            bbox: (x,y,w,h):    bounding box
        """
        x,y,h,w = bbox
        self.center_pos = np.array([x + (h - 1) / 2, y + (w - 1) / 2])
        self.size = np.array([h, w])
        w_z = self.size[0] + self.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.track_context_amount * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos, self.track_exemplar_size, s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        Args:
            img(np.ndarray): BGR image
        Return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + self.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.track_context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.track_exemplar_size / s_z
        s_x = s_z * (self.track_instance_size / self.track_exemplar_size)
        x_crop = self.get_subwindow(img, self.center_pos, self.track_instance_size, round(s_x), self.channel_average)
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.track_penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.track_window_influence) + \
                 self.window * self.track_window_influence
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.track_lr

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = score[best_idx]
        return {'bbox': bbox, 'best_score': best_score}

def get_frames(video_name):
    """
    Args:
        Path to input video frame
    Return:
        Frame
    """
    cap = cv.VideoCapture(video_name if video_name else 0)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break

def main():
    """ Sample SiamRPN Tracker
    """
    parser = argparse.ArgumentParser(description='Use this script to run SiamRPN++ Visual Tracker',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--target_net', type=str, default='target_net.onnx', help='Path to part of SiamRPN++ ran on target frame.')
    parser.add_argument('--search_net', type=str, default='search_net.onnx', help='Path to part of SiamRPN++ ran on search frame.')
    parser.add_argument('--rpn_head', type=str, default='rpn_head.onnx', help='Path to RPN Head ONNX model.')
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.input_video):
        raise OSError("Input video file does not exist")
    if not os.path.isfile(args.target_net):
        raise OSError("Target Net does not exist")
    if not os.path.isfile(args.search_net):
        raise OSError("Search Net does not exist")
    if not os.path.isfile(args.rpn_head):
        raise OSError("RPN Head Net does not exist")

    #Load the Networks
    target_net = cv.dnn.readNetFromONNX(args.target_net)
    search_net = cv.dnn.readNetFromONNX(args.search_net)
    rpn_head = cv.dnn.readNetFromONNX(args.rpn_head)
    model = ModelBuilder(target_net, search_net, rpn_head)
    tracker = SiamRPNTracker(model)

    first_frame = True
    cv.namedWindow('SiamRPN++ Tracker', cv.WINDOW_AUTOSIZE)
    for frame in get_frames(args.input_video):
        if first_frame:
            try:
                init_rect = cv.selectROI('SiamRPN++ Tracker', frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            x,y,w,h = bbox
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.imshow('SiamRPN++ Tracker', frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

if __name__ == '__main__':
    main()
