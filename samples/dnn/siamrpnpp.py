import cv2 as cv
import glob
import numpy as np
import os

"""
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
    def __init__(self, backbone_search, backbone_target, neck_1, neck_2, rpn_head):
        super(ModelBuilder, self).__init__()
        # Build Backbone Model
        self.backbone_search = backbone_search
        self.backbone_target = backbone_target
        # Build Adjusted Layer
        self.neck_1 = neck_1
        self.neck_2 = neck_2
        # Build RPN_Head
        self.rpn_head = rpn_head

    def template(self, z):
        """ Takes the template of size (1, 1, 127, 127) as an input to generate kernel
        """
        self.backbone_target.setInput(z)
        outNames = ['output_1', 'output_2', 'output_3']
        zf_1, zf_2, zf_3 = self.backbone_target.forward(outNames)
        self.neck_1.setInput(zf_1, 'input_1')
        self.neck_1.setInput(zf_2, 'input_2')
        self.neck_1.setInput(zf_3, 'input_3')
        zfs_1, zfs_2, zfs_3 = self.neck_1.forward(outNames)
        self.zfs_1 = zfs_1
        self.zfs_2 = zfs_2
        self.zfs_3 = zfs_3

    def track(self, x):
        """ Takes the search of size (1, 1, 255, 255) as an input to generate classification score and bounding box regression
        """
        self.backbone_search.setInput(x)
        outNames = ['output_1', 'output_2', 'output_3']
        xf_1, xf_2, xf_3 = self.backbone_search.forward(outNames)
        self.neck_2.setInput(xf_1, 'input_1')
        self.neck_2.setInput(xf_2, 'input_2')
        self.neck_2.setInput(xf_3, 'input_3')
        xfs_1, xfs_2, xfs_3 = self.neck_2.forward(outNames)
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
        self.anchors = None
        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride**2
        count = 0
        for r in self.ratios:
            ws = int(np.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1

def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        corner: np.array (4*N)
    Return:
        np.array (4 * N)
    """
    x1, y1, x2, y2 = corner[:4]
    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: np.array (4 * N)
    Return:
        np.array (4 * N)
    """
    x, y, w, h = center[0], center[1], center[2], center[3]
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return x1, y1, x2, y2

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
    if not video_name:
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv.imread(img)
            yield frame

def main():
    """ Sample SiamRPN Tracker
    """
    backbone_search = cv.dnn.readNetFromONNX('resnet_search.onnx')
    backbone_target = cv.dnn.readNetFromONNX('resnet_target.onnx')
    neck_1 = cv.dnn.readNetFromONNX('neck_1.onnx')
    neck_2 = cv.dnn.readNetFromONNX('neck_2.onnx')
    rpn_head = cv.dnn.readNetFromONNX('rpn_head.onnx')
    model = ModelBuilder(backbone_search, backbone_target, neck_1, neck_2, rpn_head)
    tracker = SiamRPNTracker(model)
    first_frame = True
    video_name = 'bag.avi'
    cv.namedWindow(video_name, cv.WND_PROP_FULLSCREEN)
    for frame in get_frames(video_name):
        if first_frame:
            try:
                init_rect = cv.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)
        cv.imshow(video_name, frame)
        cv.waitKey(40)

if __name__ == '__main__':
    main()
