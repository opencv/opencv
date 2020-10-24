import numpy as np


def generate_anchors(anchor_stride, anchor_ratios, anchor_scales, anchor_number, score_size):
    """
    generate anchors based on predefined configuration
    """
    anchors = np.zeros((anchor_number, 4), dtype=np.float32)
    size = anchor_stride**2
    count = 0
    for ratio in anchor_ratios:
        ws = int(np.sqrt(size * 1. / ratio))
        hs = int(ws * ratio)
        for scale in anchor_scales:
            w = ws * scale
            h = hs * scale
            anchors[count][:] = [0, 0, w, h]
            count += 1

    anchors = np.tile(anchors, score_size**2).reshape((-1, 4))
    ori = - (score_size // 2) * anchor_stride
    xx, yy = np.meshgrid([ori + anchor_stride * dx for dx in range(score_size)],
                         [ori + anchor_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_number, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_number, 1)).flatten()
    anchors[:, 0], anchors[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchors
