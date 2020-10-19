import numpy as np


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