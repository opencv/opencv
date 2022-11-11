# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import numpy as np

class Board:
    def __init__(self, w, h, square_len, euler_limit, t_limit, t_origin=None):
        assert w >= 0 and h >= 0 and square_len >= 0
        assert len(euler_limit) == len(t_limit) == 3
        self.w = w
        self.h = h
        self.square_len = square_len

        self.t_limit = t_limit
        self.euler_limit = np.array(euler_limit, dtype=np.float32)
        colors = [[1,0,0], [0,1,0], [0,0,0], [0,0,1]]
        self.colors_board = np.zeros((w*h, 3))
        self.t_origin = np.array(t_origin, dtype=np.float32)[:,None] if t_origin is not None else None
        for i in range(h):
            for j in range(w):
                if j <= w // 2 and i <= h // 2: color = colors[0]
                elif j <= w // 2 and i > h // 2: color = colors[1]
                elif j > w // 2 and i <= h // 2: color = colors[2]
                else: color = colors[3]
                self.colors_board[i*w+j] = color
        for i in range(3):
            assert len(euler_limit[i]) == len(t_limit[i]) == 2
            self.euler_limit[i] *= (np.pi / 180)

    def isProjectionValid(self, pts_proj):
        """
        projection is valid, if x coordinate of left top corner point is smaller than x of bottom right point, ie do not allow 90 deg rotation of 2D board
        also, if x coordinate of left bottom corner is smaller than x coordinate on top right corner, ie do not allow flip
        pts_proj : 2 x N
        """
        assert pts_proj.ndim == 2 and pts_proj.shape[0] == 2
        # pdb.set_trace()
        return pts_proj[0,0] < pts_proj[0,-1] and pts_proj[0,(self.h-1)*self.w] < pts_proj[0,self.w-1]

class CircleBoard(Board):
    def __init__(self, w, h, square_len, euler_limit, t_limit, t_origin=None):
        super().__init__(w, h, square_len, euler_limit, t_limit, t_origin)
        self.pattern = []
        for row in range(h):
            for col in range(w):
                if row % 2 == 1:
                    self.pattern.append([(col+.5)*square_len, square_len*(row//2+.5), 0])
                else:
                    self.pattern.append([col*square_len, (row//2)*square_len, 0])
        self.pattern = np.array(self.pattern, dtype=np.float32).T

class CheckerBoard(Board):
    def __init__(self, w, h, square_len, euler_limit, t_limit, t_origin=None):
        super().__init__(w, h, square_len, euler_limit, t_limit, t_origin)
        self.pattern = np.zeros((w * h, 3), np.float32)
        # https://stackoverflow.com/questions/37310210/camera-calibration-with-opencv-how-to-adjust-chessboard-square-size
        self.pattern[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * square_len  # only for (x,y,z=0)
        self.pattern = self.pattern.T
