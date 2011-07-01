import numpy as np
import cv2
import os

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def homotrans(H, x, y):
    xs = H[0, 0]*x + H[0, 1]*y + H[0, 2]
    ys = H[1, 0]*x + H[1, 1]*y + H[1, 2]
    s  = H[2, 0]*x + H[2, 1]*y + H[2, 2]
    return xs/s, ys/s

def to_rect(a):
    a = np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return np.array(a, np.float64).reshape(2, 2)

def rect2rect_mtx(src, dst):
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = np.float64([[ cx,  0, tx],
                    [  0, cy, ty],
                    [  0,  0,  1]])
    return M


def lookat(eye, target, up = (0, 0, 1)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= anorm(fwd)
    right = np.cross(fwd, up)
    right /= anorm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def mtx2rvec(R):
    w, u, vt = cv2.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:,0]*w[0]    # same as np.dot(R, vt[0])
    c = np.dot(vt[0], p)
    s = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(s, c)

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, linetype=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), linetype=cv2.CV_AA)
