import numpy as np
import cv2

def to_list(a):
    return [tuple(p) for p in a]

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
    Rt = np.zeros((3, 4))
    Rt[:,:3] = [right, down, fwd]
    Rt[:,3] = -np.dot(Rt[:,:3], eye)
    return Rt

def mtx2rvec(R):
    pass
    

if __name__ == '__main__':
    import cv2
    from time import clock

    '''
    w, h = 640, 480
    while True:
        img = np.zeros((h, w, 3), np.uint8)
        t = clock()
        eye = [5*cos(t), 5*sin(t), 3]
        Rt = lookat(eye, [0, 0, 0])
    '''



    eye = [1, -4, 3]
    target = [0, 0, 0]
    Rt = lookat(eye, [0, 0, 0])
    print Rt
    p = [0, 0, 0]
    print cv2.transform(np.float64([[p]]), Rt)

    print cv2.SVDecomp(Rt[:,:3])

