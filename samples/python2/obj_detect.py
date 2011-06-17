import numpy as np
import cv2, cv
import common

def detect(img, cascade):
    min_size = (20, 20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    rects = cascade.detectMultiScale(img, haar_scale, min_neighbors, haar_flags, min_size)
    if len(rects) == 0:
        return
    rects[:,2:] += rects[:,:2]
    return rects

def detect_turned(img, cascade):
    img_t = cv2.transpose(img)
    img_cw = cv2.flip(img_t, 1)
    img_ccw = cv2.flip(img_t, 0)
    r = detect(img, cascade)
    r_cw = detect(img_cw, cascade)
    r_ccw = detect(img_ccw, cascade)

    h, w = img.shape[:2]
    if r_cw is not None:
        r_cw[:,[0, 2]] = h - r_cw[:,[0, 2]] - 1
        r_cw = r_cw[:,[1,0,3,2]]
    if r_ccw is not None:
        r_ccw[:,[1, 3]] = w - r_ccw[:,[1, 3]] - 1
        r_ccw = r_ccw[:,[1,0,3,2]]
    rects = np.vstack( [a for a in [r, r_cw, r_ccw] if a is not None] )
    return rects

def process_image(fn, cascade):
    pass
    
    

if __name__ == '__main__':
    import sys
    import getopt
    args, img_mask = getopt.getopt(sys.argv[1:], '', ['cascade='])
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)


    img = cv2.imread('test.jpg')
    h, w = img.shape[:2]
    r = 512.0 / max(h, w)
    small = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)
    rects = detect_turned(small, cascade)
    print rects
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(small, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(small, (x1, y1), 2, (0, 0, 255), -1)



    cv2.imshow('img', small)
    cv2.waitKey()

    

'''


    img = cv2.imread('test.jpg')
    h, w = img.shape[:2]


    r = 512.0 / max(h, w)
    small = cv2.resize(img, (w*r, h*r), interpolation=cv2.INTER_AREA)

cv2.imshow('img', small)
cv2.waitKey()

'''