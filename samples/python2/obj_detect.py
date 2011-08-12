import numpy as np
import cv2

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def detect_turned(img, cascade):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)

    img_t = cv2.transpose(img)
    img_cw = cv2.flip(img_t, 1)
    img_ccw = cv2.flip(img_t, 0)
    r = detect(img, cascade)
    r_cw = detect(img_cw, cascade)
    r_ccw = detect(img_ccw, cascade)

    h, w = img.shape[:2]
    rects = []
    rects += [(x1, y1, x2, y2, 1, 0) for x1, y1, x2, y2 in r]
    rects += [(y1, h-x1-1, y2, h-x2-1, 0, -1) for x1, y1, x2, y2 in r_cw]
    rects += [(w-y1-1, x1, w-y2-1, x2, 0,  1) for x1, y1, x2, y2 in r_ccw]
    return rects

def process_image(fn, cascade, extract_faces=True):
    img = cv2.imread(fn)
    h, w = img.shape[:2]
    scale = max(h, w) / 512.0
    small = cv2.resize(img, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_AREA)
    rects = detect_turned(small, cascade)

    for i, (x1, y1, x2, y2, vx, vy) in enumerate(rects):
        cv2.rectangle(small, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(small, (x1, y1), 2, (0, 0, 255), -1)
        cv2.putText(small, str(i), ((x1+x2)/2, (y1+y2)/2), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))

    rects = np.float32(rects).reshape(-1,6)
    rects[:,:4] = np.around(rects[:,:4]*scale)

    faces = []
    if extract_faces:
        path, name, ext = splitfn(fn)
        face_sz = 256
        for i, r in enumerate(rects):
            p1, p2, u = r.reshape(3, 2)
            v = np.float32( [-u[1], u[0]] )
            w = np.abs(p2-p1).max()
            fscale = w / face_sz
            p0 = 0.5*(p1+p2 - w*(u+v))
            M = np.float32([u*fscale, v*fscale, p0]).T
            face = cv2.warpAffine(img, M, (face_sz, face_sz), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA)
            faces.append(face)

    return small, rects, faces
    
    

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob
    from common import splitfn, image_extensions

    args, img_args = getopt.getopt(sys.argv[1:], '', ['cascade=', 'outdir='])
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    outdir = args.get('--outdir')
    
    img_list = []
    if len(img_args) == 0:
        img_list = ['../cpp/lena.jpg']
    else:
        for mask in img_args:
            img_list.extend(glob(mask))
    img_list = [fn for fn in img_list if splitfn(fn)[-1].lower() in image_extensions]

    cascade = cv2.CascadeClassifier(cascade_fn)

    for i, fn in enumerate(img_list):
        print '%d / %d   %s' % (i+1, len(img_list), fn),
        vis, rects, faces = process_image(fn, cascade)
        if len(faces) > 0 and outdir is not None:
            path, name, ext = splitfn(fn)
            cv2.imwrite('%s/%s_all.bmp' % (outdir, name), vis)
            for face_i, face in enumerate(faces):
                cv2.imwrite('%s/%s_obj%02d.bmp' % (outdir, name, face_i), face)
        print ' - %d object(s) found' % len(faces)
        cv2.imshow('img', vis)
        cv2.waitKey(50)
    cv2.waitKey()
