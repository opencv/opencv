import numpy as np
import cv2

class VideoSynth(object):
    def __init__(self, size=None, noise=0.0, bg = None, **params):
        self.bg = None
        self.frame_size = (640, 480)
        if bg is not None:
            self.bg = cv2.imread(bg, 1)
            h, w = self.bg.shape[:2]
            self.frame_size = (w, h)
            
        if size is not None:
            w, h = map(int, size.split('x'))
            self.frame_size = (w, h)
            self.bg = cv2.resize(bg, self.frame_size)

        self.noise = float(noise)

    def read(self, dst=None):
        w, h = self.frame_size

        if self.bg is None:
            buf = np.zeros((h, w, 3), np.uint8)
        else:
            buf = self.bg.copy()
        if self.noise > 0.0:
            noise = np.zeros((h, w, 3), np.int8)
            cv2.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)
            buf = cv2.add(buf, noise, dtype=cv2.CV_8UC3)
        return True, buf


def create_capture(source):
    '''
      source: <int> or '<int>' or '<filename>' or 'synth:<params>'
    '''
    try: source = int(source)
    except ValueError: pass
    else:
        return cv2.VideoCapture(source)
    source = str(source).strip()
    if source.startswith('synth'):
        ss = filter(None, source.split(':'))
        params = dict( s.split('=') for s in ss[1:] )
        return VideoSynth(**params)
    return cv2.VideoCapture(source)


presets = dict(
    lena = 'synth:bg=../cpp/lena.jpg:noise=0.1'
)

if __name__ == '__main__':
    import sys
    import getopt

    print 'USAGE: video.py [--shotdir <dir>] [source0] [source1] ...'
    print "source: '<int>' or '<filename>' or 'synth:<params>'"
    print

    args, sources = getopt.getopt(sys.argv[1:], '', 'shotdir=')
    args = dict(args)
    shotdir = args.get('--shotdir', '.')
    if len(sources) == 0:
        sources = [ presets['lena'] ]

    print 'Press SPACE to save current frame'

    caps = map(create_capture, sources)
    shot_idx = 0
    while True:
        imgs = []
        for i, cap in enumerate(caps):
            ret, img = cap.read()
            imgs.append(img)
            cv2.imshow('capture %d' % i, img)
        ch = cv2.waitKey(1)
        if ch == 27:
            break
        if ch == ord(' '):
            for i, img in enumerate(imgs):
                fn = '%s/shot_%d_%03d.bmp' % (shotdir, i, shot_idx)
                cv2.imwrite(fn, img)
                print fn, 'saved'
            shot_idx += 1
