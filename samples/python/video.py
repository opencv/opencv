#!/usr/bin/env python

'''
Video capture sample.

Sample shows how VideoCapture class can be used to acquire video
frames from a camera of a movie file. Also the sample provides
an example of procedural video generation by an object, mimicking
the VideoCapture interface (see Chess class).

'create_capture' is a convenience function for capture creation,
falling back to procedural video in case of error.

Usage:
    video.py [--shotdir <shot path>] [source0] [source1] ...'

    sourceN is an
     - integer number for camera capture
     - name of video file
     - synth:<params> for procedural video

Synth examples:
    synth:bg=lena.jpg:noise=0.1
    synth:class=chess:bg=lena.jpg:noise=0.1:size=640x480

Keys:
    ESC    - exit
    SPACE  - save current frame to <shot path> directory

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import re
import os
import platform
import sys
import threading
from time import time
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # for Python 2 compatibility

# local modules
from tst_scene_render import TestSceneRender
import common

# Configuration
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
QUEUE_SIZE = 10  # For threaded capture

class VideoSynthBase(object):
    def __init__(self, size=None, noise=0.0, bg=None, **params):
        self.bg = None
        self.frame_size = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        if bg is not None:
            self.bg = cv.imread(cv.samples.findFile(bg))
            h, w = self.bg.shape[:2]
            self.frame_size = (w, h)

        if size is not None:
            w, h = map(int, size.split('x'))
            self.frame_size = (w, h)
            if self.bg is not None:
                self.bg = cv.resize(self.bg, self.frame_size)

        self.noise = float(noise)

    def render(self, dst):
        pass

    def read(self, dst=None):
        w, h = self.frame_size

        if self.bg is None:
            buf = np.zeros((h, w, 3), np.uint8)
        else:
            buf = self.bg.copy()

        self.render(buf)

        if self.noise > 0.0:
            noise = np.zeros((h, w, 3), np.int8)
            cv.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)
            buf = cv.add(buf, noise, dtype=cv.CV_8UC3)
        return True, buf

    def isOpened(self):
        return True

    def get(self, propId):
        # Add support for the get method to synthetic video sources
        if propId == cv.CAP_PROP_FRAME_WIDTH:
            return self.frame_size[0]
        elif propId == cv.CAP_PROP_FRAME_HEIGHT:
            return self.frame_size[1]
        elif propId == cv.CAP_PROP_FPS:
            return 30
        return 0

    def set(self, propId, value):
        # Add support for the set method to synthetic video sources
        if propId == cv.CAP_PROP_FRAME_WIDTH:
            self.frame_size = (int(value), self.frame_size[1])
            return True
        elif propId == cv.CAP_PROP_FRAME_HEIGHT:
            self.frame_size = (self.frame_size[0], int(value))
            return True
        return False

class Book(VideoSynthBase):
    def __init__(self, **kw):
        super(Book, self).__init__(**kw)
        backGr = cv.imread(cv.samples.findFile('graf1.png'))
        fgr = cv.imread(cv.samples.findFile('box.png'))
        self.render = TestSceneRender(backGr, fgr, speed=1)

    def read(self, dst=None):
        noise = np.zeros(self.render.sceneBg.shape, np.int8)
        cv.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)

        return True, cv.add(self.render.getNextFrame(), noise, dtype=cv.CV_8UC3)

class Cube(VideoSynthBase):
    def __init__(self, **kw):
        super(Cube, self).__init__(**kw)
        self.render = TestSceneRender(cv.imread(cv.samples.findFile('pca_test1.jpg')), deformation=True, speed=1)

    def read(self, dst=None):
        noise = np.zeros(self.render.sceneBg.shape, np.int8)
        cv.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)

        return True, cv.add(self.render.getNextFrame(), noise, dtype=cv.CV_8UC3)

class Chess(VideoSynthBase):
    def __init__(self, **kw):
        super(Chess, self).__init__(**kw)

        w, h = self.frame_size

        self.grid_size = sx, sy = 10, 7
        white_quads = []
        black_quads = []
        for i, j in np.ndindex(sy, sx):
            q = [[j, i, 0], [j+1, i, 0], [j+1, i+1, 0], [j, i+1, 0]]
            [white_quads, black_quads][(i + j) % 2].append(q)
        self.white_quads = np.float32(white_quads)
        self.black_quads = np.float32(black_quads)

        fx = 0.9
        self.K = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0,0.0,      1.0]])

        self.dist_coef = np.float64([-0.2, 0.1, 0, 0])
        self.t = 0

    def draw_quads(self, img, quads, color=(0, 255, 0)):
        img_quads = cv.projectPoints(quads.reshape(-1, 3), self.rvec, self.tvec, self.K, self.dist_coef)[0]
        img_quads.shape = quads.shape[:2] + (2,)
        for q in img_quads:
            cv.fillConvexPoly(img, np.int32(q*4), color, cv.LINE_AA, shift=2)

    def render(self, dst):
        t = self.t
        self.t += 1.0/30.0

        sx, sy = self.grid_size
        center = np.array([0.5*sx, 0.5*sy, 0.0])
        phi = np.pi/3 + np.sin(t*3)*np.pi/8
        c, s = np.cos(phi), np.sin(phi)
        ofs = np.array([np.sin(1.2*t), np.cos(1.8*t), 0]) * sx * 0.2
        eye_pos = center + np.array([np.cos(t)*c, np.sin(t)*c, s]) * 15.0 + ofs
        target_pos = center + ofs

        R, self.tvec = common.lookat(eye_pos, target_pos)
        self.rvec = common.mtx2rvec(R)

        self.draw_quads(dst, self.white_quads, (245, 245, 245))
        self.draw_quads(dst, self.black_quads, (10, 10, 10))


class ThreadedVideoCapture:
    """
    Class for threaded video capture to improve performance
    """
    def __init__(self, source=0, width=None, height=None):
        self.cap = cv.VideoCapture(source)
        self.queue = Queue(maxsize=QUEUE_SIZE)
        self.stopped = False

        # Set resolution if provided
        if width is not None and width > 0:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        if height is not None and height > 0:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        # Platform-specific optimizations - only apply if camera source (integer)
        if isinstance(source, int):
            try:
                system = platform.system()
                if system == 'Windows':
                    # DirectShow optimizations for Windows
                    self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
                elif system == 'Linux':
                    # V4L2 optimizations for Linux
                    self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
                elif system == 'Darwin':  # macOS
                    # AVFoundation optimizations for macOS
                    self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
            except Exception as e:
                print(f"Warning: could not set codec format: {e}")

        # Start thread
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.queue.put((ret, frame))
            else:
                # Small delay to prevent CPU overload when queue is full
                if hasattr(time, 'sleep'):
                    time.sleep(0.001)

    def read(self):
        try:
            return self.queue.get(timeout=1.0)
        except Empty:
            return (False, None)

    def isOpened(self):
        return self.cap.isOpened() and not self.stopped

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
        # Clear the queue to prevent deadlocks
        while not self.queue.empty():
            try:
                self.queue.get(block=False)
            except Empty:
                break

    def get(self, propId):
        return self.cap.get(propId)

    def set(self, propId, value):
        return self.cap.set(propId, value)


classes = dict(chess=Chess, book=Book, cube=Cube)

presets = dict(
    empty = 'synth:',
    lena = 'synth:bg=lena.jpg:noise=0.1',
    chess = 'synth:class=chess:bg=lena.jpg:noise=0.1:size=640x480',
    book = 'synth:class=book:bg=graf1.png:noise=0.1:size=640x480',
    cube = 'synth:class=cube:bg=pca_test1.jpg:noise=0.0:size=640x480'
)


def create_capture(source=0, fallback=presets['chess'], threaded=True):
    '''source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    '''
    source = str(source).strip()
    params = {}

    # Handle paths across different platforms
    if os.path.exists(source):
        source = os.path.abspath(source)
    else:
        chunks = source.split(':')
        source = chunks[0]
        if len(chunks) > 1:
            params = dict(s.split('=') for s in chunks[1:] if '=' in s)

    # Convert source to integer if it represents a number
    try:
        source = int(source)
    except ValueError:
        pass

    # Create the appropriate capture object
    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try:
            cap = Class(**params)
        except Exception as e:
            print(f"Error creating synthetic source: {e}")
            pass
    else:
        width = int(params.get('width', DEFAULT_WIDTH)) if 'width' in params else None
        height = int(params.get('height', DEFAULT_HEIGHT)) if 'height' in params else None

        # Use threaded capture for better performance if requested and if it's a camera source
        if threaded and isinstance(source, int):
            try:
                cap = ThreadedVideoCapture(source, width, height)
                if not cap.isOpened():
                    raise Exception("Failed to open threaded capture")
            except Exception as e:
                print(f"Warning: could not use threaded capture: {e}")
                cap = None

        # Fall back to regular capture if threaded didn't work or wasn't requested
        if cap is None:
            try:
                cap = cv.VideoCapture(source)
                if width is not None and width > 0:
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
                if height is not None and height > 0:
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            except Exception as e:
                print(f"Error creating regular capture: {e}")
                cap = None

        # Apply additional settings if the capture was created successfully
        if cap is not None and cap.isOpened():
            if 'size' in params:
                try:
                    w, h = map(int, params['size'].split('x'))
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
                except Exception as e:
                    print(f"Warning: could not set size: {e}")

            # Try platform-specific optimizations for non-threaded captures
            if not isinstance(cap, ThreadedVideoCapture) and isinstance(source, int):
                try:
                    system = platform.system()
                    if system == 'Windows':
                        # Optimize for Windows
                        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
                    elif system == 'Linux':
                        # Optimize for Linux V4L2
                        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
                    elif system == 'Darwin':  # macOS
                        # Optimize for macOS AVFoundation
                        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
                except Exception as e:
                    print(f"Warning: could not set codec format: {e}")

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None, threaded)
    return cap


def get_optimal_fps():
    """Return optimal FPS based on system capabilities"""
    system = platform.system()
    cpu_count = os.cpu_count() or 1

    if system == 'Windows':
        return min(30, cpu_count * 5)
    elif system == 'Darwin':  # macOS
        return min(30, cpu_count * 4)
    else:  # Linux and others
        return min(30, cpu_count * 6)


def get_optimal_format(cap):
    """Select optimal pixel format based on platform"""
    if not cap.isOpened():
        return cap

    try:
        system = platform.system()

        if system == 'Windows':
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
        elif system == 'Linux':
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
        elif system == 'Darwin':  # macOS
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
    except Exception as e:
        print(f"Warning: could not set optimal format: {e}")

    return cap


if __name__ == '__main__':
    import sys
    import getopt

    print(__doc__)

    # Add debug information
    print("Python version:", sys.version)
    print("OpenCV version:", cv.__version__)
    print("Platform:", platform.system(), platform.release())

    args, sources = getopt.getopt(sys.argv[1:], '', ['shotdir=', 'threaded='])
    args = dict(args)
    shotdir = args.get('--shotdir', '.')
    threaded = args.get('--threaded', 'True').lower() in ('true', 'yes', 't', 'y', '1')

    if len(sources) == 0:
        sources = [0]

    print("Initializing video sources:", sources)
    caps = []
    for src in sources:
        try:
            cap = create_capture(src, threaded=threaded)
            if cap is not None and cap.isOpened():
                caps.append(cap)
                print(f"Source {src} opened successfully")
            else:
                print(f"Failed to open source {src}")
        except Exception as e:
            print(f"Error with source {src}: {e}")

    if not caps:
        print("No video sources could be opened. Exiting.")
        sys.exit(1)

    shot_idx = 0

    # Determine if we should use a smaller or larger window based on screen resolution
    window_sizes = []
    for i, cap in enumerate(caps):
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        window_sizes.append((width, height))
        print(f"Source {i} resolution: {width}x{height}")

    start_time = time()
    frame_count = 0
    fps_display_interval = 2.0  # seconds
    fps = 0

    try:
        while True:
            imgs = []
            for i, cap in enumerate(caps):
                if not cap.isOpened():
                    continue

                ret, img = cap.read()
                if not ret:
                    print(f"Warning: Failed to get frame from source {i}")
                    continue

                # Calculate and display FPS
                frame_count += 1
                elapsed = time() - start_time
                if elapsed > fps_display_interval:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time()

                cv.putText(img, f"FPS: {fps:.1f}", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                imgs.append(img)
                cv.imshow(f'capture {i}', img)

            if not imgs:
                print("All video sources closed. Exiting.")
                break

            ch = cv.waitKey(1)
            if ch == 27:  # ESC
                break
            if ch == ord(' '):
                for i, img in enumerate(imgs):
                    fn = f'{shotdir}/shot_{i}_{shot_idx:03d}.png'
                    cv.imwrite(fn, img)
                    print(fn, 'saved')
                shot_idx += 1
    finally:
        # Clean up
        for cap in caps:
            if hasattr(cap, 'release'):  # For ThreadedVideoCapture
                cap.release()
        cv.destroyAllWindows()

