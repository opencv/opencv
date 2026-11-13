#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv
import io
import struct
import sys
import tempfile
import weakref

from tests_common import NewOpenCVTests

class Bindings(NewOpenCVTests):

    def check_name(self, name):
        #print(name)
        self.assertFalse(name == None)
        self.assertFalse(name == "")

    def test_registry(self):
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_ANY));
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_FFMPEG))
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_OPENCV_MJPEG))
        backends = cv.videoio_registry.getBackends()
        for backend in backends:
            self.check_name(cv.videoio_registry.getBackendName(backend))

    def test_capture_stream_file(self):
        if sys.version_info[0] < 3:
            raise self.skipTest('Python 3.x required')

        api_pref = None
        for backend in cv.videoio_registry.getStreamBufferedBackends():
            if not cv.videoio_registry.hasBackend(backend):
                continue
            if not cv.videoio_registry.isBackendBuiltIn(backend):
                _, abi, api = cv.videoio_registry.getStreamBufferedBackendPluginVersion(backend)
                if (abi < 1 or (abi == 1 and api < 2)):
                    continue
            api_pref = backend
            break

        if not api_pref:
            raise self.skipTest("No available backends")

        with open(self.find_file("cv/video/768x576.avi"), "rb") as f:
            cap = cv.VideoCapture(f, api_pref, [])
            self.assertTrue(cap.isOpened())
            hasFrame, frame = cap.read()
            self.assertTrue(hasFrame)
            self.assertEqual(frame.shape, (576, 768, 3))

    def test_capture_stream_buffer(self):
        if sys.version_info[0] < 3:
            raise self.skipTest('Python 3.x required')

        api_pref = None
        for backend in cv.videoio_registry.getStreamBufferedBackends():
            if not cv.videoio_registry.hasBackend(backend):
                continue
            if not cv.videoio_registry.isBackendBuiltIn(backend):
                _, abi, api = cv.videoio_registry.getStreamBufferedBackendPluginVersion(backend)
                if (abi < 1 or (abi == 1 and api < 2)):
                    continue
            api_pref = backend
            break

        if not api_pref:
            raise self.skipTest("No available backends")

        class BufferStream(io.BufferedIOBase):
            def __init__(self, filepath):
                self.f = open(filepath, "rb")

            def read(self, size=-1):
                return self.f.read(size)

            def seek(self, offset, whence):
                return self.f.seek(offset, whence)

            def __del__(self):
                self.f.close()

        stream = BufferStream(self.find_file("cv/video/768x576.avi"))

        cap = cv.VideoCapture(stream, api_pref, [])
        self.assertTrue(cap.isOpened())
        hasFrame, frame = cap.read()
        self.assertTrue(hasFrame)
        self.assertEqual(frame.shape, (576, 768, 3))

    def require_ffmpeg_stream(self):
        if sys.version_info[0] < 3:
            self.skipTest('Python 3.x required')
        if (cv.CAP_FFMPEG not in cv.videoio_registry.getStreamBufferedBackends()
                or not cv.videoio_registry.hasBackend(cv.CAP_FFMPEG)):
            self.skipTest("FFmpeg stream backend required")
        if not cv.videoio_registry.isBackendBuiltIn(cv.CAP_FFMPEG):
            _, abi, api = cv.videoio_registry.getStreamBufferedBackendPluginVersion(cv.CAP_FFMPEG)
            if abi < 1 or (abi == 1 and api < 2):
                self.skipTest("FFmpeg stream plugin API required")

    def test_capture_stream_large_seek(self):
        self.require_ffmpeg_stream()
        with open(self.find_file("highgui/video/sample_23976fps.mp4"), "rb") as f:
            data = f.read()

        class LargeVideoStream(io.BufferedIOBase):
            def __init__(self):
                moov_offset = data.index(b'moov') - 4
                moov_size = struct.unpack_from('>I', data, moov_offset)[0]
                self.suffix_offset = (1 << 31) + (1 << 16)
                # Preserve the media offsets by replacing the old moov with
                # free space, then place it after a second, virtual free box.
                self.prefix = (data[:moov_offset + 4] + b'free' + data[moov_offset + 8:]
                               + struct.pack('>I4s', self.suffix_offset - len(data), b'free'))
                self.suffix = data[moov_offset:moov_offset + moov_size]
                self.size = self.suffix_offset + moov_size
                self.pos = 0
                self.large_seeks = []

            def read(self, size=-1):
                size = min(size, max(0, self.size - self.pos))
                result = bytearray(size)
                end = self.pos + size
                for start, chunk in ((0, self.prefix), (self.suffix_offset, self.suffix)):
                    left, right = max(self.pos, start), min(end, start + len(chunk))
                    if left < right:
                        result[left - self.pos:right - self.pos] = chunk[left - start:right - start]
                self.pos = end
                return bytes(result)

            def seek(self, offset, whence):
                if whence == io.SEEK_SET and offset > (1 << 31) - 1:
                    self.large_seeks.append(offset)
                pos = offset + (0, self.pos, self.size)[whence]
                if pos < 0:
                    raise ValueError("negative seek position")
                self.pos = pos
                return pos

        # A virtual free box moves the MP4 index beyond 2 GiB without
        # allocating a large buffer or adding another video fixture.
        stream = LargeVideoStream()
        cap = cv.VideoCapture(stream, cv.CAP_FFMPEG, [])
        try:
            self.assertTrue(cap.isOpened())
            hasFrame, frame = cap.read()
            self.assertTrue(hasFrame)
            self.assertEqual(frame.shape, (120, 160, 3))
            self.assertTrue(stream.large_seeks)
        finally:
            cap.release()

    def check_capture_stream_callback_error(self, method, error):
        with open(self.find_file("cv/video/768x576.avi"), "rb") as f:
            data = f.read()

        class FailingStream(io.BytesIO):
            calls = 0

            def fail(self, *args):
                self.calls += 1
                if error == 'exception':
                    raise RuntimeError("stream callback failed")
                if error == 'oversized':
                    return b'\0' * (args[0] + 1)
                if error == 'overflow':
                    return 1 << 100
                if error == 'wrong_type':
                    return 'invalid result'
                return None

            def read(self, size=-1):
                return self.fail(size) if method == 'read' else super(FailingStream, self).read(size)

            def seek(self, offset, whence=0):
                return self.fail(offset, whence) if method == 'seek' else super(FailingStream, self).seek(offset, whence)

        stream = FailingStream(data)
        cap = cv.VideoCapture()
        try:
            # The backend may report failure or recover without seeking.
            # Neither a native crash nor a pending Python error is acceptable.
            try:
                cap.open(stream, cv.CAP_FFMPEG, [])
            except cv.error:
                pass
            self.assertGreater(stream.calls, 0)
            if method == 'read':
                self.assertFalse(cap.isOpened())
        finally:
            cap.release()

        valid_stream = io.BytesIO(data)
        cap = cv.VideoCapture(valid_stream, cv.CAP_FFMPEG, [])
        try:
            self.assertTrue(cap.isOpened())
            self.assertTrue(cap.read()[0])
        finally:
            cap.release()

    def test_capture_stream_read_errors(self):
        self.require_ffmpeg_stream()
        for error in ('exception', 'none', 'wrong_type', 'oversized'):
            with self.subTest(error=error):
                self.check_capture_stream_callback_error('read', error)

    def test_capture_stream_seek_errors(self):
        self.require_ffmpeg_stream()
        for error in ('exception', 'none', 'wrong_type', 'overflow'):
            with self.subTest(error=error):
                self.check_capture_stream_callback_error('seek', error)

    def test_capture_stream_release_last_reference(self):
        self.require_ffmpeg_stream()
        with open(self.find_file("cv/video/768x576.avi"), "rb") as f:
            stream = io.BytesIO(f.read())
        stream_ref = weakref.ref(stream)
        cap = cv.VideoCapture(stream, cv.CAP_FFMPEG, [])
        try:
            self.assertTrue(cap.isOpened())
            del stream
            self.assertIsNotNone(stream_ref())
            self.assertTrue(cap.read()[0])
        finally:
            cap.release()
        self.assertIsNone(stream_ref())

    def test_context_manager(self):
        video_file = self.find_file("cv/video/768x576.avi")

        with cv.VideoCapture(video_file) as cap:
            self.assertTrue(cap.isOpened(), "VideoCapture should be opened within context manager")

        with tempfile.NamedTemporaryFile(suffix='.avi') as tmp:
            with cv.VideoWriter(tmp.name, cv.VideoWriter_fourcc(*'MJPG'), 25, (640, 480)) as writer:
                self.assertTrue(isinstance(writer, cv.VideoWriter))

        try:
            with cv.VideoCapture(video_file) as cap:
                self.assertTrue(cap.isOpened())
                raise RuntimeError("Testing context manager exception safety")
        except RuntimeError:
            pass

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
