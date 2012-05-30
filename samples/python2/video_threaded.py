import numpy as np
import cv2

from multiprocessing.pool import ThreadPool
from collections import deque


if __name__ == '__main__':
    def process_frame(frame):
        # some intensive computation...
        frame = cv2.medianBlur(frame, 19)
        frame = cv2.medianBlur(frame, 19)
        frame = cv2.medianBlur(frame, 19)
        return frame

    threadn = 8
    cap = cv2.VideoCapture(0)
    pool = ThreadPool(processes = threadn)
    pending = deque()
    while True:
        while len(pending) > 0 and pending[0].ready():
            res = pending.popleft().get()
            cv2.imshow('result', res)
        if len(pending) < threadn+1:
            ret, frame = cap.read()
            task = pool.apply_async(process_frame, (frame.copy(),))
            pending.append(task)
        if cv2.waitKey(1) == 27:
            break
