import numpy as np
import cv2

from Queue import Queue
from threading import Thread
from collections import deque

class Worker(Thread):
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()
    
    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try: func(*args, **kargs)
            except Exception, e: print e
            self.tasks.task_done()

class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads): Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        self.tasks.join()

if __name__ == '__main__':
    results = deque()

    def process_frame(i, frame):
        global results
        res = cv2.medianBlur(frame, 15)
        results.append((i, res))
    
    pool = ThreadPool(4)
    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_frame = None
    last_count = -1
    while True:
        ret, frame = cap.read()
        pool.add_task(process_frame, frame_count, frame.copy())
        frame_count += 1
        while len(results) > 0:
            i, frame = results.popleft()
            if i > last_count:
                last_count, last_frame = i, frame
        if last_frame is not None:
            cv2.imshow('res', last_frame)
        if cv2.waitKey(1) == 27:
            break

    pool.wait_completion()




