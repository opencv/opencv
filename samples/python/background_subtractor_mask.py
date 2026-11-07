
'''
Showcases the use of background subtraction from a live video feed,
aswell as pass through of a known foreground parameter
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened:
        print("Capture source avaialable.")
        exit()

    # Create background subtractor
    mog2_bg_subtractor = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)
    knn_bg_subtractor = cv.createBackgroundSubtractorKNN(history=300, detectShadows=False)

    frame_count = 0
    # Allows for a frame buffer for the mask to learn pre known foreground
    show_count = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x = 100 + (frame_count % 10) * 3

        frame = cv.resize(frame, (640, 480))
        aKnownForegroundMask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Allow for models to "settle"/learn
        if frame_count > show_count:
            cv.rectangle(aKnownForegroundMask, (x,200), (x+50,300), 255, -1)
            cv.rectangle(aKnownForegroundMask, (540,180), (640,480), 255, -1)

        #MOG2 Subtraction
        mog2_with_mask = mog2_bg_subtractor.apply(frame,knownForegroundMask=aKnownForegroundMask)
        mog2_without_mask = mog2_bg_subtractor.apply(frame)

        #KNN Subtraction
        knn_with_mask = knn_bg_subtractor.apply(frame,knownForegroundMask=aKnownForegroundMask)
        knn_without_mask = knn_bg_subtractor.apply(frame)

        # Display the 3 parameter apply and the 4 parameter apply for both subtractors
        cv.imshow("MOG2 With a Foreground Mask", mog2_with_mask)
        cv.imshow("MOG2 Without a Foreground Mask", mog2_without_mask)
        cv.imshow("KNN With a Foreground Mask", knn_with_mask)
        cv.imshow("KNN Without a Foreground Mask", knn_without_mask)

        key = cv.waitKey(30)
        if key == 27:  # ESC
            break

        frame_count += 1

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
