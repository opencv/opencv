
import sys
#sys.path.insert(0, "/home/your_user/opencv-install/lib/python3.*/site-packages")
import cv2
print(cv2.__file__)
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print("Capture source avaialable.")
    exit()

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)
#bg_subtractor = cv2.createBackgroundSubtractorKNN(history=300, detectShadows=False)


frame_count = 0
# Allows for a frame buffer for the mask to learn pre known foreground
show_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    x = 100 + (frame_count % 10) * 3

    frame = cv2.resize(frame, (640, 480))
    aKnownForegroundMask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if frame_count > show_count:
        cv2.rectangle(aKnownForegroundMask, (x,200), (x+50,300), 255, -1)
        cv2.rectangle(aKnownForegroundMask, (540,180), (640,480), 255, -1)

    with_mask = bg_subtractor.apply(frame,knownForegroundMask=aKnownForegroundMask)
    without_mask = bg_subtractor.apply(frame)

    cv2.imshow("With FG Mask", with_mask)
    cv2.imshow("Without FG Mask", without_mask)

    key = cv2.waitKey(30)
    if key == 27:  # ESC
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
