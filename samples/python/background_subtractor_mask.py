
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
mog2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)
knn_bg_subtractor = cv2.createBackgroundSubtractorKNN(history=300, detectShadows=False)

frame_count = 0
# Allows for a frame buffer for the mask to learn pre known foreground
show_count = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x = 100 + (frame_count % 10) * 3

    frame = cv2.resize(frame, (640, 480))
    aKnownForegroundMask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Allow for models to "settle"/learn
    if frame_count > show_count:
        cv2.rectangle(aKnownForegroundMask, (x,200), (x+50,300), 255, -1)
        cv2.rectangle(aKnownForegroundMask, (540,180), (640,480), 255, -1)

    mog2_with_mask = mog2_bg_subtractor.apply(frame,knownForegroundMask=aKnownForegroundMask)
    mog2_without_mask = mog2_bg_subtractor.apply(frame)

    knn_with_mask = knn_bg_subtractor.apply(frame,knownForegroundMask=aKnownForegroundMask)
    knn_without_mask = knn_bg_subtractor.apply(frame)

    # Display the 3 parameter apply and the 4 parameter apply for both subtractors
    cv2.imshow("MOG2 With FG Mask", mog2_with_mask)
    cv2.imshow("MOG2 Without FG Mask", mog2_without_mask)
    cv2.imshow("KNN With FG Mask", knn_with_mask)
    cv2.imshow("KNN Without FG Mask", knn_without_mask)

    key = cv2.waitKey(30)
    if key == 27:  # ESC
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
