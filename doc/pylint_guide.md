# Guide to Handling PyLint Warnings with OpenCV

## Summary
When using the OpenCV library (`cv2`), PyLint may not recognize certain members, resulting in false-positive warnings in the Problems tab.


## Example Code
Here is a Minimum Working Example (MWE) that demonstrates the issue:

```python
import cv2

cap = cv2.VideoCapture("test.mp4")
background_subtractor = cv2.createBackgroundSubtractorMOG2()

ret, frame = cap.read()
foreground_mask = background_subtractor.apply(frame)
contours, hierarchy = cv2.findContours(
    foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

cv2.destroyAllWindows()
cap.release()