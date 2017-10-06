import cv2
import numpy as np

input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 255, 0, 0],
    [0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0, 255, 255, 0],
    [0,255, 0, 255, 0, 0, 255, 0],
    [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

kernel = np.array((
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]), dtype="int")

output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)

rate = 50
kernel = (kernel + 1) * 127
kernel = np.uint8(kernel)

kernel = cv2.resize(kernel, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
cv2.imshow("kernel", kernel)
cv2.moveWindow("kernel", 0, 0)

input_image = cv2.resize(input_image, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
cv2.imshow("Original", input_image)
cv2.moveWindow("Original", 0, 200)

output_image = cv2.resize(output_image, None , fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
cv2.imshow("Hit or Miss", output_image)
cv2.moveWindow("Hit or Miss", 500, 200)

cv2.waitKey(0)
cv2.destroyAllWindows()
