import time
import numpy as np
import cv2
## [basic_method]
def sharpen(my_image):
    my_image = cv2.cvtColor(my_image, cv2.CV_8U)

    height, width, n_channels = my_image.shape
    result = np.zeros(my_image.shape, my_image.dtype)

    ## [basic_method_loop]
    for j in range  (1, height-1):
        for i in range  (1, width-1):
            for k in range  (0, n_channels):
                sum = 5 * my_image[j, i, k] - my_image[j + 1, i, k] - my_image[j - 1, i, k]\
                     - my_image[j, i + 1, k] - my_image[j, i - 1, k];

                if sum > 255:
                    sum = 255
                if sum < 0:
                    sum = 0

                result[j, i, k] = sum
    ## [basic_method_loop]

    return result
## [basic_method]


I = cv2.imread("../data/lena.jpg")
cv2.imshow('Input Image', I)

t = round(time.time())
J = sharpen(I)
t = (time.time() - t)/1000
print "Hand written function times passed in seconds: %s" % t

cv2.imshow('Output Image', J)

t = time.time()
## [kern]
kernel = np.array([ [0,-1,0],
                    [-1,5,-1],
                    [0,-1,0] ],np.float32) # kernel should be floating point type
## [kern]

## [filter2D]
K = cv2.filter2D(I, -1, kernel) # ddepth = -1, means destination image has depth same as input image.
## [filter2D]

t = (time.time() - t)/1000
print "Built-in filter2D time passed in seconds:      %s" % t

cv2.imshow('filter2D Output Image', K)

cv2.waitKey(0)
cv2.destroyAllWindows()
