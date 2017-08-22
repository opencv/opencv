"""
@file sobel_demo.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import sys
import cv2


def main(argv):
    ## [variables]
    # First we declare the variables we are going to use
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    ## [variables]

    ## [load]
    # As usual we load our source image (src)
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    src = cv2.imread(argv[0], cv2.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    ## [load]

    ## [reduce_noise]
    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    src = cv2.GaussianBlur(src, (3, 3), 0)
    ## [reduce_noise]

    ## [convert_to_gray]
    # Convert the image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [sobel]
    # Gradient-X
    # grad_x = cv2.Scharr(gray,ddepth,1,0)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    # Gradient-Y
    # grad_y = cv2.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    ## [sobel]

    ## [convert]
    # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    ## [convert]

    ## [blend]
    ## Total Gradient (approximate)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ## [blend]

    ## [display]
    cv2.imshow(window_name, grad)
    cv2.waitKey(0)
    ## [display]

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
