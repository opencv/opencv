"""
@file filter2D.py
@brief Sample code that shows how to implement your own linear filters by using filter2D function
"""
import sys
import cv2
import numpy as np


def main(argv):
    window_name = 'filter2D Demo'

    ## [load]
    imageName = argv[0] if len(argv) > 0 else "../data/lena.jpg"

    # Loads an image
    src = cv2.imread(imageName, cv2.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: filter2D.py [image_name -- default ../data/lena.jpg] \n')
        return -1
    ## [load]
    ## [init_arguments]
    # Initialize ddepth argument for the filter
    ddepth = -1
    ## [init_arguments]
    # Loop - Will filter the image with different kernel sizes each 0.5 seconds
    ind = 0
    while True:
        ## [update_kernel]
        # Update kernel size for a normalized box filter
        kernel_size = 3 + 2 * (ind % 5)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= (kernel_size * kernel_size)
        ## [update_kernel]
        ## [apply_filter]
        # Apply filter
        dst = cv2.filter2D(src, ddepth, kernel)
        ## [apply_filter]
        cv2.imshow(window_name, dst)

        c = cv2.waitKey(500)
        if c == 27:
            break

        ind += 1

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
