"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import sys
import cv2 as cv

def main(argv):
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]

    # [load]
    imageName = argv[0] if len(argv) > 0 else "../data/lena.jpg"

    src = cv.imread(imageName, cv.IMREAD_COLOR) # Load an image

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default ../data/lena.jpg]')
        return -1
    # [load]

    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]

    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # [convert_to_gray]

    # Create Window
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # [laplacian]
    # Apply Laplace function
    dst = cv.Laplacian(src_gray, ddepth, kernel_size)
    # [laplacian]

    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)
    # [convert]

    # [display]
    cv.imshow(window_name, abs_dst)
    cv.waitKey(0)
    # [display]

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
