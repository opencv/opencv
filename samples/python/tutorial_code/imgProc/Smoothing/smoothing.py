import sys
import cv2
import numpy as np

#  Global Variables

DELAY_CAPTION = 1500
DELAY_BLUR = 100
MAX_KERNEL_LENGTH = 31

src = None
dst = None
window_name = 'Smoothing Demo'


def main(argv):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Load the source image
    imageName = argv[0] if len(argv) > 0 else "../data/lena.jpg"

    global src
    src = cv2.imread(imageName, 1)
    if src is None:
        print ('Error opening image')
        print ('Usage: smoothing.py [image_name -- default ../data/lena.jpg] \n')
        return -1

    if display_caption('Original Image') != 0:
        return 0

    global dst
    dst = np.copy(src)
    if display_dst(DELAY_CAPTION) != 0:
        return 0

    # Applying Homogeneous blur
    if display_caption('Homogeneous Blur') != 0:
        return 0

    ## [blur]
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.blur(src, (i, i))
        if display_dst(DELAY_BLUR) != 0:
            return 0
    ## [blur]

    # Applying Gaussian blur
    if display_caption('Gaussian Blur') != 0:
        return 0

    ## [gaussianblur]
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.GaussianBlur(src, (i, i), 0)
        if display_dst(DELAY_BLUR) != 0:
            return 0
    ## [gaussianblur]

    # Applying Median blur
    if display_caption('Median Blur') != 0:
        return 0

    ## [medianblur]
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.medianBlur(src, i)
        if display_dst(DELAY_BLUR) != 0:
            return 0
    ## [medianblur]

    # Applying Bilateral Filter
    if display_caption('Bilateral Blur') != 0:
        return 0

    ## [bilateralfilter]
    # Remember, bilateral is a bit slow, so as value go higher, it takes long time
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.bilateralFilter(src, i, i * 2, i / 2)
        if display_dst(DELAY_BLUR) != 0:
            return 0
    ## [bilateralfilter]

    #  Done
    display_caption('Done!')

    return 0


def display_caption(caption):
    global dst
    dst = np.zeros(src.shape, src.dtype)
    rows, cols, ch = src.shape
    cv2.putText(dst, caption,
                (int(cols / 4), int(rows / 2)),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    return display_dst(DELAY_CAPTION)


def display_dst(delay):
    cv2.imshow(window_name, dst)
    c = cv2.waitKey(delay)
    if c >= 0 : return -1
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
