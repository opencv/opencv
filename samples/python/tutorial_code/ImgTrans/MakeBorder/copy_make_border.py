"""
@file copy_make_border.py
@brief Sample code that shows the functionality of copyMakeBorder
"""
import sys
from random import randint
import cv2 as cv


def main(argv):
    ## [variables]
    # First we declare the variables we are going to use
    borderType = cv.BORDER_CONSTANT
    window_name = "copyMakeBorder Demo"
    ## [variables]
    ## [load]
    imageName = argv[0] if len(argv) > 0 else "../data/lena.jpg"

    # Loads an image
    src = cv.imread(imageName, cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: copy_make_border.py [image_name -- default ../data/lena.jpg] \n')
        return -1
    ## [load]
    # Brief how-to for this program
    print ('\n'
           '\t 	 copyMakeBorder Demo: \n'
           '	 -------------------- \n'
           ' ** Press \'c\' to set the border to a random constant value \n'
           ' ** Press \'r\' to set the border to be replicated \n'
           ' ** Press \'ESC\' to exit the program ')
    ## [create_window]
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    ## [create_window]
    ## [init_arguments]
    # Initialize arguments for the filter
    top = int(0.05 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * src.shape[1])  # shape[1] = cols
    right = left
    ## [init_arguments]
    while 1:
        ## [update_value]
        value = [randint(0, 255), randint(0, 255), randint(0, 255)]
        ## [update_value]
        ## [copymakeborder]
        dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
        ## [copymakeborder]
        ## [display]
        cv.imshow(window_name, dst)
        ## [display]
        ## [check_keypress]
        c = cv.waitKey(500)

        if c == 27:
            break
        elif c == 99: # 99 = ord('c')
            borderType = cv.BORDER_CONSTANT
        elif c == 114: # 114 = ord('r')
            borderType = cv.BORDER_REPLICATE
        ## [check_keypress]
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
