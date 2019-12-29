import cv2 as cv
import numpy as np

def main():
    import sys
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--image",required=True,help="path to input file")
    args=vars(parser.parse_args())

    # Read the image from the disk
    image = cv.imread(cv.samples.findFile(args["image"]))

    # Exit if the image is not found
    if image is None:
        print("Cannot load the image "+ args["image"])
        sys.exit(-1)

    # Show the image window
    cv.imshow("Display image ",image)
    # Wait for a keystroke in the window
    cv.waitKey(0)

if __name__ == "__main__":
    main()
