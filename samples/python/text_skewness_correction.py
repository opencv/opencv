
'''
Text skewness correction

Usage:
        python text_skewness_correction.py --image "Image path"

'''


import numpy as np
import cv2 as cv

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True,help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    image = cv.imread(args["image"])
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv.threshold(gray, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]

    # the `cv.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h),flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    cv.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv.imshow("Input", image)
    cv.imshow("Rotated", rotated)
    cv.waitKey(0)
if __name__=="__main__":
    main()