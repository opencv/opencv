import cv

def precornerdetect(image):
    # assume that the image is floating-point 
    corners = cv.CloneMat(image)
    cv.PreCornerDetect(image, corners, 3)

    dilated_corners = cv.CloneMat(image)
    cv.Dilate(corners, dilated_corners, None, 1)

    corner_mask = cv.CreateMat(image.rows, image.cols, cv.CV_8UC1)
    cv.Sub(corners, dilated_corners, corners)
    cv.CmpS(corners, 0, corner_mask, cv.CV_CMP_GE)
    return (corners, corner_mask)
