import sys
import cv

def hs_histogram(src):
    # Convert to HSV
    hsv = cv.CreateImage(cv.GetSize(src), 8, 3)
    cv.CvtColor(src, hsv, cv.CV_BGR2HSV)

    # Extract the H and S planes
    h_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    s_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    cv.Split(hsv, h_plane, s_plane, None, None)
    planes = [h_plane, s_plane]

    h_bins = 30
    s_bins = 32
    hist_size = [h_bins, s_bins]
    # hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
    h_ranges = [0, 180]
    # saturation varies from 0 (black-gray-white) to
    # 255 (pure spectrum color)
    s_ranges = [0, 255]
    ranges = [h_ranges, s_ranges]
    scale = 10
    hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)
    cv.CalcHist([cv.GetImage(i) for i in planes], hist)
    (_, max_value, _, _) = cv.GetMinMaxHistValue(hist)

    hist_img = cv.CreateImage((h_bins*scale, s_bins*scale), 8, 3)

    for h in range(h_bins):
        for s in range(s_bins):
            bin_val = cv.QueryHistValue_2D(hist, h, s)
            intensity = cv.Round(bin_val * 255 / max_value)
            cv.Rectangle(hist_img,
                         (h*scale, s*scale),
                         ((h+1)*scale - 1, (s+1)*scale - 1),
                         cv.RGB(intensity, intensity, intensity), 
                         cv.CV_FILLED)
    return hist_img
    
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

def findstereocorrespondence(image_left, image_right):
    # image_left and image_right are the input 8-bit single-channel images
    # from the left and the right cameras, respectively
    (r, c) = (image_left.rows, image_left.cols)
    disparity_left = cv.CreateMat(r, c, cv.CV_16S)
    disparity_right = cv.CreateMat(r, c, cv.CV_16S)
    state = cv.CreateStereoGCState(16, 2)
    cv.FindStereoCorrespondenceGC(image_left, image_right, disparity_left, disparity_right, state, 0)
    return (disparity_left, disparity_right)
