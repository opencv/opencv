'''
Calibrates a stereo camera setup by reading in distorted images and writing calibration matrices.
These calibration matrices can be used to produce a disparity map for depth calculation

usage:
    edit directories to suit your needs
    run script

'''
# Python 2/3 compatibility
from __future__ import print_function
from glob import glob
import os
import numpy as np
import cv2
import re

#Length of square of checkerboard in meters
CHESSBOARD_SQUARE_SIZE = .0265
#Number of inner points of the checkerboard
CHESSBOARD_CALIBRATION_SIZE = (6,9)
CHESSBOARD_OPTIONS = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH)
DRAW_IMAGE = True

OBJECT_POINT_ZERO = np.zeros((CHESSBOARD_CALIBRATION_SIZE[0] * CHESSBOARD_CALIBRATION_SIZE[1], 3), np.float32)
OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHESSBOARD_CALIBRATION_SIZE[0],0 : CHESSBOARD_CALIBRATION_SIZE[1]].T.reshape(-1, 2)*CHESSBOARD_SQUARE_SIZE

OPTIMIZE_ALPHA = 0.25

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,0.001)
unreadable = []

#Get the current directory and the place you have stored your images
os.chdir(path = os.path.realpath(__file__)[:-len(os.path.basename(__file__))])
os.chdir("../")
path = os.getcwd() + "/data/"
left_img_mask = 'left**.png'
right_img_mask = 'right**.png'
left_images = sorted(glob(path + left_img_mask))
right_images = sorted(glob(path + right_img_mask))

#Check if any images don't have a checkerboard in them. If so, take note of which ones
for x in left_images:
    image = cv2.imread(x)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    has_corners, corners = cv2.findChessboardCorners(image_grey, CHESSBOARD_CALIBRATION_SIZE, cv2.CALIB_CB_FAST_CHECK)
    if not has_corners:
        z = re.findall("\d+", x)[-1]
        unreadable.append(z)

for x in right_images:

    if x.split("/")[-1] not in unreadable:
        image = cv2.imread(x)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        has_corners, corners = cv2.findChessboardCorners(image_grey, CHESSBOARD_CALIBRATION_SIZE, cv2.CALIB_CB_FAST_CHECK)
        if not has_corners:
            z = re.findall("\d+", x)[-1]
            unreadable.append(z)
    else:
        pass

#Remove both the left and right image of any images where checkerboards are not present
if len(unreadable) > 0:
    print("Chessboard not found in the following images. Removing them...")
for x in unreadable:
    print(path + "left_" + x)
    try:
       os.remove(path + "left_" + x)
    except:
        pass
for x in unreadable:
    print(path + "right_" + x)
    try:
        os.remove(path + "right_" + x)
    except:
        pass


def findChessboards(images):
    """
    Calculates the object points and image points of chessboard images
    :param imageDirectory: the directory to look through for chessboard images
    :return: names of the files, the object points list, the image points list, resolution of the camera
    """

    file_names = []
    object_points = []
    image_points = []
    left_camera_size = None


    for image_path in sorted(images):
        image = cv2.imread(image_path)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        left_camera_size = image_grey.shape[::-1]
        has_corners, corners = cv2.findChessboardCorners(image_grey, CHESSBOARD_CALIBRATION_SIZE, cv2.CALIB_CB_FAST_CHECK)

        if has_corners:
            file_names.append(image_path)
            object_points.append(OBJECT_POINT_ZERO)
            cv2.cornerSubPix(image_grey, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
            image_points.append(corners)

        if DRAW_IMAGE:
            cv2.drawChessboardCorners(image, CHESSBOARD_CALIBRATION_SIZE, corners, has_corners)
            cv2.imshow(image_path, image)


        cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Found corners in {0} out of {1} images".format(len(image_points), len(images)))
    return file_names, object_points, image_points, left_camera_size


(left_file_names, left_object_points, left_image_points, left_camera_size) = findChessboards(left_images)
(right_file_names, right_object_points, right_image_points, left_camera_size) = findChessboards(right_images)

left_images = glob(path + left_img_mask)
right_images = glob(path + right_img_mask)

object_points = left_object_points

print("calibrating left camera...")
_, leftCameraMatrix, leftDistortionCoefficients, _, _ = cv2.calibrateCamera(
        object_points, left_image_points, left_camera_size, None, None)

print("Done")
print("calibrating right camera...")
_, rightCameraMatrix, rightDistortionCoefficients, _, _ = cv2.calibrateCamera(
        object_points, right_image_points, left_camera_size, None, None)

print("Done")
print("calibrating stereo cameras...")
(_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        object_points, left_image_points, right_image_points,
        leftCameraMatrix, leftDistortionCoefficients,
        rightCameraMatrix, rightDistortionCoefficients,
        left_camera_size, None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)

print("Done")
print("Starting stereo rectification...")
(leftRectification, rightRectification, leftProjection, rightProjection,
        Q_matrix, left_roi, right_roi) = cv2.stereoRectify(
                leftCameraMatrix, leftDistortionCoefficients,
                rightCameraMatrix, rightDistortionCoefficients,
                left_camera_size, rotationMatrix, translationVector,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

print("Done")
print("Saving the stereo calibration...")
left_xmap, left_ymap = cv2.initUndistortRectifyMap(
        leftCameraMatrix, leftDistortionCoefficients, leftRectification,
        leftProjection, left_camera_size, cv2.CV_32FC1)
right_xmap, right_ymap = cv2.initUndistortRectifyMap(
        rightCameraMatrix, rightDistortionCoefficients, rightRectification,
        rightProjection, left_camera_size, cv2.CV_32FC1)

#If you would like to save the calibration parameters
'''
np.savez_compressed(out_file1, image_size=left_camera_size,
        left_xmap=left_xmap, left_ymap=left_ymap, left_roi=left_roi,
        right_xmap=right_xmap, right_ymap=right_ymap, right_roi=right_roi, Q_matrix = Q_matrix)

np.savez_compressed(out_file2, leftRectification = leftRectification, rightRectification=rightRectification, leftProjection=leftProjection, rightProjection=rightProjection,
        Q_matrix=Q_matrix)
'''

cv2.destroyAllWindows()

print("Done!")
