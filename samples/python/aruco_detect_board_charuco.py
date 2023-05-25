#!/usr/bin/env python

"""aruco_detect_board_charuco.py
Usage example:
python aruco_detect_board_charuco.py -w=5 -h=7 -sl=0.04 -ml=0.02 -d=10 -c=../data/aruco/tutorial_camera_charuco.yml
                                     -i=../data/aruco/choriginal.jpg
"""

import argparse
import numpy as np
import cv2 as cv
import sys


def read_camera_parameters(filename):
    fs = cv.FileStorage(cv.samples.findFile(filename, False), cv.FileStorage_READ)
    if fs.isOpened():
        cam_matrix = fs.getNode("camera_matrix").mat()
        dist_coefficients = fs.getNode("distortion_coefficients").mat()
        return True, cam_matrix, dist_coefficients
    return False, [], []


def main():
    # parse command line options
    parser = argparse.ArgumentParser(description="detect markers and corners of charuco board, estimate pose of charuco"
                                     "board", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-v", "--video", help="Input from video or image file, if omitted, input comes from camera",
                        default="", action="store", dest="v")
    parser.add_argument("-i", "--image", help="Input from image file", default="", action="store", dest="img_path")
    parser.add_argument("-w", help="Number of squares in X direction", default="3", action="store", dest="w", type=int)
    parser.add_argument("-h", help="Number of squares in Y direction", default="3", action="store", dest="h", type=int)
    parser.add_argument("-sl", help="Square side length", default="1.", action="store", dest="sl", type=float)
    parser.add_argument("-ml", help="Marker side length", default="0.5", action="store", dest="ml", type=float)
    parser.add_argument("-d", help="dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,  DICT_4X4_1000=3,"
                                   "DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8,"
                                   "DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13,"
                                   "DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}",
                        default="0", action="store", dest="d", type=int)
    parser.add_argument("-ci", help="Camera id if input doesnt come from video (-v)", default="0", action="store",
                        dest="ci", type=int)
    parser.add_argument("-c", help="Input file with calibrated camera parameters", default="", action="store",
                        dest="cam_param")

    args = parser.parse_args()

    show_help = args.show_help
    if show_help:
        parser.print_help()
        sys.exit()
    width = args.w
    height = args.h
    square_len = args.sl
    marker_len = args.ml
    dict = args.d
    video = args.v
    camera_id = args.ci
    img_path = args.img_path

    cam_param = args.cam_param
    cam_matrix = []
    dist_coefficients = []
    if cam_param != "":
        _, cam_matrix, dist_coefficients = read_camera_parameters(cam_param)

    aruco_dict = cv.aruco.getPredefinedDictionary(dict)
    board_size = (width, height)
    board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)
    charuco_detector = cv.aruco.CharucoDetector(board)

    image = None
    input_video = None
    wait_time = 10
    if video != "":
        input_video = cv.VideoCapture(cv.samples.findFileOrKeep(video, False))
        image = input_video.retrieve()[1] if input_video.grab() else None
    elif img_path == "":
        input_video = cv.VideoCapture(camera_id)
        image = input_video.retrieve()[1] if input_video.grab() else None
    elif img_path != "":
        wait_time = 0
        image = cv.imread(cv.samples.findFile(img_path, False))

    if image is None:
        print("Error: unable to open video/image source")
        sys.exit(0)

    while image is not None:
        image_copy = np.copy(image)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(image)
        if not (marker_ids is None) and len(marker_ids) > 0:
            cv.aruco.drawDetectedMarkers(image_copy, marker_corners)
        if not (charuco_ids is None) and len(charuco_ids) > 0:
            cv.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids)
            if len(cam_matrix) > 0 and len(charuco_ids) >= 4:
                try:
                    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                    flag, rvec, tvec = cv.solvePnP(obj_points, img_points, cam_matrix, dist_coefficients)
                    if flag:
                        cv.drawFrameAxes(image_copy, cam_matrix, dist_coefficients, rvec, tvec, .2)
                except cv.error as error_inst:
                    print("SolvePnP recognize calibration pattern as non-planar pattern. To process this need to use "
                          "minimum 6 points. The planar pattern may be mistaken for non-planar if the pattern is "
                          "deformed or incorrect camera parameters are used.")
                    print(error_inst.err)
        cv.imshow("out", image_copy)
        key = cv.waitKey(wait_time)
        if key == 27:
            break
        image = input_video.retrieve()[1] if input_video is not None and input_video.grab() else None


if __name__ == "__main__":
    main()
