#!/usr/bin/env python3

"""Code for the Nested ArUco Markers tutorial.

Usage:
  python3 aruco_nested_detection.py --step create
  python3 aruco_nested_detection.py --step detect [--dictionary custom_nested.yml]
  python3 aruco_nested_detection.py --step pose --camera-parameters camera_parameters.yml
  python3 aruco_nested_detection.py --step custom
"""

import argparse
import numpy as np
import cv2 as cv


def create_marker():
    ## [nested_marker_create_py]
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_NESTED_10)
    marker = cv.aruco.generateImageMarkerNested(dictionary, 0, 1200)  # outer id 0, inner id 1
    cv.imwrite("pair_0.png", marker)
    ## [nested_marker_create_py]


def detect_markers(dictionary):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could not open camera 0")
    ## [nested_marker_detect_py]
    params = cv.aruco.DetectorParameters()
    params.detectNestedMarkers = True
    detector = cv.aruco.ArucoDetector(dictionary, params)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _ = detector.detectMarkers(frame)
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        cv.imshow("nested markers", frame)
        if cv.waitKey(1) == 27:
            break
    ## [nested_marker_detect_py]
    cv.destroyAllWindows()


def estimate_pose(camera_matrix, dist_coeffs):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_NESTED_10)
    params = cv.aruco.DetectorParameters()
    params.detectNestedMarkers = True
    detector = cv.aruco.ArucoDetector(dictionary, params)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could not open camera 0")
    ## [nested_marker_pose_py]
    side_length = 0.20  # printed outer side in meters
    outer_pts, inner_pts = cv.aruco.getNestedMarkerObjectPoints(dictionary, 0, side_length)
    board = cv.aruco.Board([outer_pts, inner_pts], dictionary, np.array([0, 1]))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _ = detector.detectMarkers(frame)
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        if ids is not None and len(ids) > 0:
            obj_points, img_points = board.matchImagePoints(corners, ids)
            if len(obj_points) >= 4:
                ok, rvec, tvec = cv.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                if ok:
                    cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, side_length * 0.5)
        cv.imshow("nested marker pose", frame)
        if cv.waitKey(1) == 27:
            break
    ## [nested_marker_pose_py]
    cv.destroyAllWindows()


def create_custom_dictionary():
    ## [nested_custom_create_py]
    # nest marker 1 of DICT_4X4_50, half a cell wide, in the top left cell of marker 0 (a white cell)
    binary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    cell = 240  # pixels per cell, the marker has 4 cells plus a one cell border on each side
    outer = cv.aruco.generateImageMarker(binary, 0, 6 * cell)
    inner = cv.aruco.generateImageMarker(binary, 1, cell // 2)
    outer[5 * cell // 4:7 * cell // 4, 5 * cell // 4:7 * cell // 4] = inner

    # store the white pixel ratio of every cell: id 0 is the outer marker, id 1 the inner one
    ratios = [cv.aruco.Dictionary.getCellRatiosFromImage(image, 4) for image in (outer, inner)]
    rows = [cv.aruco.Dictionary.getRatioListFromCellRatios(r) for r in ratios]
    dictionary = cv.aruco.Dictionary(np.concatenate(rows), 4, 0, cv.aruco.DICT_ENCODING_CELL_RATIO)

    storage = cv.FileStorage("custom_nested.yml", cv.FILE_STORAGE_WRITE)
    dictionary.writeDictionary(storage)
    storage.release()
    cv.imwrite("custom_nested.png", outer)
    ## [nested_custom_create_py]


def read_dictionary(path):
    ## [nested_custom_read_py]
    dictionary = cv.aruco.Dictionary()
    storage = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    if not storage.isOpened() or not dictionary.readDictionary(storage.root()):
        raise RuntimeError("could not read the dictionary from " + path)
    ## [nested_custom_read_py]
    return dictionary


def main():
    parser = argparse.ArgumentParser(description="Code for the Nested ArUco Markers tutorial.")
    parser.add_argument("--step", choices=("create", "detect", "pose", "custom"), default="create")
    parser.add_argument("--dictionary", help="custom dictionary file for the detect step")
    parser.add_argument("--camera-parameters", help="camera calibration file for the pose step")
    args = parser.parse_args()

    if args.step == "create":
        create_marker()
    elif args.step == "custom":
        create_custom_dictionary()
    elif args.step == "detect":
        if args.dictionary:
            detect_markers(read_dictionary(args.dictionary))
        else:
            detect_markers(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_NESTED_10))
    else:
        if args.camera_parameters is None:
            parser.error("--step pose requires --camera-parameters")
        storage = cv.FileStorage(args.camera_parameters, cv.FILE_STORAGE_READ)
        camera_matrix = storage.getNode("camera_matrix").mat()
        dist_coeffs = storage.getNode("distortion_coefficients").mat()
        if camera_matrix is None or dist_coeffs is None:
            parser.error("could not read camera_matrix and distortion_coefficients")
        estimate_pose(camera_matrix, dist_coeffs)


if __name__ == "__main__":
    main()
