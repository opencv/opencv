'''
ArUco2 marker detection and pose estimation tutorial.
'''

import cv2 as cv
import numpy as np

def marker_creation():
    # [marker_creation]
    # Select a dictionary
    dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12

    # Generate a marker (ID 42, 20x20 pixels per bit)
    marker_img = cv.aruco2.getFiducialMarkerImage(dictionary, 42, bitSize=20)

    cv.imwrite("marker42.png", marker_img)
    # [marker_creation]

def marker_detection(img):
    # [marker_detection]
    # Select the same dictionary used for creation
    dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12

    # Detect markers
    markers = cv.aruco2.detectFiducialMarkers(img, dictionary)

    # Iterate over detected markers
    for m in markers:
        print(f"Detected marker ID: {m.id}")
        # corners is a list of 4 points
        for i, p in enumerate(m.corners):
            print(f"  Corner {i}: {p}")

    # Visualize results
    cv.aruco2.drawFiducialMarkers(img, markers)
    # [marker_detection]

def multi_dict_detection(img):
    # [multi_dict]
    # Detect both ArUco and AprilTag markers
    dictionaries = [cv.aruco2.DICT_ARUCO_MIP_36h12, cv.aruco2.DICT_APRILTAG_36h11]
    markers = cv.aruco2.detectFiducialMarkers(img, dictionaries)

    for m in markers:
        dict_name = "ArUco" if m.dictionary == cv.aruco2.DICT_ARUCO_MIP_36h12 else "AprilTag"
        print(f"Found {dict_name} marker ID: {m.id}")
    # [multi_dict]

def detection_parameters(img):
    # [detection_params]
    # Create and customize detection parameters
    params = cv.aruco2.DetectionParameters()
    params.boxFilterSize = 15
    params.thres = 3
    params.errorCorrectionRate = 0.5  # More tolerant (0.0 is default)

    dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12
    markers = cv.aruco2.detectFiducialMarkers(img, dictionary, detectorParams=params)
    # [detection_params]

def pose_estimation(img, camera_matrix, dist_coeffs):
    # [pose_estimation]
    dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12
    markers = cv.aruco2.detectFiducialMarkers(img, dictionary)

    for m in markers:
        # Get 3D-2D point correspondences
        # marker_size is the physical side length in meters
        marker_size = 0.05
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(m, marker_size)
        
        # Estimate pose
        retval, rvec, tvec = cv.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)

        if retval:
            # Draw axis
            cv.aruco2.drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
    # [pose_estimation]

def grid_board_tutorial():
    # [grid_board]
    # Create a 4x3 grid board
    grid_size = (4, 3)
    dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12
    board_img = cv.aruco2.getGridBoardImage(grid_size, dictionary, bitSize=25)
    cv.imwrite("board.png", board_img)

    # Detect board in an image
    img = cv.imread("board_photo.jpg")
    found, board = cv.aruco2.detectGridBoard(img, grid_size, dictionary)

    if found:
        # Draw detected board
        cv.aruco2.drawGridBoard(img, board, drawMarkerIds=True)
        
        # Estimate board pose
        marker_size = 0.03  # physical size of each marker in meters
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(board, marker_size)
        
        # Load your camera calibration parameters
        # retval, rvec, tvec = cv.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
    # [grid_board]

def diamond_tutorial():
    # [diamonds]
    # Create a diamond (2x2 markers)
    dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12
    ids = (10, 11, 12, 13) # 4 ids
    diamond_img = cv.aruco2.getDiamondImage(dictionary, ids, bitSize=20)
    cv.imwrite("diamond.png", diamond_img)

    # Detect diamonds
    img = cv.imread("diamond_photo.jpg")
    diamonds = cv.aruco2.detectDiamonds(img, dictionary)

    for d in diamonds:
        print(f"Detected diamond with IDs: {d.id}")
        cv.aruco2.drawDiamonds(img, [d], drawMarkerIds=True)
        
        # Estimate diamond pose (uses 9 points: 4 corners + 4 edge centers + center)
        marker_size = 0.05
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(d, marker_size)
    # [diamonds]

def fractal_tutorial():
    # [fractals]
    # Create a fractal marker (multi-scale nested markers)
    fractal_type = cv.aruco2.FRACTAL_5L_6
    fractal_img = cv.aruco2.getFractalMarkerImage(fractal_type, bitSize=20)
    cv.imwrite("fractal.png", fractal_img)

    # Detect fractals
    img = cv.imread("fractal_photo.jpg")
    fractals = cv.aruco2.detectFractals(img, fractal_type)

    for f in fractals:
        print(f"Detected fractal ID: {f.id}")
        cv.aruco2.drawFractals(img, [f], drawAllImagePoints=True)
        
        # Estimate fractal pose
        marker_size = 0.1 # physical size of the outer marker
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(f, marker_size)
    # [fractals]

if __name__ == "__main__":
    marker_creation()
    # Mock some data for demonstration if needed, but this is mainly for snippets
