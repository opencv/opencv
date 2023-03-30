from __future__ import print_function

import os, tempfile, numpy as np

import sys
sys.path.append("../../../../../doc/pattern_tools")
import gen_pattern

import cv2 as cv

from tests_common import NewOpenCVTests

class aruco_objdetect_test(NewOpenCVTests):

    def test_aruco_dicts(self):
        cols = 3
        rows = 5
        square_size = 100
        aruco_type_ = [cv.aruco.DICT_4X4_1000, cv.aruco.DICT_5X5_1000, cv.aruco.DICT_6X6_1000,
                       cv.aruco.DICT_7X7_1000, cv.aruco.DICT_ARUCO_ORIGINAL, cv.aruco.DICT_APRILTAG_16h5,
                       cv.aruco.DICT_APRILTAG_25h9, cv.aruco.DICT_APRILTAG_36h10, cv.aruco.DICT_APRILTAG_36h11]
        aruco_type_str_ = ['DICT_4X4_1000','DICT_5X5_1000', 'DICT_6X6_1000',
                       'DICT_7X7_1000', 'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16h5',
                       'DICT_APRILTAG_25h9', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11']
        marker_size = 0.8*square_size
        board_width = cols*square_size
        board_height = rows*square_size

        for aruco_type_i in range(len(aruco_type_)):
            #draw desk using opencv
            aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type_[aruco_type_i])
            board = cv.aruco.CharucoBoard((cols, rows), square_size, marker_size, aruco_dict)
            charuco_detector = cv.aruco.CharucoDetector(board)
            markerImage = board.generateImage((cols*square_size*10, rows*square_size*10))
            cv.imwrite("cv_marker.png", markerImage)
            from_cv_img = cv.imread('cv_marker.png')

            #draw desk using svg
            pm = gen_pattern.PatternMaker(cols, rows, "out.svg", "px", square_size, 0, board_width,
                              board_height, "charuco_checkboard", marker_size, aruco_type_str_[aruco_type_i])
            pm.make_charuco_board()
            pm.save()

            drawing = svg2rlg('out.svg')
            renderPM.drawToFile(drawing, 'svg_marker.png', fmt='PNG', dpi=720)
            from_svg_img = cv.imread('svg_marker.png')

            #test
            _charucoCorners, _charucoIds, markerCorners_svg, markerIds_svg = charuco_detector.detectBoard(from_svg_img)
            _charucoCorners, _charucoIds, markerCorners_cv, markerIds_cv = charuco_detector.detectBoard(from_cv_img)

            np.testing.assert_allclose(markerCorners_svg, markerCorners_cv, 0.1, 0.1)
            np.testing.assert_allclose(markerIds_svg, markerIds_cv, 0.1, 0.1)
            

    def test_aruco_marker_sizes(self): 
        cols = 3
        rows = 5
        square_size = 100
        aruco_type =  cv.aruco.DICT_5X5_1000
        aruco_type_str = 'DICT_5X5_1000'
        marker_sizes_rate =[0.1, 0.25, 0.5, 0.75, 0.99]
        board_width = cols*square_size
        board_height = rows*square_size

        for marker_s_rate in marker_sizes_rate:
            marker_size = marker_s_rate*square_size
            #draw desk using opencv
            aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type)
            board = cv.aruco.CharucoBoard((cols, rows), square_size, marker_size, aruco_dict)
            charuco_detector = cv.aruco.CharucoDetector(board)
            markerImage = board.generateImage((cols*square_size*10, rows*square_size*10))
            cv.imwrite("cv_marker.png", markerImage)
            from_cv_img = cv.imread('cv_marker.png')

            #draw desk using svg
            pm = gen_pattern.PatternMaker(cols, rows, "out.svg", "px", square_size, 0, board_width,
                              board_height, "charuco_checkboard", marker_size, aruco_type_str+".json")
            pm.make_charuco_board()
            pm.save()

            drawing = svg2rlg('out.svg')
            renderPM.drawToFile(drawing, 'svg_marker.png', fmt='PNG', dpi=720)
            from_svg_img = cv.imread('svg_marker.png')

            #test
            _charucoCorners, _charucoIds, markerCorners_svg, markerIds_svg = charuco_detector.detectBoard(from_svg_img)
            _charucoCorners, _charucoIds, markerCorners_cv, markerIds_cv = charuco_detector.detectBoard(from_cv_img)

            np.testing.assert_allclose(markerCorners_svg, markerCorners_cv, 0.1, 0.1)
            np.testing.assert_allclose(markerIds_svg, markerIds_cv, 0.1, 0.1)
