from __future__ import print_function

import os, tempfile, numpy as np

import sys
import cv2 as cv
from tests_common import NewOpenCVTests
import gen_pattern

class aruco_objdetect_test(NewOpenCVTests):

    def test_aruco_dicts(self):
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
        except:
            raise self.skipTest("libraies svglib and reportlab not found")

        cols = 3
        rows = 5
        square_size = 100
        aruco_type = [cv.aruco.DICT_4X4_1000, cv.aruco.DICT_5X5_1000, cv.aruco.DICT_6X6_1000,
                       cv.aruco.DICT_7X7_1000, cv.aruco.DICT_ARUCO_ORIGINAL, cv.aruco.DICT_APRILTAG_16h5,
                       cv.aruco.DICT_APRILTAG_25h9, cv.aruco.DICT_APRILTAG_36h10, cv.aruco.DICT_APRILTAG_36h11]
        aruco_type_str = ['DICT_4X4_1000','DICT_5X5_1000', 'DICT_6X6_1000',
                       'DICT_7X7_1000', 'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16h5',
                       'DICT_APRILTAG_25h9', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11']
        marker_size = 0.8*square_size
        board_width = cols*square_size
        board_height = rows*square_size

        for aruco_type_i in range(len(aruco_type)):
            #draw desk using opencv
            aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type[aruco_type_i])
            board = cv.aruco.CharucoBoard((cols, rows), square_size, marker_size, aruco_dict)
            charuco_detector = cv.aruco.CharucoDetector(board)
            markerImage = board.generateImage((cols*square_size*10, rows*square_size*10))
            fd, filepng = tempfile.mkstemp(prefix="cv_marker", suffix=".png")
            os.close(fd)
            from_cv_img = markerImage

            #draw desk using svg
            fd1, filesvg = tempfile.mkstemp(prefix="out", suffix=".svg")
            os.close(fd1)
            fd2, filepng2 = tempfile.mkstemp(prefix="svg_marker", suffix=".png")
            os.close(fd2)
            basedir = os.path.abspath(os.path.dirname(__file__))
            pm = gen_pattern.PatternMaker(cols, rows, filesvg, "px", square_size, 0, board_width,
                        board_height, "charuco_checkboard", marker_size,
                        os.path.join(basedir, aruco_type_str[aruco_type_i]+'.json.gz'))
            pm.make_charuco_board()
            pm.save()
            drawing = svg2rlg(filesvg)
            renderPM.drawToFile(drawing, filepng, fmt='PNG', dpi=720)
            from_svg_img = cv.imread(filepng2)

            #test
            _charucoCorners, _charucoIds, markerCorners_svg, markerIds_svg = charuco_detector.detectBoard(from_svg_img)
            _charucoCorners, _charucoIds, markerCorners_cv, markerIds_cv = charuco_detector.detectBoard(from_cv_img)

            np.testing.assert_allclose(markerCorners_svg, markerCorners_cv, 0.1, 0.1)
            np.testing.assert_allclose(markerIds_svg, markerIds_cv, 0.1, 0.1)

    def test_aruco_marker_sizes(self):
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
        except:
            raise self.skipTest("libraies svglib and reportlab not found")

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
            fd, filepng = tempfile.mkstemp(prefix="cv_marker", suffix=".png")
            os.close(fd)
            from_cv_img = markerImage

            #draw desk using svg
            fd1, filesvg = tempfile.mkstemp(prefix="out", suffix=".svg")
            os.close(fd1)
            fd2, filepng2 = tempfile.mkstemp(prefix="svg_marker", suffix=".png")
            os.close(fd2)
            basedir = os.path.abspath(os.path.dirname(__file__))
            pm = gen_pattern.PatternMaker(cols, rows, filesvg, "px", square_size, 0, board_width,
                        board_height, "charuco_checkboard", marker_size, os.path.join(basedir, aruco_type_str+'.json.gz'))
            pm.make_charuco_board()
            pm.save()
            drawing = svg2rlg(filesvg)
            renderPM.drawToFile(drawing, filepng, fmt='PNG', dpi=720)
            from_svg_img = cv.imread(filepng2)

            #test
            _charucoCorners, _charucoIds, markerCorners_svg, markerIds_svg = charuco_detector.detectBoard(from_svg_img)
            _charucoCorners, _charucoIds, markerCorners_cv, markerIds_cv = charuco_detector.detectBoard(from_cv_img)

            np.testing.assert_allclose(markerCorners_svg, markerCorners_cv, 0.1, 0.1)
            np.testing.assert_allclose(markerIds_svg, markerIds_cv, 0.1, 0.1)
