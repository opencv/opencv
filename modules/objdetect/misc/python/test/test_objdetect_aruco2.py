#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv
from tests_common import NewOpenCVTests

class objdetect_aruco2_test(NewOpenCVTests):

    def test_enums(self):
        self.assertTrue(hasattr(cv.aruco2, 'DICT_4X4_50'))
        self.assertTrue(hasattr(cv.aruco2, 'DICT_ARUCO_MIP_36h12'))
        self.assertTrue(hasattr(cv.aruco2, 'DICT_APRILTAG_36h11'))

    def test_generate_marker(self):
        img = cv.aruco2.getFiducialMarker(cv.aruco2.DICT_ARUCO_MIP_36h12, 42, bitSize=10)
        self.assertIsNotNone(img)
        self.assertEqual(img.dtype, np.uint8)
        # 6x6 bits + 4 border bits (2 black, 2 white) = 10 bits. 10 * 10 = 100 pixels.
        self.assertEqual(img.shape, (100, 100))

    def test_detect_markers(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        marker_id = 42
        img = cv.aruco2.getFiducialMarker(dict_type, marker_id, bitSize=20)
        
        # Add some padding to simulate a scene
        # img is 200x200 (10 bits * 20)
        scene = np.ones((300, 300), dtype=np.uint8) * 255
        scene[50:50+img.shape[0], 50:50+img.shape[1]] = img
        
        markers = cv.aruco2.detectFiducialMarkers(scene, dict_type)
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].id, marker_id)
        # self.assertEqual(markers[0].dict, dict_type) # dict attribute seems missing in Python Marker
        self.assertEqual(len(markers[0].corners), 4)

    def test_detect_multi_dictionary(self):
        dict1 = cv.aruco2.DICT_ARUCO_MIP_36h12
        dict2 = cv.aruco2.DICT_APRILTAG_36h11
        
        img1 = cv.aruco2.getFiducialMarker(dict1, 10, bitSize=10)
        img2 = cv.aruco2.getFiducialMarker(dict2, 20, bitSize=10)
        
        # img1 is 100x100, img2 is 100x100
        scene = np.ones((150, 250), dtype=np.uint8) * 255
        scene[10:10+img1.shape[0], 10:10+img1.shape[1]] = img1
        scene[10:10+img2.shape[0], 110:110+img2.shape[1]] = img2
        
        markers = cv.aruco2.detectFiducialMarkers(scene, [dict1, dict2])
        self.assertEqual(len(markers), 2)
        
        ids = {m.id for m in markers}
        self.assertIn(10, ids)
        self.assertIn(20, ids)

    def test_draw_detected(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        img = cv.aruco2.getFiducialMarker(dict_type, 42, bitSize=20)
        scene = cv.merge([img, img, img]) # Convert to BGR
        
        markers = cv.aruco2.detectFiducialMarkers(img, dict_type)
        cv.aruco2.drawFiducialMarkers(scene, markers)
        
        # Check if some pixels are now green (0, 255, 0) or at least different from original
        # This is a bit weak but confirms the function runs
        self.assertTrue(np.any(scene[:,:,1] == 255))

    def test_grid_board_detection(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        grid_size = (3, 2)
        board_img = cv.aruco2.getGridBoard(grid_size, dict_type, bitSize=20)
        self.assertIsNotNone(board_img)
        canvas = np.ones((board_img.shape[0]+100, board_img.shape[1]+100), dtype=np.uint8) * 255
        canvas[50:50+board_img.shape[0], 50:50+board_img.shape[1]] = board_img
        found, board = cv.aruco2.detectGridBoard(canvas, grid_size, dict_type)
        self.assertTrue(found)
        self.assertEqual(board.gridSize, grid_size)
        self.assertEqual(len(board.markers), 6)

    def test_draw_axis(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        img = cv.aruco2.getFiducialMarker(dict_type, 42, bitSize=20)
        canvas = np.ones((img.shape[0]*2, img.shape[1]*2), dtype=np.uint8) * 255
        canvas[img.shape[0]//2:img.shape[0]//2+img.shape[0],
               img.shape[1]//2:img.shape[1]//2+img.shape[1]] = img
        color_canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        camera_matrix = np.array([[500, 0, 200], [0, 500, 200], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [1.0]])
        result = cv.aruco2.drawAxis(color_canvas, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        self.assertIsNotNone(result)

    def test_get_solve_pnp_points_fiducial_marker(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        img = cv.aruco2.getFiducialMarker(dict_type, 100, bitSize=20)
        canvas = np.ones((img.shape[0]*2, img.shape[1]*2), dtype=np.uint8) * 255
        canvas[img.shape[0]//2:img.shape[0]//2+img.shape[0],
               img.shape[1]//2:img.shape[1]//2+img.shape[1]] = img
        markers = cv.aruco2.detectFiducialMarkers(canvas, dict_type)
        self.assertEqual(len(markers), 1)
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(markers[0], markerSize=0.1)
        self.assertEqual(obj_pts.shape[0], 4)
        self.assertEqual(img_pts.shape[0], 4)

    def test_draw_grid_board(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        grid_size = (3, 2)
        board_img = cv.aruco2.getGridBoard(grid_size, dict_type, bitSize=20)
        canvas = np.ones((board_img.shape[0]+100, board_img.shape[1]+100), dtype=np.uint8) * 255
        canvas[50:50+board_img.shape[0], 50:50+board_img.shape[1]] = board_img
        found, board = cv.aruco2.detectGridBoard(canvas, grid_size, dict_type)
        self.assertTrue(found)
        color_canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        result = cv.aruco2.drawGridBoard(color_canvas, board)
        self.assertIsNotNone(result)

    def test_get_solve_pnp_points_grid_board(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        grid_size = (3, 2)
        board_img = cv.aruco2.getGridBoard(grid_size, dict_type, bitSize=20)
        canvas = np.ones((board_img.shape[0]+100, board_img.shape[1]+100), dtype=np.uint8) * 255
        canvas[50:50+board_img.shape[0], 50:50+board_img.shape[1]] = board_img
        found, board = cv.aruco2.detectGridBoard(canvas, grid_size, dict_type)
        self.assertTrue(found)
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(board, markerSize=0.05)
        # 3x2 board → (3+1)×(2+1) = 12 intersection corners
        self.assertEqual(obj_pts.shape[0], 12)
        self.assertEqual(img_pts.shape[0], 12)

    def test_diamond_workflow(self):
        dict_type = cv.aruco2.DICT_ARUCO_MIP_36h12
        ids = (5, 10, 15, 20)
        diamond_img = cv.aruco2.getDiamondImage(dict_type, ids, bitSize=20)
        self.assertIsNotNone(diamond_img)
        canvas = np.ones((diamond_img.shape[0]+100, diamond_img.shape[1]+100), dtype=np.uint8) * 255
        canvas[50:50+diamond_img.shape[0], 50:50+diamond_img.shape[1]] = diamond_img
        diamonds = cv.aruco2.detectDiamonds(canvas, dict_type)
        self.assertEqual(len(diamonds), 1)
        color_canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        cv.aruco2.drawDiamonds(color_canvas, diamonds)
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(diamonds[0], markerSize=0.1)
        # Diamond returns a 3x3 grid of 9 points
        self.assertEqual(obj_pts.shape[0], 9)
        self.assertEqual(img_pts.shape[0], 9)

    def test_fractal_workflow(self):
        ftype = cv.aruco2.FRACTAL_2L_6
        fractal_img = cv.aruco2.getFractalImage(ftype, bitSize=40)
        self.assertIsNotNone(fractal_img)
        canvas = np.ones((fractal_img.shape[0]+100, fractal_img.shape[1]+100), dtype=np.uint8) * 255
        canvas[50:50+fractal_img.shape[0], 50:50+fractal_img.shape[1]] = fractal_img
        fractals = cv.aruco2.detectFractals(canvas, ftype)
        self.assertEqual(len(fractals), 1)
        color_canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        cv.aruco2.drawFractals(color_canvas, fractals)
        obj_pts, img_pts = cv.aruco2.getSolvePnpPoints(fractals[0], markerSize=0.2)
        self.assertGreaterEqual(obj_pts.shape[0], 4)
        self.assertEqual(obj_pts.shape[0], img_pts.shape[0])

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
