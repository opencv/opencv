#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, tempfile, numpy as np
from math import pi

import cv2 as cv

from tests_common import NewOpenCVTests

def getSyntheticRT(yaw, pitch, distance):
    rvec = np.zeros((3, 1), np.float64)
    tvec = np.zeros((3, 1), np.float64)

    rotPitch = np.array([[-pitch], [0], [0]])
    rotYaw = np.array([[0], [yaw], [0]])

    rvec, tvec = cv.composeRT(rotPitch, np.zeros((3, 1), np.float64),
                              rotYaw, np.zeros((3, 1), np.float64))[:2]

    tvec = np.array([[0], [0], [distance]])
    return rvec, tvec

# see test_aruco_utils.cpp
def projectMarker(img, board, markerIndex, cameraMatrix, rvec, tvec, markerBorder):
    markerSizePixels = 100
    markerImg = cv.aruco.generateImageMarker(board.getDictionary(), board.getIds()[markerIndex], markerSizePixels, borderBits=markerBorder)

    distCoeffs = np.zeros((5, 1), np.float64)
    maxCoord = board.getRightBottomCorner()
    objPoints = board.getObjPoints()[markerIndex]
    for i in range(len(objPoints)):
        objPoints[i][0] -= maxCoord[0] / 2
        objPoints[i][1] -= maxCoord[1] / 2
        objPoints[i][2] -= maxCoord[2] / 2

    corners, _ = cv.projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs)

    originalCorners = np.array([
        [0, 0],
        [markerSizePixels, 0],
        [markerSizePixels, markerSizePixels],
        [0, markerSizePixels],
    ], np.float32)

    transformation = cv.getPerspectiveTransform(originalCorners, corners)

    borderValue = 127
    aux = cv.warpPerspective(markerImg, transformation, img.shape, None, cv.INTER_NEAREST, cv.BORDER_CONSTANT, borderValue)

    assert(img.shape == aux.shape)
    mask = (aux == borderValue).astype(np.uint8)
    img = img * mask + aux * (1 - mask)
    return img

def projectChessboard(squaresX, squaresY, squareSize, imageSize, cameraMatrix, rvec, tvec):
    img = np.ones(imageSize, np.uint8) * 255
    distCoeffs = np.zeros((5, 1), np.float64)
    for y in range(squaresY):
        startY = y * squareSize
        for x in range(squaresX):
            if (y % 2 != x % 2):
                continue
            startX = x * squareSize

            squareCorners = np.array([[startX - squaresX*squareSize/2,
                                       startY - squaresY*squareSize/2,
                                       0]], np.float32)
            squareCorners = np.stack((squareCorners[0],
                                      squareCorners[0] + [squareSize, 0, 0],
                                      squareCorners[0] + [squareSize, squareSize, 0],
                                      squareCorners[0] + [0, squareSize, 0]))

            projectedCorners, _ = cv.projectPoints(squareCorners, rvec, tvec, cameraMatrix, distCoeffs)
            projectedCorners = projectedCorners.astype(np.int64)
            projectedCorners = projectedCorners.reshape(1, 4, 2)
            img = cv.fillPoly(img, [projectedCorners], 0)

    return img

def projectCharucoBoard(board, cameraMatrix, yaw, pitch, distance, imageSize, markerBorder):
    rvec, tvec = getSyntheticRT(yaw, pitch, distance)

    img = np.ones(imageSize, np.uint8) * 255
    for indexMarker in range(len(board.getIds())):
        img = projectMarker(img, board, indexMarker, cameraMatrix, rvec, tvec, markerBorder)

    chessboard = projectChessboard(board.getChessboardSize()[0], board.getChessboardSize()[1],
                                   board.getSquareLength(), imageSize, cameraMatrix, rvec, tvec)

    chessboard = (chessboard != 0).astype(np.uint8)
    img = img * chessboard
    return img, rvec, tvec

class aruco_objdetect_test(NewOpenCVTests):

    def test_board(self):
        p1 = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        p2 = np.array([[1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 0, 0]], dtype=np.float32)
        objPoints = np.array([p1, p2])
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        ids = np.array([0, 1])

        board = cv.aruco.Board(objPoints, dictionary, ids)
        np.testing.assert_array_equal(board.getIds().squeeze(), ids)
        np.testing.assert_array_equal(np.ravel(np.array(board.getObjPoints())), np.ravel(np.concatenate([p1, p2])))

    def test_idsAccessibility(self):

        ids = np.arange(17)
        rev_ids = ids[::-1]

        aruco_dict  = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
        board = cv.aruco.CharucoBoard((7, 5), 1, 0.5, aruco_dict)

        np.testing.assert_array_equal(board.getIds().squeeze(), ids)

        board = cv.aruco.CharucoBoard((7, 5), 1, 0.5, aruco_dict, rev_ids)
        np.testing.assert_array_equal(board.getIds().squeeze(), rev_ids)

        board = cv.aruco.CharucoBoard((7, 5), 1, 0.5, aruco_dict, ids)
        np.testing.assert_array_equal(board.getIds().squeeze(), ids)

    def test_identify(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        expected_idx = 9
        expected_rotation = 2
        bit_marker = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)

        check, idx, rotation = aruco_dict.identify(bit_marker, 0)

        self.assertTrue(check, True)
        self.assertEqual(idx, expected_idx)
        self.assertEqual(rotation, expected_rotation)

    def test_getDistanceToId(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        idx = 7
        rotation = 3
        bit_marker = np.array([[0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
        dist = aruco_dict.getDistanceToId(bit_marker, idx)

        self.assertEqual(dist, 0)

    def test_aruco_detector(self):
        aruco_params = cv.aruco.DetectorParameters()
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
        id = 2
        marker_size = 100
        offset = 10
        img_marker = cv.aruco.generateImageMarker(aruco_dict, id, marker_size, aruco_params.markerBorderBits)
        img_marker = np.pad(img_marker, pad_width=offset, mode='constant', constant_values=255)
        gold_corners = np.array([[offset, offset],[marker_size+offset-1.0,offset],
                                 [marker_size+offset-1.0,marker_size+offset-1.0],
                                 [offset, marker_size+offset-1.0]], dtype=np.float32)
        corners, ids, rejected = aruco_detector.detectMarkers(img_marker)

        self.assertEqual(1, len(ids))
        self.assertEqual(id, ids[0])
        for i in range(0, len(corners)):
            np.testing.assert_array_equal(gold_corners, corners[i].reshape(4, 2))

    def test_aruco_detector_refine(self):
        aruco_params = cv.aruco.DetectorParameters()
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
        board_size = (3, 4)
        board = cv.aruco.GridBoard(board_size, 5.0, 1.0, aruco_dict)
        board_image = board.generateImage((board_size[0]*50, board_size[1]*50), marginSize=10)

        corners, ids, rejected = aruco_detector.detectMarkers(board_image)
        self.assertEqual(board_size[0]*board_size[1], len(ids))

        part_corners, part_ids, part_rejected = corners[:-1], ids[:-1], list(rejected)
        part_rejected.append(corners[-1])

        refine_corners, refine_ids, refine_rejected, recovered_ids = aruco_detector.refineDetectedMarkers(board_image, board, part_corners, part_ids, part_rejected)

        self.assertEqual(board_size[0] * board_size[1], len(refine_ids))
        self.assertEqual(1, len(recovered_ids))

        self.assertEqual(ids[-1], refine_ids[-1])
        self.assertEqual((1, 4, 2), refine_corners[0].shape)
        np.testing.assert_array_equal(corners, refine_corners)

    def test_charuco_refine(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
        board_size = (3, 4)
        board = cv.aruco.CharucoBoard(board_size, 1., .7, aruco_dict)
        aruco_detector = cv.aruco.ArucoDetector(aruco_dict)
        charuco_detector = cv.aruco.CharucoDetector(board)
        cell_size = 100
        image = board.generateImage((cell_size*board_size[0], cell_size*board_size[1]))
        camera = np.array([[1, 0, 0.5],
                           [0, 1, 0.5],
                           [0, 0, 1]])
        dist = np.array([0, 0, 0, 0, 0], dtype=np.float32).reshape(1, -1)

        # generate gold corners of the ArUco markers for the test
        gold_corners = np.array(board.getObjPoints())[:, :, 0:2]*cell_size

        # detect corners
        markerCorners, markerIds, _ = aruco_detector.detectMarkers(image)

        # test refine
        rejected = [markerCorners[-1]]
        markerCorners, markerIds = markerCorners[:-1], markerIds[:-1]
        markerCorners, markerIds, _, _ = aruco_detector.refineDetectedMarkers(image, board, markerCorners, markerIds,
                                                                              rejected, cameraMatrix=camera, distCoeffs=dist)

        charucoCorners, charucoIds, _, _ = charuco_detector.detectBoard(image, markerCorners=markerCorners,
                                                                        markerIds=markerIds)
        self.assertEqual(len(charucoIds), 6)
        self.assertEqual(len(markerIds), 6)

        for i, id in enumerate(markerIds.reshape(-1)):
            np.testing.assert_allclose(gold_corners[id], markerCorners[i].reshape(4, 2), 0.01, 1.)

    def test_write_read_dictionary(self):
        try:
            aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
            markers_gold = aruco_dict.bytesList

            # write aruco_dict
            fd, filename = tempfile.mkstemp(prefix="opencv_python_aruco_dict_", suffix=".yml")
            os.close(fd)

            fs_write = cv.FileStorage(filename, cv.FileStorage_WRITE)
            aruco_dict.writeDictionary(fs_write)
            fs_write.release()

            # reset aruco_dict
            aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

            # read aruco_dict
            fs_read = cv.FileStorage(filename, cv.FileStorage_READ)
            aruco_dict.readDictionary(fs_read.root())
            fs_read.release()

            # check equal
            self.assertEqual(aruco_dict.markerSize, 5)
            self.assertEqual(aruco_dict.maxCorrectionBits, 3)
            np.testing.assert_array_equal(aruco_dict.bytesList, markers_gold)

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_charuco_detector(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        board_size = (3, 3)
        board = cv.aruco.CharucoBoard(board_size, 1.0, .8, aruco_dict)
        charuco_detector = cv.aruco.CharucoDetector(board)
        cell_size = 100

        image = board.generateImage((cell_size*board_size[0], cell_size*board_size[1]))

        list_gold_corners = []
        for i in range(1, board_size[0]):
            for j in range(1, board_size[1]):
                list_gold_corners.append((j*cell_size, i*cell_size))
        gold_corners = np.array(list_gold_corners, dtype=np.float32)

        charucoCorners, charucoIds, markerCorners, markerIds = charuco_detector.detectBoard(image)

        self.assertEqual(len(charucoIds), 4)
        for i in range(0, 4):
            self.assertEqual(charucoIds[i], i)
        np.testing.assert_allclose(gold_corners, charucoCorners.reshape(-1, 2), 0.01, 0.1)

    def test_detect_diamonds(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        board_size = (3, 3)
        board = cv.aruco.CharucoBoard(board_size, 1.0, .8, aruco_dict)
        charuco_detector = cv.aruco.CharucoDetector(board)
        cell_size = 120

        image = board.generateImage((cell_size*board_size[0], cell_size*board_size[1]))

        list_gold_corners = [(cell_size, cell_size), (2*cell_size, cell_size), (2*cell_size, 2*cell_size),
                             (cell_size, 2*cell_size)]
        gold_corners = np.array(list_gold_corners, dtype=np.float32)

        diamond_corners, diamond_ids, marker_corners, marker_ids = charuco_detector.detectDiamonds(image)

        self.assertEqual(diamond_ids.size, 4)
        self.assertEqual(marker_ids.size, 4)
        for i in range(0, 4):
            self.assertEqual(diamond_ids[0][i], i)
        np.testing.assert_allclose(gold_corners, np.array(diamond_corners, dtype=np.float32).reshape(-1, 2), 0.01, 0.1)

    # check no segfault when cameraMatrix or distCoeffs are not initialized
    def test_charuco_no_segfault_params(self):
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        board = cv.aruco.CharucoBoard((10, 10), 0.019, 0.015, dictionary)
        charuco_parameters = cv.aruco.CharucoParameters()
        detector = cv.aruco.CharucoDetector(board)
        detector.setCharucoParameters(charuco_parameters)

        self.assertIsNone(detector.getCharucoParameters().cameraMatrix)
        self.assertIsNone(detector.getCharucoParameters().distCoeffs)

    def test_charuco_no_segfault_params_constructor(self):
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        board = cv.aruco.CharucoBoard((10, 10), 0.019, 0.015, dictionary)
        charuco_parameters = cv.aruco.CharucoParameters()
        detector = cv.aruco.CharucoDetector(board, charucoParams=charuco_parameters)

        self.assertIsNone(detector.getCharucoParameters().cameraMatrix)
        self.assertIsNone(detector.getCharucoParameters().distCoeffs)

    # similar to C++ test CV_CharucoDetection.accuracy
    def test_charuco_detector_accuracy(self):
        iteration = 0
        cameraMatrix = np.eye(3, 3, dtype=np.float64)
        imgSize = (500, 500)
        params = cv.aruco.DetectorParameters()
        params.minDistanceToBorder = 3

        board = cv.aruco.CharucoBoard((4, 4), 0.03, 0.015, cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250))
        detector = cv.aruco.CharucoDetector(board, detectorParams=params)

        cameraMatrix[0, 0] = cameraMatrix[1, 1] = 600
        cameraMatrix[0, 2] = imgSize[0] / 2
        cameraMatrix[1, 2] = imgSize[1] / 2

        # for different perspectives
        distCoeffs = np.zeros((5, 1), dtype=np.float64)
        for distance in [0.2, 0.4]:
            for yaw in range(-55, 51, 25):
                for pitch in range(-55, 51, 25):
                    markerBorder = iteration % 2 + 1
                    iteration += 1

                    # create synthetic image
                    img, rvec, tvec = projectCharucoBoard(board, cameraMatrix, yaw * pi / 180, pitch * pi / 180, distance, imgSize, markerBorder)

                    params.markerBorderBits = markerBorder
                    detector.setDetectorParameters(params)

                    if (iteration % 2 != 0):
                        charucoParameters = cv.aruco.CharucoParameters()
                        charucoParameters.cameraMatrix = cameraMatrix
                        charucoParameters.distCoeffs = distCoeffs
                        detector.setCharucoParameters(charucoParameters)

                    charucoCorners, charucoIds, corners, ids = detector.detectBoard(img)

                    self.assertGreater(len(ids), 0)

                    copyChessboardCorners = board.getChessboardCorners()
                    copyChessboardCorners -= np.array(board.getRightBottomCorner()) / 2

                    projectedCharucoCorners, _ = cv.projectPoints(copyChessboardCorners, rvec, tvec, cameraMatrix, distCoeffs)

                    if charucoIds is None:
                        self.assertEqual(iteration, 46)
                        continue

                    for i in range(len(charucoIds)):
                        currentId = charucoIds[i]
                        self.assertLess(currentId, len(board.getChessboardCorners()))

                        reprErr = cv.norm(charucoCorners[i] - projectedCharucoCorners[currentId])
                        self.assertLessEqual(reprErr, 5)

    def test_aruco_match_image_points(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        board_size = (3, 4)
        board = cv.aruco.GridBoard(board_size, 5.0, 1.0, aruco_dict)
        aruco_corners = np.array(board.getObjPoints())[:, :, :2]
        aruco_ids = board.getIds()
        obj_points, img_points = board.matchImagePoints(aruco_corners, aruco_ids)
        aruco_corners = aruco_corners.reshape(-1, 2)

        self.assertEqual(aruco_corners.shape[0], obj_points.shape[0])
        self.assertEqual(img_points.shape[0], obj_points.shape[0])
        self.assertEqual(2, img_points.shape[1])
        np.testing.assert_array_equal(aruco_corners, obj_points[:, :2].reshape(-1, 2))

    def test_charuco_match_image_points(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        board_size = (3, 4)
        board = cv.aruco.CharucoBoard(board_size, 5.0, 1.0, aruco_dict)
        chessboard_corners = np.array(board.getChessboardCorners())[:, :2]
        chessboard_ids = board.getIds()
        obj_points, img_points = board.matchImagePoints(chessboard_corners, chessboard_ids)

        self.assertEqual(chessboard_corners.shape[0], obj_points.shape[0])
        self.assertEqual(img_points.shape[0], obj_points.shape[0])
        self.assertEqual(2, img_points.shape[1])
        np.testing.assert_array_equal(chessboard_corners, obj_points[:, :2].reshape(-1, 2))

    def test_draw_detected_markers(self):
        detected_points = [[[10, 10], [50, 10], [50, 50], [10, 50]]]
        img = np.zeros((60, 60), dtype=np.uint8)

        # add extra dimension in Python to create Nx4 Mat with 2 channels
        points1 = np.array(detected_points).reshape(-1, 4, 1, 2)
        img = cv.aruco.drawDetectedMarkers(img, points1, borderColor=255)

        # check that the marker borders are painted
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(contours), 1)
        self.assertEqual(img[10, 10], 255)
        self.assertEqual(img[50, 10], 255)
        self.assertEqual(img[50, 50], 255)
        self.assertEqual(img[10, 50], 255)

        # must throw Exception without extra dimension
        points2 = np.array(detected_points)
        with self.assertRaises(Exception):
            img = cv.aruco.drawDetectedMarkers(img, points2, borderColor=255)

    def test_draw_detected_charuco(self):
        detected_points = [[[10, 10], [50, 10], [50, 50], [10, 50]]]
        img = np.zeros((60, 60), dtype=np.uint8)

        # add extra dimension in Python to create Nx1 Mat with 2 channels
        points = np.array(detected_points).reshape(-1, 1, 2)
        img = cv.aruco.drawDetectedCornersCharuco(img, points, cornerColor=255)

        # check that the 4 charuco corners are painted
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(contours), 4)
        for contour in contours:
            center_x = round(np.average(contour[:, 0, 0]))
            center_y = round(np.average(contour[:, 0, 1]))
            center = [center_x, center_y]
            self.assertTrue(center in detected_points[0])

        # must throw Exception without extra dimension
        points2 = np.array(detected_points)
        with self.assertRaises(Exception):
            img = cv.aruco.drawDetectedCornersCharuco(img, points2, borderColor=255)

    def test_draw_detected_diamonds(self):
        detected_points = [[[10, 10], [50, 10], [50, 50], [10, 50]]]
        img = np.zeros((60, 60), dtype=np.uint8)

        # add extra dimension in Python to create Nx4 Mat with 2 channels
        points = np.array(detected_points).reshape(-1, 4, 1, 2)
        img = cv.aruco.drawDetectedDiamonds(img, points, borderColor=255)

        # check that the diamonds borders are painted
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(contours), 1)
        self.assertEqual(img[10, 10], 255)
        self.assertEqual(img[50, 10], 255)
        self.assertEqual(img[50, 50], 255)
        self.assertEqual(img[10, 50], 255)

        # must throw Exception without extra dimension
        points2 = np.array(detected_points)
        with self.assertRaises(Exception):
            img = cv.aruco.drawDetectedDiamonds(img, points2, borderColor=255)

    def test_multi_dict_arucodetector(self):
        aruco_params = cv.aruco.DetectorParameters()
        aruco_dicts = [
                cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250),
                cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
            ]
        aruco_detector = cv.aruco.ArucoDetector(aruco_dicts, aruco_params)
        id = 2
        marker_size = 100
        offset = 10
        img_marker1 = cv.aruco.generateImageMarker(aruco_dicts[0], id, marker_size, aruco_params.markerBorderBits)
        img_marker1 = np.pad(img_marker1, pad_width=offset, mode='constant', constant_values=255)
        img_marker2 = cv.aruco.generateImageMarker(aruco_dicts[1], id, marker_size, aruco_params.markerBorderBits)
        img_marker2 = np.pad(img_marker2, pad_width=offset, mode='constant', constant_values=255)
        img_markers = np.concatenate((img_marker1, img_marker2), axis=1)

        corners, ids, rejected, dictIndices = aruco_detector.detectMarkersMultiDict(img_markers)

        self.assertEqual(2, len(ids))
        self.assertEqual(id, ids[0])
        self.assertEqual(id, ids[1])
        self.assertEqual(2, len(dictIndices))
        self.assertEqual(0, dictIndices[0])
        self.assertEqual(1, dictIndices[1])

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
