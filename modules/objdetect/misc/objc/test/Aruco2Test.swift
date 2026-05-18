//
//  Aruco2Test.swift
//
//  Tests for ArUco2 ObjC/Swift bindings.
//

import XCTest
import OpenCV

class Aruco2Test: OpenCVTestCase {

    // MARK: - Enum constants

    func testEnumConstants() {
        XCTAssertEqual(Aruco2_DICT_4X4_50.rawValue, 0)
        XCTAssertEqual(Aruco2_DICT_ARUCO_MIP_36h12.rawValue, 21)
        XCTAssertEqual(Aruco2_DICT_APRILTAG_36h11.rawValue, 20)
        XCTAssertEqual(Aruco2_FRACTAL_2L_6.rawValue, 0)
        XCTAssertEqual(Aruco2_FRACTAL_5L_6.rawValue, 3)
    }

    // MARK: - getFiducialMarker

    func testGetFiducialMarkerSize() {
        // DICT_ARUCO_MIP_36h12: 6x6 bits + 2 black + 2 white border = 10 bits per side
        let img = Mat()
        Aruco2.getFiducialMarker(img: img,
                                 dictionary: Aruco2_DICT_ARUCO_MIP_36h12.rawValue,
                                 id: 42,
                                 bitSize: 10,
                                 externalBorder: true)
        XCTAssertFalse(img.empty())
        XCTAssertEqual(img.rows(), 100)   // 10 bits * 10 px/bit
        XCTAssertEqual(img.cols(), 100)
    }

    func testGetFiducialMarkerDefaultParams() {
        let img = Mat()
        Aruco2.getFiducialMarker(img: img,
                                 dictionary: Aruco2_DICT_4X4_50.rawValue,
                                 id: 0)
        XCTAssertFalse(img.empty())
    }

    func testGetFiducialMarkerNoExternalBorder() {
        let img = Mat()
        Aruco2.getFiducialMarker(img: img,
                                 dictionary: Aruco2_DICT_4X4_50.rawValue,
                                 id: 1,
                                 bitSize: 20,
                                 externalBorder: false)
        XCTAssertFalse(img.empty())
    }

    // MARK: - detectFiducialMarkers (single dict)

    func testDetectFiducialMarkersBasic() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let markerImg = Mat()
        Aruco2.getFiducialMarker(img: markerImg, dictionary: dictType, id: 42, bitSize: 20, externalBorder: true)

        let scene = Mat(rows: markerImg.rows() + 100,
                        cols: markerImg.cols() + 100,
                        type: CvType.CV_8UC1,
                        scalar: Scalar(255))
        markerImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + markerImg.rows(),
                                        colStart: 50, colEnd: 50 + markerImg.cols()))

        let markers = Aruco2.detectFiducialMarkers(image: scene, dict: dictType)
        XCTAssertEqual(markers.count, 1)
        XCTAssertEqual(markers[0].id, 42)
        XCTAssertEqual(markers[0].corners.count, 4)
    }

    func testDetectFiducialMarkersWithParams() {
        let dictType = Aruco2_DICT_4X4_50.rawValue
        let markerImg = Mat()
        Aruco2.getFiducialMarker(img: markerImg, dictionary: dictType, id: 0, bitSize: 20, externalBorder: true)

        let scene = Mat(rows: markerImg.rows() + 100,
                        cols: markerImg.cols() + 100,
                        type: CvType.CV_8UC1,
                        scalar: Scalar(255))
        markerImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + markerImg.rows(),
                                        colStart: 50, colEnd: 50 + markerImg.cols()))

        let params = Aruco2DetectionParameters()
        params.boxFilterSize = 15
        let markers = Aruco2.detectFiducialMarkers(image: scene, dict: dictType, params: params)
        XCTAssertEqual(markers.count, 1)
        XCTAssertEqual(markers[0].id, 0)
    }

    // MARK: - detectFiducialMarkers (multi-dict)

    func testDetectFiducialMarkersMultiDict() {
        let dict1 = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let dict2 = Aruco2_DICT_APRILTAG_36h11.rawValue

        let img1 = Mat()
        Aruco2.getFiducialMarker(img: img1, dictionary: dict1, id: 10, bitSize: 10, externalBorder: true)
        let img2 = Mat()
        Aruco2.getFiducialMarker(img: img2, dictionary: dict2, id: 20, bitSize: 10, externalBorder: true)

        let h = img1.rows()
        let w = img1.cols()
        let scene = Mat(rows: h + 100, cols: w * 2 + 150, type: CvType.CV_8UC1, scalar: Scalar(255))
        img1.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + h, colStart: 50, colEnd: 50 + w))
        img2.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + h, colStart: w + 100, colEnd: w + 100 + w))

        let markers = Aruco2.detectFiducialMarkersMulti(image: scene,
                                                         dicts: [NSNumber(value: dict1), NSNumber(value: dict2)],
                                                         params: nil)
        XCTAssertEqual(markers.count, 2)
        let ids = Set(markers.map { $0.id })
        XCTAssertTrue(ids.contains(10))
        XCTAssertTrue(ids.contains(20))
    }

    // MARK: - drawFiducialMarkers

    func testDrawFiducialMarkers() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let markerImg = Mat()
        Aruco2.getFiducialMarker(img: markerImg, dictionary: dictType, id: 42, bitSize: 20, externalBorder: true)

        let scene = Mat(rows: markerImg.rows() + 100,
                        cols: markerImg.cols() + 100,
                        type: CvType.CV_8UC1,
                        scalar: Scalar(255))
        markerImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + markerImg.rows(),
                                        colStart: 50, colEnd: 50 + markerImg.cols()))

        let markers = Aruco2.detectFiducialMarkers(image: scene, dict: dictType)
        XCTAssertFalse(markers.isEmpty)

        // Draw on a 3-channel copy
        let colorScene = Mat(rows: scene.rows(), cols: scene.cols(),
                             type: CvType.CV_8UC3, scalar: Scalar(255, 255, 255))
        Aruco2.drawFiducialMarkers(image: colorScene, markers: markers, borderColor: Scalar(0, 255, 0, 255))
        XCTAssertFalse(colorScene.empty())
    }

    // MARK: - DetectionParameters

    func testDetectionParametersDefaults() {
        let params = Aruco2DetectionParameters()
        XCTAssertEqual(params.boxFilterSize, 15)
        XCTAssertEqual(params.thres, 3)
        XCTAssertEqual(params.minSize, 10)
        XCTAssertEqual(params.maxAttemptsPerCandidate, 5)
        XCTAssertEqual(params.markerBorderBits, 1)
        XCTAssertEqual(params.errorCorrectionRate, 0.0, accuracy: 1e-6)
        XCTAssertFalse(params.detectInvertedMarker)
    }

    func testDetectionParametersSetters() {
        let params = Aruco2DetectionParameters()
        params.boxFilterSize = 21
        params.errorCorrectionRate = 0.5
        params.detectInvertedMarker = true
        XCTAssertEqual(params.boxFilterSize, 21)
        XCTAssertEqual(params.errorCorrectionRate, 0.5, accuracy: 1e-6)
        XCTAssertTrue(params.detectInvertedMarker)
    }

    // MARK: - getSolvePnpPoints (FiducialMarker)

    func testGetSolvePnpPointsFiducialMarker() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let markerImg = Mat()
        Aruco2.getFiducialMarker(img: markerImg, dictionary: dictType, id: 100, bitSize: 20, externalBorder: false)

        let rowOff = markerImg.rows() / 2
        let colOff = markerImg.cols() / 2
        let scene = Mat(rows: markerImg.rows() * 2, cols: markerImg.cols() * 2,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        markerImg.copy(to: scene.submat(rowStart: rowOff, rowEnd: rowOff + markerImg.rows(),
                                        colStart: colOff, colEnd: colOff + markerImg.cols()))

        let markers = Aruco2.detectFiducialMarkers(image: scene, dict: dictType)
        XCTAssertEqual(markers.count, 1)

        let objPoints = Mat()
        let imgPoints = Mat()
        Aruco2.getSolvePnpPoints(marker: markers[0], objPoints: objPoints, imgPoints: imgPoints, markerSize: 0.1)
        XCTAssertEqual(objPoints.total(), 4)
        XCTAssertEqual(imgPoints.total(), 4)
    }

    // MARK: - drawAxis

    func testDrawAxis() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let markerImg = Mat()
        Aruco2.getFiducialMarker(img: markerImg, dictionary: dictType, id: 42, bitSize: 20, externalBorder: true)

        let colorImg = Mat(rows: markerImg.rows(), cols: markerImg.cols(),
                           type: CvType.CV_8UC3, scalar: Scalar(255, 255, 255))

        let cameraMatrix = Mat(rows: 3, cols: 3, type: CvType.CV_64F)
        try? cameraMatrix.put(row: 0, col: 0, data: [500.0, 0.0, Double(colorImg.cols()) / 2,
                                                      0.0, 500.0, Double(colorImg.rows()) / 2,
                                                      0.0, 0.0, 1.0] as [Double])
        let distCoeffs = Mat.zeros(4, cols: 1, type: CvType.CV_64F)
        let rvec = Mat.zeros(3, cols: 1, type: CvType.CV_64F)
        let tvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try? tvec.put(row: 0, col: 0, data: [0.0, 0.0, 1.0] as [Double])

        Aruco2.drawAxis(image: colorImg, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs,
                        rvec: rvec, tvec: tvec, length: 0.1)
        XCTAssertFalse(colorImg.empty())
    }

    // MARK: - GridBoard

    func testGetGridBoard() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let boardSize = Size2i(width: 3, height: 2)
        let boardImg = Mat()
        Aruco2.getGridBoard(img: boardImg, boardSize: boardSize, dict: dictType)
        XCTAssertFalse(boardImg.empty())
    }

    func testDetectGridBoard() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let gridSize = Size2i(width: 3, height: 2)
        let boardImg = Mat()
        Aruco2.getGridBoard(img: boardImg, boardSize: gridSize, dict: dictType, bitSize: 20, ids: nil)

        let scene = Mat(rows: boardImg.rows() + 100, cols: boardImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        boardImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + boardImg.rows(),
                                       colStart: 50, colEnd: 50 + boardImg.cols()))

        let board = Aruco2GridBoard()
        let found = Aruco2.detectGridBoard(image: scene, gridSize: gridSize, dict: dictType, board: board)
        XCTAssertTrue(found)
        XCTAssertGreaterThan(board.markers.count, 0)
    }

    func testGetSolvePnpPointsGridBoard() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let gridSize = Size2i(width: 3, height: 2)
        let boardImg = Mat()
        Aruco2.getGridBoard(img: boardImg, boardSize: gridSize, dict: dictType, bitSize: 20, ids: nil)

        let scene = Mat(rows: boardImg.rows() + 100, cols: boardImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        boardImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + boardImg.rows(),
                                       colStart: 50, colEnd: 50 + boardImg.cols()))

        let board = Aruco2GridBoard()
        let found = Aruco2.detectGridBoard(image: scene, gridSize: gridSize, dict: dictType, board: board)
        XCTAssertTrue(found)

        let objPoints = Mat()
        let imgPoints = Mat()
        Aruco2.getSolvePnpPoints(board: board, objPoints: objPoints, imgPoints: imgPoints, markerSize: 0.05)
        XCTAssertGreaterThan(objPoints.total(), 0)
        XCTAssertEqual(objPoints.total(), imgPoints.total())
    }

    func testDrawGridBoard() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let gridSize = Size2i(width: 3, height: 2)
        let boardImg = Mat()
        Aruco2.getGridBoard(img: boardImg, boardSize: gridSize, dict: dictType, bitSize: 20, ids: nil)

        let scene = Mat(rows: boardImg.rows() + 100, cols: boardImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        boardImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + boardImg.rows(),
                                       colStart: 50, colEnd: 50 + boardImg.cols()))

        let board = Aruco2GridBoard()
        let found = Aruco2.detectGridBoard(image: scene, gridSize: gridSize, dict: dictType, board: board)
        XCTAssertTrue(found)

        let colorScene = Mat(rows: scene.rows(), cols: scene.cols(),
                             type: CvType.CV_8UC3, scalar: Scalar(255, 255, 255))
        Aruco2.drawGridBoard(image: colorScene, board: board, color: Scalar(0, 255, 0, 255), drawMarkerIds: true)
        XCTAssertFalse(colorScene.empty())
    }

    // MARK: - Diamond

    func testGetDiamondImage() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let ids = Int4(v0: 5, v1: 10, v2: 15, v3: 20)
        let img = Mat()
        Aruco2.getDiamondImage(img: img, dictionary: dictType, ids: ids, bitSize: 20)
        XCTAssertFalse(img.empty())
    }

    func testDetectDiamonds() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let ids = Int4(v0: 5, v1: 10, v2: 15, v3: 20)
        let diamondImg = Mat()
        Aruco2.getDiamondImage(img: diamondImg, dictionary: dictType, ids: ids, bitSize: 20)

        let scene = Mat(rows: diamondImg.rows() + 100, cols: diamondImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        diamondImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + diamondImg.rows(),
                                         colStart: 50, colEnd: 50 + diamondImg.cols()))

        let diamonds = Aruco2.detectDiamonds(image: scene, dict: dictType)
        XCTAssertEqual(diamonds.count, 1)
    }

    func testGetSolvePnpPointsDiamond() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let ids = Int4(v0: 5, v1: 10, v2: 15, v3: 20)
        let diamondImg = Mat()
        Aruco2.getDiamondImage(img: diamondImg, dictionary: dictType, ids: ids, bitSize: 20)

        let scene = Mat(rows: diamondImg.rows() + 100, cols: diamondImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        diamondImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + diamondImg.rows(),
                                         colStart: 50, colEnd: 50 + diamondImg.cols()))

        let diamonds = Aruco2.detectDiamonds(image: scene, dict: dictType)
        XCTAssertEqual(diamonds.count, 1)

        let objPoints = Mat()
        let imgPoints = Mat()
        Aruco2.getSolvePnpPoints(diamond: diamonds[0], objPoints: objPoints, imgPoints: imgPoints, markerSize: 0.1)
        XCTAssertGreaterThan(objPoints.total(), 0)
        XCTAssertEqual(objPoints.total(), imgPoints.total())
    }

    func testDrawDiamonds() {
        let dictType = Aruco2_DICT_ARUCO_MIP_36h12.rawValue
        let ids = Int4(v0: 5, v1: 10, v2: 15, v3: 20)
        let diamondImg = Mat()
        Aruco2.getDiamondImage(img: diamondImg, dictionary: dictType, ids: ids, bitSize: 20)

        let scene = Mat(rows: diamondImg.rows() + 100, cols: diamondImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        diamondImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + diamondImg.rows(),
                                         colStart: 50, colEnd: 50 + diamondImg.cols()))

        let diamonds = Aruco2.detectDiamonds(image: scene, dict: dictType)
        let colorScene = Mat(rows: scene.rows(), cols: scene.cols(),
                             type: CvType.CV_8UC3, scalar: Scalar(255, 255, 255))
        Aruco2.drawDiamonds(image: colorScene, diamonds: diamonds, color: Scalar(0, 255, 0, 255), drawMarkerIds: false)
        XCTAssertFalse(colorScene.empty())
    }

    // MARK: - FractalMarker

    func testGetFractalImage() {
        let img = Mat()
        Aruco2.getFractalImage(img: img, ftype: Aruco2_FRACTAL_2L_6.rawValue, bitSize: 40)
        XCTAssertFalse(img.empty())
        XCTAssertEqual(img.rows(), img.cols())
    }

    func testDetectFractals() {
        let ftype = Aruco2_FRACTAL_2L_6.rawValue
        let fractalImg = Mat()
        Aruco2.getFractalImage(img: fractalImg, ftype: ftype, bitSize: 40)

        let scene = Mat(rows: fractalImg.rows() + 100, cols: fractalImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        fractalImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + fractalImg.rows(),
                                         colStart: 50, colEnd: 50 + fractalImg.cols()))

        let fractals = Aruco2.detectFractals(image: scene, ftype: ftype)
        XCTAssertEqual(fractals.count, 1)
        XCTAssertEqual(fractals[0].corners.count, 4)
    }

    func testGetSolvePnpPointsFractalMarker() {
        let ftype = Aruco2_FRACTAL_2L_6.rawValue
        let fractalImg = Mat()
        Aruco2.getFractalImage(img: fractalImg, ftype: ftype, bitSize: 40)

        let scene = Mat(rows: fractalImg.rows() + 100, cols: fractalImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        fractalImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + fractalImg.rows(),
                                         colStart: 50, colEnd: 50 + fractalImg.cols()))

        let fractals = Aruco2.detectFractals(image: scene, ftype: ftype)
        XCTAssertEqual(fractals.count, 1)

        let objPoints = Mat()
        let imgPoints = Mat()
        Aruco2.getSolvePnpPoints(fractal: fractals[0], objPoints: objPoints, imgPoints: imgPoints, markerSize: 0.2)
        XCTAssertGreaterThanOrEqual(objPoints.total(), 4)
        XCTAssertEqual(objPoints.total(), imgPoints.total())
    }

    func testDrawFractals() {
        let ftype = Aruco2_FRACTAL_2L_6.rawValue
        let fractalImg = Mat()
        Aruco2.getFractalImage(img: fractalImg, ftype: ftype, bitSize: 40)

        let scene = Mat(rows: fractalImg.rows() + 100, cols: fractalImg.cols() + 100,
                        type: CvType.CV_8UC1, scalar: Scalar(255))
        fractalImg.copy(to: scene.submat(rowStart: 50, rowEnd: 50 + fractalImg.rows(),
                                         colStart: 50, colEnd: 50 + fractalImg.cols()))

        let fractals = Aruco2.detectFractals(image: scene, ftype: ftype)
        let colorScene = Mat(rows: scene.rows(), cols: scene.cols(),
                             type: CvType.CV_8UC3, scalar: Scalar(255, 255, 255))
        Aruco2.drawFractals(image: colorScene, fractals: fractals, color: Scalar(0, 255, 0, 255), drawAllImagePoints: true)
        XCTAssertFalse(colorScene.empty())
    }

    // MARK: - FiducialMarker struct

    func testFiducialMarkerStruct() {
        let m = Aruco2FiducialMarker()
        XCTAssertEqual(m.id, -1)
        m.id = 42
        XCTAssertEqual(m.id, 42)
        XCTAssertEqual(m.corners.count, 0)
    }

    // MARK: - GridBoard struct

    func testGridBoardStruct() {
        let board = Aruco2GridBoard()
        board.gridSize = Size2i(width: 4, height: 3)
        XCTAssertEqual(board.gridSize.width, 4)
        XCTAssertEqual(board.gridSize.height, 3)
    }

    // MARK: - Diamond struct

    func testDiamondStruct() {
        let d = Aruco2Diamond()
        d.id = Int4(v0: 1, v1: 2, v2: 3, v3: 4)
        XCTAssertEqual(d.id.v0, 1)
        XCTAssertEqual(d.id.v3, 4)
    }

    // MARK: - FractalMarker struct

    func testFractalMarkerStruct() {
        let f = Aruco2FractalMarker()
        XCTAssertEqual(f.id, -1)
        f.id = 7
        XCTAssertEqual(f.id, 7)
    }
}
