//
//  Calib3dTest.swift
//
//  Created by Giles Payne on 2020/05/26.
//

import XCTest
import OpenCV

class Calib3dTest: OpenCVTestCase {

    var size = Size()

    override func setUp() {
        super.setUp()
        size = Size(width: 3, height: 3)
    }

    override func tearDown() {
        super.tearDown()
    }

    func testComposeRTMatMatMatMatMatMat() throws {
        let rvec1 = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try rvec1.put(row: 0, col: 0, data: [0.5302828, 0.19925919, 0.40105945] as [Float])
        let tvec1 = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try tvec1.put(row: 0, col: 0, data: [0.81438506, 0.43713298, 0.2487897] as [Float])
        let rvec2 = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try rvec2.put(row: 0, col: 0, data: [0.77310503, 0.76209372, 0.30779448] as [Float])
        let tvec2 = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try tvec2.put(row: 0, col: 0, data: [0.70243168, 0.4784472, 0.79219002] as [Float])

        let rvec3 = Mat()
        let tvec3 = Mat()

        let outRvec = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try outRvec.put(row: 0, col: 0, data: [1.418641, 0.88665926, 0.56020796])
        let outTvec = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try outTvec.put(row: 0, col: 0, data: [1.4560841, 1.0680628, 0.81598103])

        Calib3d.composeRT(rvec1: rvec1, tvec1: tvec1, rvec2: rvec2, tvec2: tvec2, rvec3: rvec3, tvec3: tvec3)

        try assertMatEqual(outRvec, rvec3, OpenCVTestCase.EPS)
        try assertMatEqual(outTvec, tvec3, OpenCVTestCase.EPS)
    }

    func testFilterSpecklesMatDoubleIntDouble() throws {
        gray_16s_1024.copy(to: dst)
        let center = Point(x: gray_16s_1024.rows() / 2, y: gray_16s_1024.cols() / 2)
        Imgproc.circle(img: dst, center: center, radius: 1, color: Scalar.all(4096))

        try assertMatNotEqual(gray_16s_1024, dst)
        Calib3d.filterSpeckles(img: dst, newVal: 1024.0, maxSpeckleSize: 100, maxDiff: 0.0)
        try assertMatEqual(gray_16s_1024, dst)
    }

    func testFindChessboardCornersMatSizeMat() {
        let patternSize = Size(width: 9, height: 6)
        let corners = MatOfPoint2f()
        Calib3d.findChessboardCorners(image: grayChess, patternSize: patternSize, corners: corners)
        XCTAssertFalse(corners.empty())
    }

    func testFindChessboardCornersMatSizeMatInt() {
        let patternSize = Size(width: 9, height: 6)
        let corners = MatOfPoint2f()
        Calib3d.findChessboardCorners(image: grayChess, patternSize: patternSize, corners: corners, flags: Calib3d.CALIB_CB_ADAPTIVE_THRESH + Calib3d.CALIB_CB_NORMALIZE_IMAGE + Calib3d.CALIB_CB_FAST_CHECK)
        XCTAssertFalse(corners.empty())
    }

    func testFind4QuadCornerSubpix() {
        let patternSize = Size(width: 9, height: 6)
        let corners = MatOfPoint2f()
        let region_size = Size(width: 5, height: 5)
        Calib3d.findChessboardCorners(image: grayChess, patternSize: patternSize, corners: corners)
        Calib3d.find4QuadCornerSubpix(img: grayChess, corners: corners, region_size: region_size)
        XCTAssertFalse(corners.empty())
    }

    func testFindCirclesGridMatSizeMat() {
        let size = 300
        let img = Mat(rows:Int32(size), cols:Int32(size), type:CvType.CV_8U)
        img.setTo(scalar: Scalar(255))
        let centers = Mat()

        XCTAssertFalse(Calib3d.findCirclesGrid(image: img, patternSize: Size(width: 5, height: 5), centers: centers))

        for i in 0..<5 {
            for j in 0..<5 {
                let x = Int32(size * (2 * i + 1) / 10)
                let y = Int32(size * (2 * j + 1) / 10)
                let pt = Point(x: x, y: y)
                Imgproc.circle(img: img, center: pt, radius: 10, color: Scalar(0), thickness: -1)
            }
        }

        XCTAssert(Calib3d.findCirclesGrid(image: img, patternSize:Size(width:5, height:5), centers:centers))

        XCTAssertEqual(25, centers.rows())
        XCTAssertEqual(1, centers.cols())
        XCTAssertEqual(CvType.CV_32FC2, centers.type())
    }

    func testFindCirclesGridMatSizeMatInt() {
        let size:Int32 = 300
        let img = Mat(rows:size, cols: size, type: CvType.CV_8U)
        img.setTo(scalar: Scalar(255))
        let centers = Mat()

        XCTAssertFalse(Calib3d.findCirclesGrid(image: img, patternSize: Size(width: 3, height: 5), centers: centers, flags: Calib3d.CALIB_CB_CLUSTERING | Calib3d.CALIB_CB_ASYMMETRIC_GRID))

        let step = size * 2 / 15
        let offsetx = size / 6
        let offsety = (size - 4 * step) / 2
        for i:Int32 in 0...2 {
            for j:Int32 in 0...4 {
                let pt = Point(x: offsetx + (2 * i + j % 2) * step, y: offsety + step * j)
                Imgproc.circle(img: img, center: pt, radius: 10, color: Scalar(0), thickness: -1)
            }
        }

        XCTAssert(Calib3d.findCirclesGrid(image: img, patternSize: Size(width: 3, height: 5), centers: centers, flags: Calib3d.CALIB_CB_CLUSTERING | Calib3d.CALIB_CB_ASYMMETRIC_GRID))

        XCTAssertEqual(15, centers.rows())
        XCTAssertEqual(1, centers.cols())
        XCTAssertEqual(CvType.CV_32FC2, centers.type())
    }

    func testFindHomographyListOfPointListOfPoint() throws {
        let NUM:Int32 = 20

        let originalPoints = MatOfPoint2f()
        originalPoints.alloc(NUM)
        let transformedPoints = MatOfPoint2f()
        transformedPoints.alloc(NUM)

        for i:Int32 in 0..<NUM {
            let x:Float = Float.random(in: -50...50)
            let y:Float = Float.random(in: -50...50)
            try originalPoints.put(row:i, col:0, data:[x, y])
            try transformedPoints.put(row:i, col:0, data:[y, x])
        }

        let hmg = Calib3d.findHomography(srcPoints: originalPoints, dstPoints: transformedPoints)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_64F)
        try truth!.put(row:0, col:0, data:[0, 1, 0, 1, 0, 0, 0, 0, 1] as [Double])
        try assertMatEqual(truth!, hmg, OpenCVTestCase.EPS)
    }

    func testReprojectImageTo3DMatMatMat() throws {
        let transformMatrix = Mat(rows: 4, cols: 4, type: CvType.CV_64F)
        try transformMatrix.put(row:0, col:0, data:[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] as [Double])

        let disparity = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)

        var disp = [Float].init(repeating: 0.0, count: Int(OpenCVTestCase.matSize * OpenCVTestCase.matSize))
        for i in 0..<Int(OpenCVTestCase.matSize) {
            for j in 0..<Int(OpenCVTestCase.matSize) {
                disp[i * Int(OpenCVTestCase.matSize) + j] = Float(i - j)
            }
        }
        try disparity.put(row:0, col:0, data:disp)

        let _3dPoints = Mat()

        Calib3d.reprojectImageTo3D(disparity: disparity, _3dImage: _3dPoints, Q: transformMatrix)

        XCTAssertEqual(CvType.CV_32FC3, _3dPoints.type())
        XCTAssertEqual(OpenCVTestCase.matSize, _3dPoints.rows())
        XCTAssertEqual(OpenCVTestCase.matSize, _3dPoints.cols())

        truth = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC3)

        var _truth = [Float](repeating: 0.0, count: Int(OpenCVTestCase.matSize * OpenCVTestCase.matSize * 3))
        for i:Int in 0..<Int(OpenCVTestCase.matSize) {
            for j:Int in 0..<Int(OpenCVTestCase.matSize) {
                let start:Int = (i * Int(OpenCVTestCase.matSize) + j) * 3
                _truth[start + 0] = Float(i)
                _truth[start + 1] = Float(j)
                _truth[start + 2] = Float(i - j)
            }
        }
        try truth!.put(row: 0, col: 0, data: _truth)

        try assertMatEqual(truth!, _3dPoints, OpenCVTestCase.EPS)
    }

    func testReprojectImageTo3DMatMatMatBoolean() throws {
        let transformMatrix = Mat(rows: 4, cols: 4, type: CvType.CV_64F)
        try transformMatrix.put(row: 0, col: 0, data: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] as [Double])

        let disparity = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)

        var disp = [Float](repeating: 0.0, count: Int(OpenCVTestCase.matSize * OpenCVTestCase.matSize))
        for i in 0..<Int(OpenCVTestCase.matSize) {
            for j in 0..<Int(OpenCVTestCase.matSize) {
                disp[i * Int(OpenCVTestCase.matSize) + j] = Float(i - j)
            }
        }
        disp[0] = -.greatestFiniteMagnitude
        try disparity.put(row: 0, col: 0, data: disp)

        let _3dPoints = Mat()

        Calib3d.reprojectImageTo3D(disparity: disparity, _3dImage: _3dPoints, Q: transformMatrix, handleMissingValues: true)

        XCTAssertEqual(CvType.CV_32FC3, _3dPoints.type())
        XCTAssertEqual(OpenCVTestCase.matSize, _3dPoints.rows())
        XCTAssertEqual(OpenCVTestCase.matSize, _3dPoints.cols())

        truth = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC3)

        var _truth = [Float](repeating: 0.0, count:Int(OpenCVTestCase.matSize * OpenCVTestCase.matSize * 3))
        for i in 0..<Int(OpenCVTestCase.matSize) {
            for j in 0..<Int(OpenCVTestCase.matSize) {
                _truth[(i * Int(OpenCVTestCase.matSize) + j) * 3 + 0] = Float(i)
                _truth[(i * Int(OpenCVTestCase.matSize) + j) * 3 + 1] = Float(j)
                _truth[(i * Int(OpenCVTestCase.matSize) + j) * 3 + 2] = Float(i - j)
            }
        }
        _truth[2] = 10000
        try truth!.put(row: 0, col: 0, data: _truth)

        try assertMatEqual(truth!, _3dPoints, OpenCVTestCase.EPS)
    }

    func testReprojectImageTo3DMatMatMatBooleanInt() throws {
        let transformMatrix = Mat(rows: 4, cols: 4, type: CvType.CV_64F)
        try transformMatrix.put(row: 0, col: 0, data: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] as [Double])

        let disparity = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)

        var disp = [Float](repeating: 0.0, count: Int(OpenCVTestCase.matSize * OpenCVTestCase.matSize))
        for i in 0..<Int(OpenCVTestCase.matSize) {
            for j in 0..<Int(OpenCVTestCase.matSize) {
                disp[i * Int(OpenCVTestCase.matSize) + j] = Float(i - j)
            }
        }
        try disparity.put(row:0, col:0, data:disp)

        let _3dPoints = Mat()

        Calib3d.reprojectImageTo3D(disparity: disparity, _3dImage: _3dPoints, Q: transformMatrix, handleMissingValues: false, ddepth: CvType.CV_16S)

        XCTAssertEqual(CvType.CV_16SC3, _3dPoints.type())
        XCTAssertEqual(OpenCVTestCase.matSize, _3dPoints.rows())
        XCTAssertEqual(OpenCVTestCase.matSize, _3dPoints.cols())

        truth = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_16SC3)

        var _truth = [Int16](repeating: 0, count: Int(OpenCVTestCase.matSize * OpenCVTestCase.matSize * 3))
        for i in 0..<Int(OpenCVTestCase.matSize) {
            for j in 0..<Int(OpenCVTestCase.matSize) {
                let start = (i * Int(OpenCVTestCase.matSize) + j) * 3
                _truth[start + 0] = Int16(i)
                _truth[start + 1] = Int16(j)
                _truth[start + 2] = Int16(i - j)
            }
        }
        try truth!.put(row: 0, col: 0, data: _truth)

        try assertMatEqual(truth!, _3dPoints, OpenCVTestCase.EPS)
    }

    func testRodriguesMatMat() throws {
        let r = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        let R = Mat(rows: 3, cols: 3, type: CvType.CV_32F)

        try r.put(row:0, col:0, data:[.pi, 0, 0] as [Float])

        Calib3d.Rodrigues(src: r, dst: R)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row:0, col:0, data:[1, 0, 0, 0, -1, 0, 0, 0, -1] as [Float])
        try assertMatEqual(truth!, R, OpenCVTestCase.EPS)

        let r2 = Mat()
        Calib3d.Rodrigues(src: R, dst: r2)

        try assertMatEqual(r, r2, OpenCVTestCase.EPS)
    }

    func testSolvePnPListOfPoint3ListOfPointMatMatMatMat() throws {
        let intrinsics = Mat.eye(rows: 3, cols: 3, type: CvType.CV_64F)
        try intrinsics.put(row: 0, col: 0, data: [400] as [Double])
        try intrinsics.put(row: 1, col: 1, data: [400] as [Double])
        try intrinsics.put(row: 0, col: 2, data: [640 / 2] as [Double])
        try intrinsics.put(row: 1, col: 2, data: [480 / 2] as [Double])

        let minPnpPointsNum: Int32 = 4

        let points3d = MatOfPoint3f()
        points3d.alloc(minPnpPointsNum)
        let points2d = MatOfPoint2f()
        points2d.alloc(minPnpPointsNum)

        for i in 0..<minPnpPointsNum {
            let x = Float.random(in: -50...50)
            let y = Float.random(in: -50...50)
            try points2d.put(row: i, col: 0, data: [x, y]) //add(Point(x, y))
            try points3d.put(row: i, col: 0, data: [0, y, x]) // add(Point3(0, y, x))
        }

        let rvec = Mat()
        let tvec = Mat()
        Calib3d.solvePnP(objectPoints: points3d, imagePoints: points2d, cameraMatrix: intrinsics, distCoeffs: MatOfDouble(), rvec: rvec, tvec: tvec)

        let truth_rvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_rvec.put(row: 0, col: 0, data: [0, .pi / 2, 0] as [Double])

        let truth_tvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_tvec.put(row: 0, col: 0, data: [-320, -240, 400] as [Double])

        try assertMatEqual(truth_rvec, rvec, OpenCVTestCase.EPS)
        try assertMatEqual(truth_tvec, tvec, OpenCVTestCase.EPS)
    }

    func testComputeCorrespondEpilines() throws {
        let fundamental = Mat(rows: 3, cols: 3, type: CvType.CV_64F)
        try fundamental.put(row: 0, col: 0, data: [0, -0.577, 0.288, 0.577, 0, 0.288, -0.288, -0.288, 0])
        let left = MatOfPoint2f()
        left.alloc(1)
        try left.put(row: 0, col: 0, data: [2, 3] as [Float]) //add(Point(x, y))
        let lines = Mat()
        let truth = Mat(rows: 1, cols: 1, type: CvType.CV_32FC3)
        try truth.put(row: 0, col: 0, data: [-0.70735186, 0.70686162, -0.70588124])
        Calib3d.computeCorrespondEpilines(points: left, whichImage: 1, F: fundamental, lines: lines)
        try assertMatEqual(truth, lines, OpenCVTestCase.EPS)
    }

    func testConstants()
    {
        // calib3d.hpp: some constants have conflict with constants from 'fisheye' namespace
        XCTAssertEqual(1, Calib3d.CALIB_USE_INTRINSIC_GUESS)
        XCTAssertEqual(2, Calib3d.CALIB_FIX_ASPECT_RATIO)
        XCTAssertEqual(4, Calib3d.CALIB_FIX_PRINCIPAL_POINT)
        XCTAssertEqual(8, Calib3d.CALIB_ZERO_TANGENT_DIST)
        XCTAssertEqual(16, Calib3d.CALIB_FIX_FOCAL_LENGTH)
        XCTAssertEqual(32, Calib3d.CALIB_FIX_K1)
        XCTAssertEqual(64, Calib3d.CALIB_FIX_K2)
        XCTAssertEqual(128, Calib3d.CALIB_FIX_K3)
        XCTAssertEqual(0x0800, Calib3d.CALIB_FIX_K4)
        XCTAssertEqual(0x1000, Calib3d.CALIB_FIX_K5)
        XCTAssertEqual(0x2000, Calib3d.CALIB_FIX_K6)
        XCTAssertEqual(0x4000, Calib3d.CALIB_RATIONAL_MODEL)
        XCTAssertEqual(0x8000, Calib3d.CALIB_THIN_PRISM_MODEL)
        XCTAssertEqual(0x10000, Calib3d.CALIB_FIX_S1_S2_S3_S4)
        XCTAssertEqual(0x40000, Calib3d.CALIB_TILTED_MODEL)
        XCTAssertEqual(0x80000, Calib3d.CALIB_FIX_TAUX_TAUY)
        XCTAssertEqual(0x100000, Calib3d.CALIB_USE_QR)
        XCTAssertEqual(0x200000, Calib3d.CALIB_FIX_TANGENT_DIST)
        XCTAssertEqual(0x100, Calib3d.CALIB_FIX_INTRINSIC)
        XCTAssertEqual(0x200, Calib3d.CALIB_SAME_FOCAL_LENGTH)
        XCTAssertEqual(0x400, Calib3d.CALIB_ZERO_DISPARITY)
        XCTAssertEqual((1 << 17), Calib3d.CALIB_USE_LU)
        XCTAssertEqual((1 << 22), Calib3d.CALIB_USE_EXTRINSIC_GUESS)
    }

    func testSolvePnPGeneric_regression_16040() throws {
        let intrinsics = Mat.eye(rows: 3, cols: 3, type: CvType.CV_64F)
        try intrinsics.put(row: 0, col: 0, data: [400] as [Double])
        try intrinsics.put(row: 1, col: 1, data: [400] as [Double])
        try intrinsics.put(row: 0, col: 2, data: [640 / 2] as [Double])
        try intrinsics.put(row: 1, col: 2, data: [480 / 2] as [Double])

        let minPnpPointsNum: Int32 = 4

        let points3d = MatOfPoint3f()
        points3d.alloc(minPnpPointsNum)
        let points2d = MatOfPoint2f()
        points2d.alloc(minPnpPointsNum)

        for i in 0..<minPnpPointsNum {
            let x = Float.random(in: -50...50)
            let y = Float.random(in: -50...50)
            try points2d.put(row: i, col: 0, data: [x, y]) //add(Point(x, y))
            try points3d.put(row: i, col: 0, data: [0, y, x]) // add(Point3(0, y, x))
        }

        var rvecs = [Mat]()
        var tvecs = [Mat]()

        let rvec = Mat()
        let tvec = Mat()

        let reprojectionError = Mat(rows: 2, cols: 1, type: CvType.CV_64FC1)

        Calib3d.solvePnPGeneric(objectPoints: points3d, imagePoints: points2d, cameraMatrix: intrinsics, distCoeffs: MatOfDouble(), rvecs: &rvecs, tvecs: &tvecs, useExtrinsicGuess: false, flags: .SOLVEPNP_IPPE, rvec: rvec, tvec: tvec, reprojectionError: reprojectionError)

        let truth_rvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_rvec.put(row: 0, col: 0, data: [0, .pi / 2, 0] as [Double])

        let truth_tvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_tvec.put(row: 0, col: 0, data: [-320, -240, 400] as [Double])

        try assertMatEqual(truth_rvec, rvecs[0], 10 * OpenCVTestCase.EPS)
        try assertMatEqual(truth_tvec, tvecs[0], 1000 * OpenCVTestCase.EPS)
    }

    func testGetDefaultNewCameraMatrixMat() {
        let mtx = Calib3d.getDefaultNewCameraMatrix(cameraMatrix: gray0)

        XCTAssertFalse(mtx.empty())
        XCTAssertEqual(0, Core.countNonZero(src: mtx))
    }

    func testGetDefaultNewCameraMatrixMatSizeBoolean() {
        let mtx = Calib3d.getDefaultNewCameraMatrix(cameraMatrix: gray0, imgsize: size, centerPrincipalPoint: true)

        XCTAssertFalse(mtx.empty())
        XCTAssertFalse(0 == Core.countNonZero(src: mtx))
        // TODO_: write better test
    }

    func testUndistortMatMatMatMat() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(3))
        let cameraMatrix = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try cameraMatrix.put(row: 0, col: 0, data: [1, 0, 1] as [Float])
        try cameraMatrix.put(row: 1, col: 0, data: [0, 1, 2] as [Float])
        try cameraMatrix.put(row: 2, col: 0, data: [0, 0, 1] as [Float])

        let distCoeffs = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try distCoeffs.put(row: 0, col: 0, data: [1, 3, 2, 4] as [Float])

        Calib3d.undistort(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [0, 0, 0] as [Float])
        try truth!.put(row: 1, col: 0, data: [0, 0, 0] as [Float])
        try truth!.put(row: 2, col: 0, data: [0, 3, 0] as [Float])

        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testUndistortMatMatMatMatMat() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(3))
        let cameraMatrix = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try cameraMatrix.put(row: 0, col: 0, data: [1, 0, 1] as [Float])
        try cameraMatrix.put(row: 1, col: 0, data: [0, 1, 2] as [Float])
        try cameraMatrix.put(row: 2, col: 0, data: [0, 0, 1] as [Float])

        let distCoeffs = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try distCoeffs.put(row: 0, col: 0, data: [2, 1, 4, 5] as [Float])

        let newCameraMatrix = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(1))

        Calib3d.undistort(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs, newCameraMatrix: newCameraMatrix)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(3))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    //undistortPoints(List<Point> src, List<Point> dst, Mat cameraMatrix, Mat distCoeffs)
    func testUndistortPointsListOfPointListOfPointMatMat() {
        let src = MatOfPoint2f(array: [Point2f(x: 1, y: 2), Point2f(x: 3, y: 4), Point2f(x: -1, y: -1)])
        let dst = MatOfPoint2f()
        let cameraMatrix = Mat.eye(rows: 3, cols: 3, type: CvType.CV_64FC1)
        let distCoeffs = Mat(rows: 8, cols: 1, type: CvType.CV_64FC1, scalar: Scalar(0))

        Calib3d.undistortPoints(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs)

        XCTAssertEqual(src.toArray(), dst.toArray())
    }
}
