//
//  Calib3dTest.swift
//
//  Created by Giles Payne on 2020/05/26.
//

import XCTest
import OpenCV

class Cv3dTest: OpenCVTestCase {

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

        Cv3d.composeRT(rvec1: rvec1, tvec1: tvec1, rvec2: rvec2, tvec2: tvec2, rvec3: rvec3, tvec3: tvec3)

        try assertMatEqual(outRvec, rvec3, OpenCVTestCase.EPS)
        try assertMatEqual(outTvec, tvec3, OpenCVTestCase.EPS)
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

        let hmg = Cv3d.findHomography(srcPoints: originalPoints, dstPoints: transformedPoints)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_64F)
        try truth!.put(row:0, col:0, data:[0, 1, 0, 1, 0, 0, 0, 0, 1] as [Double])
        try assertMatEqual(truth!, hmg, OpenCVTestCase.EPS)
    }

    func testRodriguesMatMat() throws {
        let r = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        let R = Mat(rows: 3, cols: 3, type: CvType.CV_32F)

        try r.put(row:0, col:0, data:[.pi, 0, 0] as [Float])

        Cv3d.Rodrigues(src: r, dst: R)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row:0, col:0, data:[1, 0, 0, 0, -1, 0, 0, 0, -1] as [Float])
        try assertMatEqual(truth!, R, OpenCVTestCase.EPS)

        let r2 = Mat()
        Cv3d.Rodrigues(src: R, dst: r2)

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
        Cv3d.solvePnP(objectPoints: points3d, imagePoints: points2d, cameraMatrix: intrinsics, distCoeffs: MatOfDouble(), rvec: rvec, tvec: tvec)

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
        Cv3d.computeCorrespondEpilines(points: left, whichImage: 1, F: fundamental, lines: lines)
        try assertMatEqual(truth, lines, OpenCVTestCase.EPS)
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

        Cv3d.solvePnPGeneric(objectPoints: points3d, imagePoints: points2d, cameraMatrix: intrinsics, distCoeffs: MatOfDouble(), rvecs: &rvecs, tvecs: &tvecs, useExtrinsicGuess: false, flags: .SOLVEPNP_IPPE, rvec: rvec, tvec: tvec, reprojectionError: reprojectionError)

        let truth_rvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_rvec.put(row: 0, col: 0, data: [0, .pi / 2, 0] as [Double])

        let truth_tvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_tvec.put(row: 0, col: 0, data: [-320, -240, 400] as [Double])

        try assertMatEqual(truth_rvec, rvecs[0], 10 * OpenCVTestCase.EPS)
        try assertMatEqual(truth_tvec, tvecs[0], 1000 * OpenCVTestCase.EPS)
    }

    func testGetDefaultNewCameraMatrixMat() {
        let mtx = Cv3d.getDefaultNewCameraMatrix(cameraMatrix: gray0)

        XCTAssertFalse(mtx.empty())
        XCTAssertEqual(0, Core.countNonZero(src: mtx))
    }

    func testGetDefaultNewCameraMatrixMatSizeBoolean() {
        let mtx = Cv3d.getDefaultNewCameraMatrix(cameraMatrix: gray0, imgsize: size, centerPrincipalPoint: true)

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

        Cv3d.undistort(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs)

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

        Cv3d.undistort(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs, newCameraMatrix: newCameraMatrix)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(3))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    //undistortPoints(List<Point> src, List<Point> dst, Mat cameraMatrix, Mat distCoeffs)
    func testUndistortPointsListOfPointListOfPointMatMat() {
        let src = MatOfPoint2f(array: [Point2f(x: 1, y: 2), Point2f(x: 3, y: 4), Point2f(x: -1, y: -1)])
        let dst = MatOfPoint2f()
        let cameraMatrix = Mat.eye(rows: 3, cols: 3, type: CvType.CV_64FC1)
        let distCoeffs = Mat(rows: 8, cols: 1, type: CvType.CV_64FC1, scalar: Scalar(0))

        Cv3d.undistortPoints(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs)

        XCTAssertEqual(src.toArray(), dst.toArray())
    }
}
