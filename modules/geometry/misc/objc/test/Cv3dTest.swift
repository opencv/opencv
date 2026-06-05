//
//  Calib3dTest.swift
//
//  Created by Giles Payne on 2020/05/26.
//

import XCTest
import OpenCV

class GeometryTest: OpenCVTestCase {

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

        Geometry.composeRT(rvec1: rvec1, tvec1: tvec1, rvec2: rvec2, tvec2: tvec2, rvec3: rvec3, tvec3: tvec3)

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

        let hmg = Geometry.findHomography(srcPoints: originalPoints, dstPoints: transformedPoints)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_64F)
        try truth!.put(row:0, col:0, data:[0, 1, 0, 1, 0, 0, 0, 0, 1] as [Double])
        try assertMatEqual(truth!, hmg, OpenCVTestCase.EPS)
    }

    func testRodriguesMatMat() throws {
        let r = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        let R = Mat(rows: 3, cols: 3, type: CvType.CV_32F)

        try r.put(row:0, col:0, data:[.pi, 0, 0] as [Float])

        Geometry.Rodrigues(src: r, dst: R)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row:0, col:0, data:[1, 0, 0, 0, -1, 0, 0, 0, -1] as [Float])
        try assertMatEqual(truth!, R, OpenCVTestCase.EPS)

        let r2 = Mat()
        Geometry.Rodrigues(src: R, dst: r2)

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
        Geometry.solvePnP(objectPoints: points3d, imagePoints: points2d, cameraMatrix: intrinsics, distCoeffs: MatOfDouble(), rvec: rvec, tvec: tvec)

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
        Geometry.computeCorrespondEpilines(points: left, whichImage: 1, F: fundamental, lines: lines)
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

        Geometry.solvePnPGeneric(objectPoints: points3d, imagePoints: points2d, cameraMatrix: intrinsics, distCoeffs: MatOfDouble(), rvecs: &rvecs, tvecs: &tvecs, useExtrinsicGuess: false, flags: .SOLVEPNP_IPPE, rvec: rvec, tvec: tvec, reprojectionError: reprojectionError)

        let truth_rvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_rvec.put(row: 0, col: 0, data: [0, .pi / 2, 0] as [Double])

        let truth_tvec = Mat(rows: 3, cols: 1, type: CvType.CV_64F)
        try truth_tvec.put(row: 0, col: 0, data: [-320, -240, 400] as [Double])

        try assertMatEqual(truth_rvec, rvecs[0], 10 * OpenCVTestCase.EPS)
        try assertMatEqual(truth_tvec, tvecs[0], 1000 * OpenCVTestCase.EPS)
    }

    func testGetDefaultNewCameraMatrixMat() {
        let mtx = Geometry.getDefaultNewCameraMatrix(cameraMatrix: gray0)

        XCTAssertFalse(mtx.empty())
        XCTAssertEqual(0, Core.countNonZero(src: mtx))
    }

    func testGetDefaultNewCameraMatrixMatSizeBoolean() {
        let mtx = Geometry.getDefaultNewCameraMatrix(cameraMatrix: gray0, imgsize: size, centerPrincipalPoint: true)

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

        Geometry.undistort(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs)

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

        Geometry.undistort(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs, newCameraMatrix: newCameraMatrix)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(3))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    //undistortPoints(List<Point> src, List<Point> dst, Mat cameraMatrix, Mat distCoeffs)
    func testUndistortPointsListOfPointListOfPointMatMat() {
        let src = MatOfPoint2f(array: [Point2f(x: 1, y: 2), Point2f(x: 3, y: 4), Point2f(x: -1, y: -1)])
        let dst = MatOfPoint2f()
        let cameraMatrix = Mat.eye(rows: 3, cols: 3, type: CvType.CV_64FC1)
        let distCoeffs = Mat(rows: 8, cols: 1, type: CvType.CV_64FC1, scalar: Scalar(0))

        Geometry.undistortPoints(src: src, dst: dst, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs)

        XCTAssertEqual(src.toArray(), dst.toArray())
    }

    func testApproxPolyDP() {
        let curve = [Point2f(x: 1, y: 3), Point2f(x: 2, y: 4), Point2f(x: 3, y: 5), Point2f(x: 4, y: 4), Point2f(x: 5, y: 3)]

        var approxCurve = [Point2f]()

        Geometry.approxPolyDP(curve: curve, approxCurve: &approxCurve, epsilon: OpenCVTestCase.EPS, closed: true)

        let approxCurveGold = [Point2f(x: 1, y: 3), Point2f(x: 3, y: 5), Point2f(x: 5, y: 3)]

        XCTAssert(approxCurve == approxCurveGold)
    }

    func testArcLength() {
        let curve = [Point2f(x: 1, y: 3), Point2f(x: 2, y: 4), Point2f(x: 3, y: 5), Point2f(x: 4, y: 4), Point2f(x: 5, y: 3)]

        let arcLength = Geometry.arcLength(curve: curve, closed: false)

        XCTAssertEqual(5.656854249, arcLength, accuracy:0.000001)
    }

    func testContourAreaMat() throws {
        let contour = Mat(rows: 1, cols: 4, type: CvType.CV_32FC2)
        try contour.put(row: 0, col: 0, data: [0, 0, 10, 0, 10, 10, 5, 4] as [Float])

        let area = Geometry.contourArea(contour: contour)

        XCTAssertEqual(45.0, area, accuracy: OpenCVTestCase.EPS)
    }

    func testContourAreaMatBoolean() throws {
        let contour = Mat(rows: 1, cols: 4, type: CvType.CV_32FC2)
        try contour.put(row: 0, col: 0, data: [0, 0, 10, 0, 10, 10, 5, 4] as [Float])

        let area = Geometry.contourArea(contour: contour, oriented: true)

        XCTAssertEqual(45.0, area, accuracy: OpenCVTestCase.EPS)
    }

    func testConvexHullMatMatBooleanBoolean() {
        let points = [Point(x: 2, y: 0),
                      Point(x: 4, y: 0),
                      Point(x: 3, y: 2),
                      Point(x: 0, y: 2),
                      Point(x: 2, y: 1),
                      Point(x: 3, y: 1)]

        var hull = [Int32]()

        Geometry.convexHull(points: points, hull: &hull, clockwise: true)

        XCTAssert([3, 2, 1, 0] == hull)
    }

    func testConvexityDefects() throws {
        let points = [Point(x: 20, y: 0),
                      Point(x: 40, y: 0),
                      Point(x: 30, y: 20),
                      Point(x: 0,  y: 20),
                      Point(x: 20, y: 10),
                      Point(x: 30, y: 10)]

        var hull = [Int32]()
        Geometry.convexHull(points: points, hull: &hull)

        var convexityDefects = [Int4]()
        Geometry.convexityDefects(contour: points, convexhull: hull, convexityDefects: &convexityDefects)

        XCTAssertTrue(Int4(v0: 3, v1: 0, v2: 5, v3: 3620) == convexityDefects[0])
    }

    func testGetAffineTransform() throws {
        let src = [Point2f(x: 2, y: 3), Point2f(x: 3, y: 1), Point2f(x: 1, y: 4)]
        let dst = [Point2f(x: 3, y: 3), Point2f(x: 7, y: 4), Point2f(x: 5, y: 6)]

        let transform = Geometry.getAffineTransform(src: src, dst: dst)

        let truth = Mat(rows: 2, cols: 3, type: CvType.CV_64FC1)

        try truth.put(row: 0, col: 0, data: [-8.0, -6.0, 37.0])
        try truth.put(row: 1, col: 0, data: [-7.0, -4.0, 29.0])
        try assertMatEqual(truth, transform, OpenCVTestCase.EPS)
    }

    func testGetRotationMatrix2D() throws {
        let center = Point2f(x: 0, y: 0)

        dst = Geometry.getRotationMatrix2D(center: center, angle: 0, scale: 1)

        truth = Mat(rows: 2, cols: 3, type: CvType.CV_64F)
        try truth!.put(row: 0, col: 0, data: [1.0, 0.0, 0.0])
        try truth!.put(row: 1, col: 0, data: [0.0, 1.0, 0.0])

        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testInvertAffineTransform() throws {
        let src = Mat(rows: 2, cols: 3, type: CvType.CV_64F, scalar: Scalar(1))

        Geometry.invertAffineTransform(M: src, iM: dst)

        truth = Mat(rows: 2, cols: 3, type: CvType.CV_64F, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testIsContourConvex() {
        let contour1 = [Point(x: 0, y: 0), Point(x: 10, y: 0), Point(x: 10, y: 10), Point(x: 5, y: 4)]

        XCTAssertFalse(Geometry.isContourConvex(contour: contour1))

        let contour2 = [Point(x: 0, y: 0), Point(x: 10, y: 0), Point(x: 10, y: 10), Point(x: 5, y: 6)]

        XCTAssert(Geometry.isContourConvex(contour: contour2))
    }

    func testMinAreaRect() {
        let points = [Point2f(x: 1, y: 1), Point2f(x: 5, y: 1), Point2f(x: 4, y: 3), Point2f(x: 6, y: 2)]

        let rrect = Geometry.minAreaRect(points: points)

        XCTAssertEqual(Size2f(width: 5, height: 2), rrect.size)
        XCTAssertEqual(0.0, rrect.angle)
        XCTAssertEqual(Point2f(x: 3.5, y: 2), rrect.center)
    }

    func testMinEnclosingCircle() {
        let points = [Point2f(x: 0, y: 0), Point2f(x: -100, y: 0), Point2f(x: 0, y: -100), Point2f(x: 100, y: 0), Point2f(x: 0, y: 100)]
        let actualCenter = Point2f()
        var radius:Float = 0

        Geometry.minEnclosingCircle(points: points, center: actualCenter, radius: &radius)

        XCTAssertEqual(Point2f(x: 0, y: 0), actualCenter)
        XCTAssertEqual(100.0, radius, accuracy: 1.0)
    }

    func testPointPolygonTest() {
        let contour = [Point2f(x: 0, y: 0), Point2f(x: 1, y: 3), Point2f(x: 3, y: 4), Point2f(x: 4, y: 3), Point2f(x: 2, y: 1)]
        let sign1 = Geometry.pointPolygonTest(contour: contour, pt: Point2f(x: 2, y: 2), measureDist: false)
        XCTAssertEqual(1.0, sign1)

        let sign2 = Geometry.pointPolygonTest(contour: contour, pt: Point2f(x: 4, y: 4), measureDist: true)
        XCTAssertEqual(-sqrt(0.5), sign2)
    }


}
