//
//  ImgprocTest.swift
//
//  Created by Giles Payne on 2020/02/08.
//

import XCTest
import OpenCV

class ImgprocTest: OpenCVTestCase {

    let anchorPoint = Point(x: 2, y: 2)
    let imgprocSz: Int32 = 2
    let size = Size(width: 3, height: 3)

    func testAccumulateMatMat() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let dst = getMat(CvType.CV_64F, vals: [0])
        let dst2 = src.clone()

        Imgproc.accumulate(src: src, dst: dst)
        Imgproc.accumulate(src: src, dst: dst2)

        try assertMatEqual(src, dst, OpenCVTestCase.EPS)
        try assertMatEqual(getMat(CvType.CV_64F, vals: [4]), dst2, OpenCVTestCase.EPS)
    }

    func testAccumulateMatMatMat() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let mask = makeMask(getMat(CvType.CV_8U, vals: [1]))
        let dst = getMat(CvType.CV_64F, vals: [0])
        let dst2 = src.clone()

        Imgproc.accumulate(src: src, dst: dst, mask: mask)
        Imgproc.accumulate(src: src, dst: dst2, mask: mask)

        try assertMatEqual(makeMask(getMat(CvType.CV_64F, vals: [2])), dst, OpenCVTestCase.EPS)
        try assertMatEqual(makeMask(getMat(CvType.CV_64F, vals: [4]), vals: [2]), dst2, OpenCVTestCase.EPS)
    }

    func testAccumulateProductMatMatMat() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let dst = getMat(CvType.CV_64F, vals: [0])
        let dst2 = src.clone()

        Imgproc.accumulateProduct(src1: src, src2: src, dst: dst)
        Imgproc.accumulateProduct(src1: src, src2: dst, dst: dst2)

        try assertMatEqual(getMat(CvType.CV_64F, vals:[4]), dst, OpenCVTestCase.EPS)
        try assertMatEqual(getMat(CvType.CV_64F, vals:[10]), dst2, OpenCVTestCase.EPS)
    }

    func testAccumulateProductMatMatMatMat() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let mask = makeMask(getMat(CvType.CV_8U, vals: [1]))
        let dst = getMat(CvType.CV_64F, vals: [0])
        let dst2 = src.clone()

        Imgproc.accumulateProduct(src1: src, src2: src, dst: dst, mask: mask)
        Imgproc.accumulateProduct(src1: src, src2: dst, dst: dst2, mask: mask)

        try assertMatEqual(makeMask(getMat(CvType.CV_64F, vals: [4])), dst, OpenCVTestCase.EPS)
        try assertMatEqual(makeMask(getMat(CvType.CV_64F, vals: [10]), vals:[2]), dst2, OpenCVTestCase.EPS)
    }

    func testAccumulateSquareMatMat() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let dst = getMat(CvType.CV_64F, vals: [0])
        let dst2 = src.clone()

        Imgproc.accumulateSquare(src: src, dst: dst)
        Imgproc.accumulateSquare(src: src, dst: dst2)

        try assertMatEqual(getMat(CvType.CV_64F, vals: [4]), dst, OpenCVTestCase.EPS)
        try assertMatEqual(getMat(CvType.CV_64F, vals: [6]), dst2, OpenCVTestCase.EPS)
    }

    func testAccumulateSquareMatMatMat() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let mask = makeMask(getMat(CvType.CV_8U, vals: [1]))
        let dst = getMat(CvType.CV_64F, vals: [0])
        let dst2 = src.clone()

        Imgproc.accumulateSquare(src: src, dst: dst, mask: mask)
        Imgproc.accumulateSquare(src: src, dst: dst2, mask: mask)

        try assertMatEqual(makeMask(getMat(CvType.CV_64F, vals: [4])), dst, OpenCVTestCase.EPS)
        try assertMatEqual(makeMask(getMat(CvType.CV_64F, vals: [6]), vals: [2]), dst2, OpenCVTestCase.EPS)
    }

    func testAccumulateWeightedMatMatDouble() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let dst = getMat(CvType.CV_64F, vals: [4])
        let dst2 = src.clone()

        Imgproc.accumulateWeighted(src: src, dst: dst, alpha: 0.5)
        Imgproc.accumulateWeighted(src: src, dst: dst2, alpha: 2)

        try assertMatEqual(getMat(CvType.CV_64F, vals: [3]), dst, OpenCVTestCase.EPS)
        try assertMatEqual(getMat(CvType.CV_64F, vals: [2]), dst2, OpenCVTestCase.EPS)
    }

    func testAccumulateWeightedMatMatDoubleMat() throws {
        let src = getMat(CvType.CV_64F, vals: [2])
        let mask = makeMask(getMat(CvType.CV_8U, vals: [1]))
        let dst = getMat(CvType.CV_64F, vals: [4])
        let dst2 = src.clone()

        Imgproc.accumulateWeighted(src: src, dst: dst, alpha: 0.5, mask: mask)
        Imgproc.accumulateWeighted(src: src, dst: dst2, alpha: 2, mask: mask)

        try assertMatEqual(makeMask(getMat(CvType.CV_64F, vals: [3]), vals: [4]), dst, OpenCVTestCase.EPS)
        try assertMatEqual(getMat(CvType.CV_64F, vals: [2]), dst2, OpenCVTestCase.EPS)
    }

    func testAdaptiveThreshold() {
        let src = makeMask(getMat(CvType.CV_8U, vals: [50]), vals:[20])
        let dst = Mat()

        Imgproc.adaptiveThreshold(src: src, dst: dst, maxValue: 1, adaptiveMethod: .ADAPTIVE_THRESH_MEAN_C, thresholdType: .THRESH_BINARY, blockSize: 3, C: 0)

        XCTAssertEqual(src.rows(), Core.countNonZero(src: dst))
    }

    func testApproxPolyDP() {
        let curve = [Point2f(x: 1, y: 3), Point2f(x: 2, y: 4), Point2f(x: 3, y: 5), Point2f(x: 4, y: 4), Point2f(x: 5, y: 3)]

        let approxCurve = NSMutableArray()

        Imgproc.approxPolyDP(curve: curve, approxCurve: approxCurve, epsilon: OpenCVTestCase.EPS, closed: true)

        let approxCurveGold = [Point2f(x: 1, y: 3), Point2f(x: 3, y: 5), Point2f(x: 5, y: 3)]

        XCTAssert(approxCurve as! [Point2f] == approxCurveGold)
    }

    func testArcLength() {
        let curve = [Point2f(x: 1, y: 3), Point2f(x: 2, y: 4), Point2f(x: 3, y: 5), Point2f(x: 4, y: 4), Point2f(x: 5, y: 3)]

        let arcLength = Imgproc.arcLength(curve: curve, closed: false)

        XCTAssertEqual(5.656854249, arcLength, accuracy:0.000001)
    }

    func testBilateralFilterMatMatIntDoubleDouble() throws {
        Imgproc.bilateralFilter(src: gray255, dst: dst, d: 5, sigmaColor: 10, sigmaSpace: 5)

        try assertMatEqual(gray255, dst)
    }

    func testBilateralFilterMatMatIntDoubleDoubleInt() throws {
        Imgproc.bilateralFilter(src: gray255, dst: dst, d: 5, sigmaColor: 10, sigmaSpace: 5, borderType: .BORDER_REFLECT)

        try assertMatEqual(gray255, dst)
    }

    func testBlurMatMatSize() throws {
        Imgproc.blur(src: gray0, dst: dst, ksize: size)
        try assertMatEqual(gray0, dst)

        Imgproc.blur(src: gray255, dst: dst, ksize: size)
        try assertMatEqual(gray255, dst)
    }

    func testBlurMatMatSizePoint() throws {
        Imgproc.blur(src: gray0, dst: dst, ksize: size, anchor: anchorPoint)
        try assertMatEqual(gray0, dst)
    }

    func testBlurMatMatSizePointInt() throws {
        Imgproc.blur(src: gray0, dst: dst, ksize: size, anchor: anchorPoint, borderType: .BORDER_REFLECT)
        try assertMatEqual(gray0, dst)
    }

    func testBoundingRect() {
        let points = [Point(x: 0, y: 0), Point(x: 0, y: 4), Point(x: 4, y: 0), Point(x: 4, y: 4)]
        let p1 = Point(x: 1, y: 1)
        let p2 = Point(x: -5, y: -2)

        let bbox = Imgproc.boundingRect(array: MatOfPoint(array: points))

        XCTAssert(bbox.contains(p1))
        XCTAssertFalse(bbox.contains(p2))
    }

    func testBoxFilterMatMatIntSize() throws {
        let size = Size(width: 3, height: 3)
        Imgproc.boxFilter(src: gray0, dst: dst, ddepth: 8, ksize: size)
        try assertMatEqual(gray0, dst)
    }

    func testBoxFilterMatMatIntSizePointBoolean() throws {
        Imgproc.boxFilter(src: gray255, dst: dst, ddepth: 8, ksize: size, anchor: anchorPoint, normalize: false)
        try assertMatEqual(gray255, dst)
    }

    func testBoxFilterMatMatIntSizePointBooleanInt() throws {
        Imgproc.boxFilter(src: gray255, dst: dst, ddepth: 8, ksize: size, anchor: anchorPoint, normalize: false, borderType: .BORDER_REFLECT)
        try assertMatEqual(gray255, dst)
    }

    func testCalcBackProject() {
        let images = [grayChess]
        let channels = IntVector([0])
        let histSize = IntVector([10])
        let ranges = FloatVector([0, 256])

        let hist = Mat()
        Imgproc.calcHist(images: images, channels: channels, mask: Mat(), hist: hist, histSize: histSize, ranges: ranges)
        Core.normalize(src: hist, dst: hist)

        Imgproc.calcBackProject(images: images, channels: channels, hist: hist, dst: dst, ranges: ranges, scale: 255)

        XCTAssertEqual(grayChess.size(), dst.size())
        XCTAssertEqual(grayChess.depth(), dst.depth())
        XCTAssertFalse(0 == Core.countNonZero(src: dst))
    }

    func testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat() throws {
        let images = [gray128]
        let channels = IntVector([0])
        let histSize = IntVector([10])
        let ranges = FloatVector([0, 256])
        let hist = Mat()

        Imgproc.calcHist(images: images, channels: channels, mask: Mat(), hist: hist, histSize: histSize, ranges: ranges)

        truth = Mat(rows: 10, cols: 1, type: CvType.CV_32F, scalar: Scalar.all(0))
        try truth!.put(row: 5, col: 0, data: [100] as [Float])
        try assertMatEqual(truth!, hist, OpenCVTestCase.EPS)
    }

    func testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat2D() throws {
        let images = [gray255, gray128]
        let channels = IntVector([0, 1])
        let histSize = IntVector([10, 10])
        let ranges = FloatVector([0, 256, 0, 256])
        let hist = Mat()

        Imgproc.calcHist(images: images, channels: channels, mask: Mat(), hist: hist, histSize: histSize, ranges: ranges)

        truth = Mat(rows: 10, cols: 10, type: CvType.CV_32F, scalar: Scalar.all(0))
        try truth!.put(row: 9, col: 5, data: [100] as [Float])
        try assertMatEqual(truth!, hist, OpenCVTestCase.EPS)
    }

    func testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat3D() throws {
        let images = [rgbLena]

        let hist3D = Mat()
        let histList = [Mat(), Mat(), Mat()]

        let histSize = IntVector([10])
        let ranges = FloatVector([0, 256])

        for i:Int in 0..<Int(rgbLena.channels()) {
            Imgproc.calcHist(images: images, channels: IntVector([Int32(i)]), mask: Mat(), hist: histList[i], histSize: histSize, ranges: ranges)

            XCTAssertEqual(10, histList[i].checkVector(elemChannels: 1))
        }

        Core.merge(mv: histList, dst: hist3D)

        XCTAssertEqual(CvType.CV_32FC3, hist3D.type())
        XCTAssertEqual(10, hist3D.checkVector(elemChannels: 3))

        let truth = Mat(rows: 10, cols: 1, type: CvType.CV_32FC3)
        try truth.put(row: 0, col: 0,
                  data: [0, 24870, 0,
                 1863, 31926, 1,
                 56682, 37677, 2260,
                 77278, 44751, 32436,
                 69397, 41343, 18526,
                 27180, 40407, 18658,
                 21101, 15993, 32042,
                 8343, 18585, 47786,
                 300, 6567, 80988,
                 0, 25, 29447] as [Float])

        try assertMatEqual(truth, hist3D, OpenCVTestCase.EPS)
    }

    func testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloatBoolean() throws {
        let images = [gray255, gray128]
        let channels = IntVector([0, 1])
        let histSize = IntVector([10, 10])
        let ranges = FloatVector([0, 256, 0, 256])
        let hist = Mat()

        Imgproc.calcHist(images: images, channels: channels, mask: Mat(), hist: hist, histSize: histSize, ranges: ranges, accumulate: true)

        truth = Mat(rows: 10, cols: 10, type: CvType.CV_32F, scalar: Scalar.all(0))
        try truth!.put(row: 9, col: 5, data: [100] as [Float])
        try assertMatEqual(truth!, hist, OpenCVTestCase.EPS)
    }

    func testCannyMatMatDoubleDouble() throws {
        Imgproc.Canny(image: gray255, edges: dst, threshold1: 5, threshold2: 10)
        try assertMatEqual(gray0, dst)
    }

    func testCannyMatMatDoubleDoubleIntBoolean() throws {
        Imgproc.Canny(image: gray0, edges: dst, threshold1: 5, threshold2: 10, apertureSize: 5, L2gradient: true)
        try assertMatEqual(gray0, dst)
    }

    func testCompareHist() throws {
        let H1 = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        let H2 = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try H1.put(row: 0, col: 0, data: [1, 2, 3] as [Float])
        try H2.put(row: 0, col: 0, data: [4, 5, 6] as [Float])

        let distance = Imgproc.compareHist(H1: H1, H2: H2, method: .HISTCMP_CORREL)

        XCTAssertEqual(1.0, distance, accuracy: OpenCVTestCase.EPS)
    }

    func testContourAreaMat() throws {
        let contour = Mat(rows: 1, cols: 4, type: CvType.CV_32FC2)
        try contour.put(row: 0, col: 0, data: [0, 0, 10, 0, 10, 10, 5, 4] as [Float])

        let area = Imgproc.contourArea(contour: contour)

        XCTAssertEqual(45.0, area, accuracy: OpenCVTestCase.EPS)
    }

    func testContourAreaMatBoolean() throws {
        let contour = Mat(rows: 1, cols: 4, type: CvType.CV_32FC2)
        try contour.put(row: 0, col: 0, data: [0, 0, 10, 0, 10, 10, 5, 4] as [Float])

        let area = Imgproc.contourArea(contour: contour, oriented: true)

        XCTAssertEqual(45.0, area, accuracy: OpenCVTestCase.EPS)
    }

    func testConvertMapsMatMatMatMatInt() throws {
        let map1 = Mat(rows: 1, cols: 4, type: CvType.CV_32FC1, scalar: Scalar(1))
        let map2 = Mat(rows: 1, cols: 4, type: CvType.CV_32FC1, scalar: Scalar(2))
        let dstmap1 = Mat(rows: 1, cols: 4, type: CvType.CV_16SC2)
        let dstmap2 = Mat(rows: 1, cols: 4, type: CvType.CV_16UC1)

        Imgproc.convertMaps(map1: map1, map2: map2, dstmap1: dstmap1, dstmap2: dstmap2, dstmap1type: CvType.CV_16SC2)

        let truthMap1 = Mat(rows: 1, cols: 4, type: CvType.CV_16SC2)
        try truthMap1.put(row: 0, col: 0, data: [1, 2, 1, 2, 1, 2, 1, 2] as [Int16])
        try assertMatEqual(truthMap1, dstmap1)
        let truthMap2 = Mat(rows: 1, cols: 4, type: CvType.CV_16UC1, scalar: Scalar(0))
        try assertMatEqual(truthMap2, dstmap2)
    }

    func testConvertMapsMatMatMatMatIntBoolean() throws {
        let map1 = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1, scalar: Scalar(2))
        let map2 = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1, scalar: Scalar(4))
        let dstmap1 = Mat(rows: 1, cols: 3, type: CvType.CV_16SC2)
        let dstmap2 = Mat(rows: 1, cols: 3, type: CvType.CV_16UC1)

        Imgproc.convertMaps(map1: map1, map2: map2, dstmap1: dstmap1, dstmap2: dstmap2, dstmap1type: CvType.CV_16SC2, nninterpolation: false)
        // TODO_: write better test (last param == true)

        let truthMap1 = Mat(rows: 1, cols: 3, type: CvType.CV_16SC2)
        try truthMap1.put(row: 0, col: 0, data: [2, 4, 2, 4, 2, 4] as [Int16])
        try assertMatEqual(truthMap1, dstmap1)
        let truthMap2 = Mat(rows: 1, cols: 3, type: CvType.CV_16UC1, scalar: Scalar(0))
        try assertMatEqual(truthMap2, dstmap2)
    }

    func testConvexHullMatMat() {
        let points = [Point(x: 20, y: 0),
                    Point(x: 40, y: 0),
                    Point(x: 30, y: 20),
                    Point(x: 0,  y: 20),
                    Point(x: 20, y: 10),
                    Point(x: 30, y: 10)]

        let hull = IntVector()

        Imgproc.convexHull(points: points, hull: hull)

        let expHull = IntVector([0, 1, 2, 3])
        XCTAssert(expHull.array == hull.array)
    }

    func testConvexHullMatMatBooleanBoolean() {
        let points = [Point(x: 2, y: 0),
                      Point(x: 4, y: 0),
                      Point(x: 3, y: 2),
                      Point(x: 0, y: 2),
                      Point(x: 2, y: 1),
                      Point(x: 3, y: 1)]

        let hull = IntVector()

        Imgproc.convexHull(points: points, hull: hull, clockwise: true)

        let expHull = IntVector([3, 2, 1, 0])
        XCTAssert(expHull.array == hull.array)
    }

    func testConvexityDefects() throws {
        let points = [Point(x: 20, y: 0),
                      Point(x: 40, y: 0),
                      Point(x: 30, y: 20),
                      Point(x: 0,  y: 20),
                      Point(x: 20, y: 10),
                      Point(x: 30, y: 10)]

        let hull = IntVector()
        Imgproc.convexHull(points: points, hull: hull)

        let convexityDefects = NSMutableArray()
        Imgproc.convexityDefects(contour: points, convexhull: hull, convexityDefects: convexityDefects)

        XCTAssertTrue(Int4(v0: 3, v1: 0, v2: 5, v3: 3620) == (convexityDefects[0] as! Int4))
    }

    func testCornerEigenValsAndVecsMatMatIntInt() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32FC1)
        try src.put(row: 0, col: 0, data: [1, 2] as [Float])
        try src.put(row: 1, col: 0, data: [4, 2] as [Float])

        let blockSize:Int32 = 3
        let ksize:Int32 = 5

        // TODO: eigen vals and vectors returned = 0 for most src matrices
        Imgproc.cornerEigenValsAndVecs(src: src, dst: dst, blockSize: blockSize, ksize: ksize)
        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32FC(6), scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCornerEigenValsAndVecsMatMatIntIntInt() throws {
        let src = Mat(rows: 4, cols: 4, type: CvType.CV_32FC1, scalar: Scalar(128))

        let blockSize:Int32 = 3
        let ksize:Int32 = 5

        truth = Mat(rows: 4, cols: 4, type: CvType.CV_32FC(6), scalar: Scalar(0))

        Imgproc.cornerEigenValsAndVecs(src: src, dst: dst, blockSize: blockSize, ksize: ksize, borderType: .BORDER_REFLECT)
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCornerHarrisMatMatIntIntDouble() throws {
        truth = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC1, scalar: Scalar(0))
        let blockSize:Int32 = 5
        let ksize:Int32 = 7
        let k = 0.1
        Imgproc.cornerHarris(src: gray128, dst: dst, blockSize: blockSize, ksize: ksize, k: k)
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCornerHarrisMatMatIntIntDoubleInt() throws {
        truth = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC1, scalar: Scalar(0))
        let blockSize:Int32 = 5
        let ksize:Int32 = 7
        let k = 0.1
        Imgproc.cornerHarris(src: gray255, dst: dst, blockSize: blockSize, ksize: ksize, k: k, borderType: .BORDER_REFLECT)
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCornerMinEigenValMatMatInt() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32FC1)
        try src.put(row: 0, col: 0, data: [1, 2] as [Float])
        try src.put(row: 1, col: 0, data: [2, 1] as [Float])
        let blockSize:Int32 = 5

        Imgproc.cornerMinEigenVal(src: src, dst: dst, blockSize: blockSize)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32FC1, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)

        Imgproc.cornerMinEigenVal(src: gray255, dst: dst, blockSize: blockSize)

        truth = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC1, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCornerMinEigenValMatMatIntInt() throws {
        let src = Mat.eye(rows: 3, cols: 3, type: CvType.CV_32FC1)
        let blockSize:Int32 = 3
        let ksize:Int32 = 5

        Imgproc.cornerMinEigenVal(src: src, dst: dst, blockSize: blockSize, ksize: ksize)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32FC1)
        try truth!.put(row: 0, col: 0, data: [1.0 / 18, 1.0 / 36, 1.0 / 18] as [Float])
        try truth!.put(row: 1, col: 0, data: [1.0 / 36, 1.0 / 18, 1.0 / 36] as [Float])
        try truth!.put(row: 2, col: 0, data: [1.0 / 18, 1.0 / 36, 1.0 / 18] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCornerMinEigenValMatMatIntIntInt() throws {
        let src = Mat.eye(rows: 3, cols: 3, type: CvType.CV_32FC1)
        let blockSize:Int32 = 3
        let ksize:Int32 = 5

        Imgproc.cornerMinEigenVal(src: src, dst: dst, blockSize: blockSize, ksize: ksize, borderType: .BORDER_REFLECT)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32FC1)
        try truth!.put(row: 0, col: 0, data: [0.68055558, 0.92708349, 0.5868057])
        try truth!.put(row: 1, col: 0, data: [0.92708343, 0.92708343, 0.92708343])
        try truth!.put(row: 2, col: 0, data: [0.58680564, 0.92708343, 0.68055564])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCornerSubPix() {
        let img = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_8U, scalar: Scalar(128))
        let truthPosition = Point(x: img.cols() / 2, y: img.rows() / 2)

        let r = Rect(point: Point(x: 0, y: 0), point: truthPosition)
        Imgproc.rectangle(img: img, pt1: r.tl(), pt2: r.br(), color: Scalar(0), thickness: Core.FILLED)
        let corners = MatOfPoint2f(array: [Point2f(x: Float(truthPosition.x + 1), y: Float(truthPosition.y + 1))])
        let winSize = Size(width: 2, height: 2)
        let zeroZone = Size(width: -1, height: -1)
        let criteria = TermCriteria(type: TermCriteria.eps, maxCount: 0, epsilon: 0.01)

        Imgproc.cornerSubPix(image: img, corners: corners, winSize: winSize, zeroZone: zeroZone, criteria: criteria)

        assertPoint2fEquals(Point2f(x: Float(truthPosition.x), y: Float(truthPosition.y)), corners.toArray()[0], OpenCVTestCase.weakFEPS)
    }

    func testDilateMatMatMat() throws {
        let kernel = Mat()

        Imgproc.dilate(src: gray255, dst: dst, kernel: kernel)

        try assertMatEqual(gray255, dst)

        Imgproc.dilate(src: gray1, dst: dst, kernel: kernel)

        try assertMatEqual(gray1, dst)
    }

    func testDistanceTransformWithLabels() throws {
        let dstLables = getMat(CvType.CV_32SC1, vals: [0])
        let labels = Mat()

        Imgproc.distanceTransform(src: gray128, dst: dst, labels: labels, distanceType: .DIST_L2, maskSize: .DIST_MASK_3)

        try assertMatEqual(dstLables, labels)
        try assertMatEqual(getMat(CvType.CV_32FC1, vals: [8192]), dst, OpenCVTestCase.EPS)
    }

    func testDrawContoursMatListOfMatIntScalar() {
        let gray0clone = gray0.clone()
        Imgproc.rectangle(img: gray0clone, pt1: Point(x: 1, y: 2), pt2: Point(x: 7, y: 8), color: Scalar(100))
        let contours = NSMutableArray()
        Imgproc.findContours(image: gray0clone, contours: contours, hierarchy: Mat(), mode: .RETR_EXTERNAL, method: .CHAIN_APPROX_SIMPLE)

        Imgproc.drawContours(image: gray0clone, contours: contours as! [[Point]], contourIdx: -1, color: Scalar(0))

        XCTAssertEqual(0, Core.countNonZero(src: gray0clone))
    }

    func testDrawContoursMatListOfMatIntScalarInt() {
        let gray0clone = gray0.clone()
        Imgproc.rectangle(img: gray0clone, pt1: Point(x: 1, y: 2), pt2: Point(x: 7, y: 8), color: Scalar(100))
        let contours = NSMutableArray()
        Imgproc.findContours(image: gray0clone, contours: contours, hierarchy: Mat(), mode: .RETR_EXTERNAL, method: .CHAIN_APPROX_SIMPLE)

        Imgproc.drawContours(image: gray0clone, contours: contours as! [[Point]], contourIdx: -1, color: Scalar(0), thickness: Core.FILLED)

        XCTAssertEqual(0, Core.countNonZero(src: gray0clone))
    }


    func testEqualizeHist() throws {
        Imgproc.equalizeHist(src: gray0, dst: dst)
        try assertMatEqual(gray0, dst)

        Imgproc.equalizeHist(src: gray255, dst: dst)
        try assertMatEqual(gray255, dst)
    }

    func testErodeMatMatMat() throws {
        let kernel = Mat()

        Imgproc.erode(src: gray128, dst: dst, kernel: kernel)

        try assertMatEqual(gray128, dst)
    }

    func testErodeMatMatMatPointInt() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_8U)
        try src.put(row: 0, col: 0, data: [15, 9, 10] as [Int8])
        try src.put(row: 1, col: 0, data: [10, 8, 12] as [Int8])
        try src.put(row: 2, col: 0, data: [12, 20, 25] as [Int8])
        let kernel = Mat()

        Imgproc.erode(src: src, dst: dst, kernel: kernel, anchor: anchorPoint, iterations: 10)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_8U, scalar: Scalar(8))
        try assertMatEqual(truth!, dst)
    }

    func testErodeMatMatMatPointIntIntScalar() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_8U)
        try src.put(row: 0, col: 0, data: [15, 9, 10] as [Int8])
        try src.put(row: 1, col: 0, data: [10, 8, 12] as [Int8])
        try src.put(row: 2, col: 0, data: [12, 20, 25] as [Int8])
        let kernel = Mat()
        let sc = Scalar(3, 3)

        Imgproc.erode(src: src, dst: dst, kernel: kernel, anchor: anchorPoint, iterations: 10, borderType: .BORDER_REFLECT, borderValue: sc)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_8U, scalar: Scalar(8))
        try assertMatEqual(truth!, dst)
    }

    func testFilter2DMatMatIntMat() throws {
        let src = Mat.eye(rows: 4, cols: 4, type: CvType.CV_32F)
        let kernel = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(1))

        Imgproc.filter2D(src: src, dst: dst, ddepth: -1, kernel: kernel)

        truth = Mat(rows: 4, cols: 4, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [2, 2, 1, 0] as [Float])
        try truth!.put(row: 1, col: 0, data: [2, 2, 1, 0] as [Float])
        try truth!.put(row: 2, col: 0, data: [1, 1, 2, 1] as [Float])
        try truth!.put(row: 3, col: 0, data: [0, 0, 1, 2] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testFilter2DMatMatIntMatPointDoubleInt() throws {
        let kernel = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(0))
        let point = Point(x: 0, y: 0)

        Imgproc.filter2D(src: gray128, dst: dst, ddepth: -1, kernel: kernel, anchor: point, delta: 2, borderType: .BORDER_CONSTANT)

        try assertMatEqual(gray2, dst)
    }

    func testFindContoursMatListOfMatMatIntInt() {
        let img = Mat(rows: 50, cols: 50, type: CvType.CV_8UC1, scalar: Scalar(0))
        let contours = NSMutableArray()
        let hierarchy = Mat()

        Imgproc.findContours(image: img, contours: contours, hierarchy: hierarchy, mode: .RETR_EXTERNAL, method: .CHAIN_APPROX_SIMPLE)

        // no contours on empty image
        XCTAssertEqual(contours.count, 0)
        XCTAssertEqual(contours.count, hierarchy.total())

        Imgproc.rectangle(img: img, pt1: Point(x: 10, y: 20), pt2: Point(x: 20, y: 30), color: Scalar(100), thickness: 3, lineType: .LINE_AA, shift: 0)
        Imgproc.rectangle(img: img, pt1: Point(x: 30, y: 35), pt2: Point(x: 40, y: 45), color: Scalar(200))

        Imgproc.findContours(image: img, contours: contours, hierarchy: hierarchy, mode: .RETR_EXTERNAL, method: .CHAIN_APPROX_SIMPLE)

        // two contours of two rectangles
        XCTAssertEqual(contours.count, 2)
        XCTAssertEqual(contours.count, hierarchy.total())
    }

    func testFindContoursMatListOfMatMatIntIntPoint() throws {
        let img = Mat(rows: 50, cols: 50, type: CvType.CV_8UC1, scalar: Scalar(0))
        let img2 = img.submat(rowStart: 5, rowEnd: 50, colStart: 3, colEnd: 50)
        let contours = NSMutableArray()
        let contours2 = NSMutableArray()
        let hierarchy = Mat()

        Imgproc.rectangle(img: img, pt1: Point(x: 10, y: 20), pt2: Point(x: 20, y: 30), color: Scalar(100), thickness: 3, lineType: .LINE_AA, shift: 0)
        Imgproc.rectangle(img: img, pt1: Point(x: 30, y: 35), pt2: Point(x: 40, y: 45), color: Scalar(200))

        Imgproc.findContours(image: img, contours: contours, hierarchy: hierarchy, mode: .RETR_EXTERNAL, method: .CHAIN_APPROX_SIMPLE)
        Imgproc.findContours(image: img2, contours: contours2, hierarchy: hierarchy, mode: .RETR_EXTERNAL, method: .CHAIN_APPROX_SIMPLE, offset: Point(x: 3, y: 5))

        XCTAssertEqual(contours.count, contours2.count)
        XCTAssert(contours[0] as! [Point] == contours2[0] as! [Point])
    }

    func testFitEllipse() {
        let points = [Point2f(x: 0, y: 0), Point2f(x: -1, y: 1), Point2f(x: 1, y: 1), Point2f(x: 1, y: -1), Point2f(x: -1, y: -1)]
        let rrect = Imgproc.fitEllipse(points: points)

        let FIT_ELLIPSE_CENTER_EPS:Float = 0.01
        let FIT_ELLIPSE_SIZE_EPS:Float = 0.4

        assertPoint2fEquals(Point2f(x: 0, y: 0), rrect.center, FIT_ELLIPSE_CENTER_EPS)
        XCTAssertEqual(Float(2.828), rrect.size.width, accuracy: FIT_ELLIPSE_SIZE_EPS)
        XCTAssertEqual(Float(2.828), rrect.size.height, accuracy: FIT_ELLIPSE_SIZE_EPS)
    }

    func testFitLine() throws {
        let points = Mat(rows: 1, cols: 4, type: CvType.CV_32FC2)
        try points.put(row: 0, col: 0, data: [0, 0, 2, 3, 3, 4, 5, 8] as [Float])

        let linePoints = Mat(rows: 4, cols: 1, type: CvType.CV_32FC1)
        try linePoints.put(row: 0, col: 0, data: [0.53198653, 0.84675282, 2.5, 3.75] as [Float])

        Imgproc.fitLine(points: points, line: dst, distType: .DIST_L12, param: 0, reps: 0.01, aeps: 0.01)

        try assertMatEqual(linePoints, dst, OpenCVTestCase.EPS)
    }

    func testFloodFillMatMatPointScalar() throws {
        let mask = Mat(rows: OpenCVTestCase.matSize + 2, cols: OpenCVTestCase.matSize + 2, type: CvType.CV_8U, scalar: Scalar(0))
        let img = gray0
        Imgproc.circle(img: mask, center: Point(x: OpenCVTestCase.matSize / 2 + 1, y: OpenCVTestCase.matSize / 2 + 1), radius: 3, color: Scalar(2))

        let retval = Imgproc.floodFill(image: img, mask: mask, seedPoint: Point(x: OpenCVTestCase.matSize / 2, y: OpenCVTestCase.matSize / 2), newVal: Scalar(1))

        XCTAssertEqual(Core.countNonZero(src: img), retval)
        Imgproc.circle(img: mask, center: Point(x: OpenCVTestCase.matSize / 2 + 1, y: OpenCVTestCase.matSize / 2 + 1), radius: 3, color: Scalar(0))
        XCTAssertEqual(retval + 4 * (OpenCVTestCase.matSize + 1), Core.countNonZero(src: mask))
        try assertMatEqual(mask.submat(rowStart: 1, rowEnd: OpenCVTestCase.matSize + 1, colStart: 1, colEnd: OpenCVTestCase.matSize + 1), img)
    }

    func testFloodFillMatMatPointScalar_WithoutMask() {
        let img = gray0
        Imgproc.circle(img: img, center: Point(x: OpenCVTestCase.matSize / 2, y: OpenCVTestCase.matSize / 2), radius: 3, color: Scalar(2))

        // TODO: ideally we should pass null instead of "new Mat()"
        let retval = Imgproc.floodFill(image: img, mask: Mat(), seedPoint: Point(x: OpenCVTestCase.matSize / 2, y: OpenCVTestCase.matSize / 2), newVal: Scalar(1))

        Imgproc.circle(img: img, center: Point(x: OpenCVTestCase.matSize / 2, y: OpenCVTestCase.matSize / 2), radius: 3, color: Scalar(0))
        XCTAssertEqual(Core.countNonZero(src: img), retval)
    }

    func testGaussianBlurMatMatSizeDouble() throws {
        Imgproc.GaussianBlur(src: gray0, dst: dst, ksize: size, sigmaX: 1)
        try assertMatEqual(gray0, dst)

        Imgproc.GaussianBlur(src: gray2, dst: dst, ksize: size, sigmaX: 1)
        try assertMatEqual(gray2, dst)
    }

    func testGaussianBlurMatMatSizeDoubleDouble() throws {
        Imgproc.GaussianBlur(src: gray2, dst: dst, ksize: size, sigmaX: 0, sigmaY: 0)

        try assertMatEqual(gray2, dst)
    }

    func testGaussianBlurMatMatSizeDoubleDoubleInt() throws {
        Imgproc.GaussianBlur(src: gray2, dst: dst, ksize: size, sigmaX: 1, sigmaY: 3, borderType: .BORDER_REFLECT)

        try assertMatEqual(gray2, dst)
    }

    func testGetAffineTransform() throws {
        let src = [Point2f(x: 2, y: 3), Point2f(x: 3, y: 1), Point2f(x: 1, y: 4)]
        let dst = [Point2f(x: 3, y: 3), Point2f(x: 7, y: 4), Point2f(x: 5, y: 6)]

        let transform = Imgproc.getAffineTransform(src: src, dst: dst)

        let truth = Mat(rows: 2, cols: 3, type: CvType.CV_64FC1)

        try truth.put(row: 0, col: 0, data: [-8.0, -6.0, 37.0])
        try truth.put(row: 1, col: 0, data: [-7.0, -4.0, 29.0])
        try assertMatEqual(truth, transform, OpenCVTestCase.EPS)
    }

    func testGetDerivKernelsMatMatIntIntInt() throws {
        let kx = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        let ky = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        let expKx = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        let expKy = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try kx.put(row: 0, col: 0, data: [1, 1] as [Float])
        try kx.put(row: 1, col: 0, data: [1, 1] as [Float])
        try ky.put(row: 0, col: 0, data: [2, 2] as [Float])
        try ky.put(row: 1, col: 0, data: [2, 2] as [Float])
        try expKx.put(row: 0, col: 0, data: [1, -2, 1] as [Float])
        try expKy.put(row: 0, col: 0, data: [1, -2, 1] as [Float])

        Imgproc.getDerivKernels(kx: kx, ky: ky, dx: 2, dy: 2, ksize: 3)

        try assertMatEqual(expKx, kx, OpenCVTestCase.EPS)
        try assertMatEqual(expKy, ky, OpenCVTestCase.EPS)
    }

    func testGetDerivKernelsMatMatIntIntIntBooleanInt() throws {
        let kx = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        let ky = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        let expKx = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        let expKy = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try kx.put(row: 0, col: 0, data: [1, 1] as [Float])
        try kx.put(row: 1, col: 0, data: [1, 1] as [Float])
        try ky.put(row: 0, col: 0, data: [2, 2] as [Float])
        try ky.put(row: 1, col: 0, data: [2, 2] as [Float])
        try expKx.put(row: 0, col: 0, data: [1, -2, 1] as [Float])
        try expKy.put(row: 0, col: 0, data: [1, -2, 1] as [Float])

        Imgproc.getDerivKernels(kx: kx, ky: ky, dx: 2, dy: 2, ksize: 3, normalize: true, ktype: CvType.CV_32F)

        try assertMatEqual(expKx, kx, OpenCVTestCase.EPS)
        try assertMatEqual(expKy, ky, OpenCVTestCase.EPS)
        // TODO_: write better test
    }

    func testGetGaussianKernelIntDouble() throws {
        dst = Imgproc.getGaussianKernel(ksize: 1, sigma: 0.5)

        truth = Mat(rows: 1, cols: 1, type: CvType.CV_64FC1, scalar: Scalar(1))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testGetGaussianKernelIntDoubleInt() throws {
        dst = Imgproc.getGaussianKernel(ksize: 3, sigma: 0.8, ktype: CvType.CV_32F)

        truth = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [0.23899426, 0.52201146, 0.23899426] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testGetRectSubPixMatSizePointMat() throws {
        let size = Size(width: 3, height: 3)
        let center = Point2f(x: Float(gray255.cols() / 2), y: Float(gray255.rows() / 2))

        Imgproc.getRectSubPix(image: gray255, patchSize: size, center: center, patch: dst)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_8U, scalar: Scalar(255))
        try assertMatEqual(truth!, dst)
    }

    func testGetRectSubPixMatSizePointMatInt() throws {
        let src = Mat(rows: 10, cols: 10, type: CvType.CV_32F, scalar: Scalar(2))
        let patchSize = Size(width: 5, height: 5)
        let center = Point2f(x: Float(src.cols() / 2), y: Float(src.rows() / 2))

        Imgproc.getRectSubPix(image: src, patchSize: patchSize, center: center, patch: dst)

        truth = Mat(rows: 5, cols: 5, type: CvType.CV_32F, scalar: Scalar(2))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testGetRotationMatrix2D() throws {
        let center = Point2f(x: 0, y: 0)

        dst = Imgproc.getRotationMatrix2D(center: center, angle: 0, scale: 1)

        truth = Mat(rows: 2, cols: 3, type: CvType.CV_64F)
        try truth!.put(row: 0, col: 0, data: [1.0, 0.0, 0.0])
        try truth!.put(row: 1, col: 0, data: [0.0, 1.0, 0.0])

        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testGetStructuringElementIntSize() throws {
        dst = Imgproc.getStructuringElement(shape: .MORPH_RECT, ksize: size)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_8UC1, scalar: Scalar(1))
        try assertMatEqual(truth!, dst)
    }

    func testGetStructuringElementIntSizePoint() throws {
        dst = Imgproc.getStructuringElement(shape: .MORPH_CROSS, ksize: size, anchor: anchorPoint)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_8UC1)
        try truth!.put(row: 0, col: 0, data: [0, 0, 1] as [Int8])
        try truth!.put(row: 1, col: 0, data: [0, 0, 1] as [Int8])
        try truth!.put(row: 2, col: 0, data: [1, 1, 1] as [Int8])
        try assertMatEqual(truth!, dst)
    }

    func testGoodFeaturesToTrackMatListOfPointIntDoubleDouble() {
        let src = gray0
        Imgproc.rectangle(img: src, pt1: Point(x: 2, y: 2), pt2: Point(x: 8, y: 8), color: Scalar(100), thickness: -1)
        let lp = NSMutableArray()

        Imgproc.goodFeaturesToTrack(image: src, corners: lp, maxCorners: 100, qualityLevel: 0.01, minDistance: 3)

        XCTAssertEqual(4, lp.count)
    }

    func testGoodFeaturesToTrackMatListOfPointIntDoubleDoubleMatIntBooleanDouble() {
        let src = gray0
        Imgproc.rectangle(img: src, pt1: Point(x: 2, y: 2), pt2: Point(x: 8, y: 8), color: Scalar(100), thickness: -1)
        let lp = NSMutableArray()

        Imgproc.goodFeaturesToTrack(image: src, corners: lp, maxCorners: 100, qualityLevel: 0.01, minDistance: 3, mask: gray1, blockSize: 4, gradientSize: 3, useHarrisDetector: true, k: 0)

        XCTAssertEqual(4, lp.count)
    }

    func testHoughCirclesMatMatIntDoubleDouble() {
        let sz:Int32 = 512
        let img = Mat(rows: sz, cols: sz, type: CvType.CV_8U, scalar: Scalar(128))
        let circles = Mat()

        Imgproc.HoughCircles(image: img, circles: circles, method: .HOUGH_GRADIENT, dp: 2.0, minDist: Double(img.rows() / 4))

        XCTAssertEqual(0, circles.cols())
    }

    func testHoughCirclesMatMatIntDoubleDouble1() {
        let sz: Int32 = 512
        let img = Mat(rows: sz, cols: sz, type: CvType.CV_8U, scalar: Scalar(128))
        let circles = Mat()

        let center = Point(x: img.cols() / 2, y: img.rows() / 2)
        let radius = min(img.cols() / 4, img.rows() / 4)
        Imgproc.circle(img: img, center: center, radius: radius, color: colorBlack, thickness: 3)

        Imgproc.HoughCircles(image: img, circles: circles, method: .HOUGH_GRADIENT, dp: 2.0, minDist: Double(img.rows() / 4))

        XCTAssertEqual(1, circles.cols())
    }

    func testHoughLinesMatMatDoubleDoubleInt() {
        let sz:Int32 = 512
        let img = Mat(rows: sz, cols: sz, type: CvType.CV_8U, scalar: Scalar(0))
        let point1 = Point(x: 50, y: 50)
        let point2 = Point(x: img.cols() / 2, y: img.rows() / 2)
        Imgproc.line(img: img, pt1: point1, pt2: point2, color: colorWhite, thickness: 1)
        let lines = Mat()

        Imgproc.HoughLines(image: img, lines: lines, rho: 1, theta: 3.1415926/180, threshold: 100)

        XCTAssertEqual(1, lines.cols())
    }

    func testHoughLinesPMatMatDoubleDoubleInt() {
        let sz:Int32 = 512
        let img = Mat(rows: sz, cols: sz, type: CvType.CV_8U, scalar: Scalar(0))
        let point1 = Point(x: 0, y: 0)
        let point2 = Point(x: sz, y: sz)
        let point3 = Point(x: sz, y: 0)
        let point4 = Point(x: 2*sz/3, y: sz/3)
        Imgproc.line(img: img, pt1: point1, pt2: point2, color: Scalar.all(255), thickness: 1)
        Imgproc.line(img: img, pt1: point3, pt2: point4, color: Scalar.all(255), thickness: 1)
        let lines = Mat()

        Imgproc.HoughLinesP(image: img, lines: lines, rho: 1, theta: 3.1415926/180, threshold: 100)

        XCTAssertEqual(2, lines.rows())
    }

    func testIntegral2MatMatMat() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(3))
        let expSum = Mat(rows: 4, cols: 4, type: CvType.CV_64F)
        let expSqsum = Mat(rows: 4, cols: 4, type: CvType.CV_64F)
        let sum = Mat()
        let sqsum = Mat()

        try expSum.put(row: 0, col: 0, data: [0.0, 0.0, 0.0, 0.0])
        try expSum.put(row: 1, col: 0, data: [0.0, 3.0, 6.0, 9.0])
        try expSum.put(row: 2, col: 0, data: [0.0, 6.0, 12.0, 18.0])
        try expSum.put(row: 3, col: 0, data: [0.0, 9.0, 18.0, 27.0])

        try expSqsum.put(row: 0, col: 0, data: [0.0, 0.0, 0.0, 0.0])
        try expSqsum.put(row: 1, col: 0, data: [0.0, 9.0, 18.0, 27.0])
        try expSqsum.put(row: 2, col: 0, data: [0.0, 18.0, 36.0, 54.0])
        try expSqsum.put(row: 3, col: 0, data: [0.0, 27.0, 54.0, 81.0])

        Imgproc.integral(src: src, sum: sum, sqsum: sqsum)

        try assertMatEqual(expSum, sum, OpenCVTestCase.EPS)
        try assertMatEqual(expSqsum, sqsum, OpenCVTestCase.EPS)
    }

    func testIntegral2MatMatMatInt() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(3))
        let expSum = Mat(rows: 4, cols: 4, type: CvType.CV_64F)
        let expSqsum = Mat(rows: 4, cols: 4, type: CvType.CV_64F)
        let sum = Mat()
        let sqsum = Mat()

        try expSum.put(row: 0, col: 0, data: [0.0, 0.0, 0.0, 0.0])
        try expSum.put(row: 1, col: 0, data: [0.0, 3.0, 6.0, 9.0])
        try expSum.put(row: 2, col: 0, data: [0.0, 6.0, 12.0, 18.0])
        try expSum.put(row: 3, col: 0, data: [0.0, 9.0, 18.0, 27.0])

        try expSqsum.put(row: 0, col: 0, data: [0.0, 0.0, 0.0, 0.0])
        try expSqsum.put(row: 1, col: 0, data: [0.0, 9.0, 18.0, 27.0])
        try expSqsum.put(row: 2, col: 0, data: [0.0, 18.0, 36.0, 54.0])
        try expSqsum.put(row: 3, col: 0, data: [0.0, 27.0, 54.0, 81.0])

        Imgproc.integral(src: src, sum: sum, sqsum: sqsum, sdepth: CvType.CV_64F, sqdepth: CvType.CV_64F)

        try assertMatEqual(expSum, sum, OpenCVTestCase.EPS)
        try assertMatEqual(expSqsum, sqsum, OpenCVTestCase.EPS)
    }

    func testIntegral3MatMatMatMat() throws {
        let src = Mat(rows: 1, cols: 1, type: CvType.CV_32F, scalar: Scalar(1))
        let expSum = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_64F)
        let expSqsum = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_64F)
        let expTilted = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_64F)
        let sum = Mat()
        let sqsum = Mat()
        let tilted = Mat()

        try expSum.put(row: 0, col: 0, data: [0.0, 0.0])
        try expSum.put(row: 1, col: 0, data: [0.0, 1.0])

        try expSqsum.put(row: 0, col: 0, data: [0.0, 0.0])
        try expSqsum.put(row: 1, col: 0, data: [0.0, 1.0])

        try expTilted.put(row: 0, col: 0, data: [0.0, 0.0])
        try expTilted.put(row: 1, col: 0, data: [0.0, 1.0])

        Imgproc.integral(src: src, sum: sum, sqsum: sqsum, tilted: tilted)

        try assertMatEqual(expSum, sum, OpenCVTestCase.EPS)
        try assertMatEqual(expSqsum, sqsum, OpenCVTestCase.EPS)
        try assertMatEqual(expTilted, tilted, OpenCVTestCase.EPS)
    }

    func testIntegral3MatMatMatMatInt() throws {
        let src = Mat(rows: 1, cols: 1, type: CvType.CV_32F, scalar: Scalar(1))
        let expSum = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_64F)
        let expSqsum = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_64F)
        let expTilted = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_64F)
        let sum = Mat()
        let sqsum = Mat()
        let tilted = Mat()

        try expSum.put(row: 0, col: 0, data: [0.0, 0.0])
        try expSum.put(row: 1, col: 0, data: [0.0, 1.0])

        try expSqsum.put(row: 0, col: 0, data: [0.0, 0.0])
        try expSqsum.put(row: 1, col: 0, data: [0.0, 1.0])

        try expTilted.put(row: 0, col: 0, data: [0.0, 0.0])
        try expTilted.put(row: 1, col: 0, data: [0.0, 1.0])

        Imgproc.integral(src: src, sum: sum, sqsum: sqsum, tilted: tilted, sdepth: CvType.CV_64F, sqdepth: CvType.CV_64F)

        try assertMatEqual(expSum, sum, OpenCVTestCase.EPS)
        try assertMatEqual(expSqsum, sqsum, OpenCVTestCase.EPS)
        try assertMatEqual(expTilted, tilted, OpenCVTestCase.EPS)
    }

    func testIntegralMatMat() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(2))

        Imgproc.integral(src: src, sum: dst)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_64F)
        try truth!.put(row: 0, col: 0, data: [0.0, 0.0, 0.0])
        try truth!.put(row: 1, col: 0, data: [0.0, 2.0, 4.0])
        try truth!.put(row: 2, col: 0, data: [0.0, 4.0, 8.0])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testIntegralMatMatInt() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(2))

        Imgproc.integral(src: src, sum: dst, sdepth: CvType.CV_64F)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_64F)
        try truth!.put(row: 0, col: 0, data: [0.0, 0.0, 0.0])
        try truth!.put(row: 1, col: 0, data: [0.0, 2.0, 4.0])
        try truth!.put(row: 2, col: 0, data: [0.0, 4.0, 8.0])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testInvertAffineTransform() throws {
        let src = Mat(rows: 2, cols: 3, type: CvType.CV_64F, scalar: Scalar(1))

        Imgproc.invertAffineTransform(M: src, iM: dst)

        truth = Mat(rows: 2, cols: 3, type: CvType.CV_64F, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testIsContourConvex() {
        let contour1 = [Point(x: 0, y: 0), Point(x: 10, y: 0), Point(x: 10, y: 10), Point(x: 5, y: 4)]

        XCTAssertFalse(Imgproc.isContourConvex(contour: contour1))

        let contour2 = [Point(x: 0, y: 0), Point(x: 10, y: 0), Point(x: 10, y: 10), Point(x: 5, y: 6)]

        XCTAssert(Imgproc.isContourConvex(contour: contour2))
    }

    func testLaplacianMatMatInt() throws {
        Imgproc.Laplacian(src: gray0, dst: dst, ddepth: CvType.CV_8U)

        try assertMatEqual(gray0, dst)
    }

    func testLaplacianMatMatIntIntDoubleDouble() throws {
        let src = Mat.eye(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)

        Imgproc.Laplacian(src: src, dst: dst, ddepth: CvType.CV_32F, ksize: 1, scale: 2, delta: OpenCVTestCase.EPS)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [-7.9990001, 8.0009995])
        try truth!.put(row: 1, col: 0, data: [8.0009995, -7.9990001])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testLaplacianMatMatIntIntDoubleDoubleInt() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(2))

        Imgproc.Laplacian(src: src, dst: dst, ddepth: CvType.CV_32F, ksize: 1, scale: 2, delta: OpenCVTestCase.EPS, borderType: .BORDER_REFLECT)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F, scalar: Scalar(0.00099945068))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testMatchShapes() throws {
        let contour1 = Mat(rows: 1, cols: 4, type: CvType.CV_32FC2)
        let contour2 = Mat(rows: 1, cols: 4, type: CvType.CV_32FC2)
        try contour1.put(row: 0, col: 0, data: [1, 1, 5, 1, 4, 3, 6, 2] as [Float])
        try contour2.put(row: 0, col: 0, data: [1, 1, 6, 1, 4, 1, 2, 5] as [Float])

        let distance = Imgproc.matchShapes(contour1: contour1, contour2: contour2, method: .CONTOURS_MATCH_I1, parameter: 1)

        XCTAssertEqual(2.81109697365334, distance, accuracy:OpenCVTestCase.EPS)
    }

    func testMatchTemplate() throws {
        let image = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U)
        let templ = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U)
        try image.put(row: 0, col: 0, data: [1, 2, 3, 4] as [Int8])
        try templ.put(row: 0, col: 0, data: [5, 6, 7, 8] as [Int8])

        Imgproc.matchTemplate(image: image, templ: templ, result: dst, method: .TM_CCORR)

        truth = Mat(rows: 1, cols: 1, type: CvType.CV_32F, scalar: Scalar(70))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)

        Imgproc.matchTemplate(image: gray255, templ: gray0, result: dst, method: .TM_CCORR)

        truth = Mat(rows: 1, cols: 1, type: CvType.CV_32F, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testMedianBlur() throws {
        Imgproc.medianBlur(src: gray255, dst: dst, ksize: 5)
        try assertMatEqual(gray255, dst)

        Imgproc.medianBlur(src: gray2, dst: dst, ksize: 3)
        try assertMatEqual(gray2, dst)
        // TODO_: write better test
    }

    func testMinAreaRect() {
        let points = [Point2f(x: 1, y: 1), Point2f(x: 5, y: 1), Point2f(x: 4, y: 3), Point2f(x: 6, y: 2)]

        let rrect = Imgproc.minAreaRect(points: points)

        XCTAssertEqual(Size2f(width: 2, height: 5), rrect.size)
        XCTAssertEqual(-90.0, rrect.angle)
        XCTAssertEqual(Point2f(x: 3.5, y: 2), rrect.center)
    }

    func testMinEnclosingCircle() {
        let points = [Point2f(x: 0, y: 0), Point2f(x: -100, y: 0), Point2f(x: 0, y: -100), Point2f(x: 100, y: 0), Point2f(x: 0, y: 100)]
        let actualCenter = Point2f()
        var radius:Float = 0

        Imgproc.minEnclosingCircle(points: points, center: actualCenter, radius: &radius)

        XCTAssertEqual(Point2f(x: 0, y: 0), actualCenter)
        XCTAssertEqual(100.0, radius, accuracy: 1.0)
    }

    func testMorphologyExMatMatIntMat() throws {
        Imgproc.morphologyEx(src: gray255, dst: dst, op: MorphTypes.MORPH_GRADIENT, kernel: gray0)

        try assertMatEqual(gray0, dst)
    }

    func testMorphologyExMatMatIntMatPointInt() throws {
        let src = Mat.eye(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U)

        let kernel = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U, scalar: Scalar(0))
        let point = Point(x: 0, y: 0)

        Imgproc.morphologyEx(src: src, dst: dst, op: MorphTypes.MORPH_CLOSE, kernel: kernel, anchor: point, iterations: 10)

        truth = Mat.eye(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U)
        try assertMatEqual(truth!, dst)
    }


    func testMorphologyExMatMatIntMatPointIntIntScalar() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U)
        try src.put(row: 0, col: 0, data: [2, 1] as [Int8])
        try src.put(row: 1, col: 0, data: [2, 1] as [Int8])

        let kernel = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U, scalar: Scalar(1))
        let point = Point(x: 1, y: 1)
        let sc = Scalar(3, 3)

        Imgproc.morphologyEx(src: src, dst: dst, op: MorphTypes.MORPH_TOPHAT, kernel: kernel, anchor: point, iterations: 10, borderType: .BORDER_REFLECT, borderValue: sc)
        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8U)
        try truth!.put(row: 0, col: 0, data: [1, 0] as [Int8])
        try truth!.put(row: 1, col: 0, data: [1, 0] as [Int8])
        try assertMatEqual(truth!, dst)
    }

    func testPointPolygonTest() {
        let contour = [Point2f(x: 0, y: 0), Point2f(x: 1, y: 3), Point2f(x: 3, y: 4), Point2f(x: 4, y: 3), Point2f(x: 2, y: 1)]
        let sign1 = Imgproc.pointPolygonTest(contour: contour, pt: Point2f(x: 2, y: 2), measureDist: false)
        XCTAssertEqual(1.0, sign1)

        let sign2 = Imgproc.pointPolygonTest(contour: contour, pt: Point2f(x: 4, y: 4), measureDist: true)
        XCTAssertEqual(-sqrt(0.5), sign2)
    }

    func testPreCornerDetectMatMatInt() throws {
        let src = Mat(rows: 4, cols: 4, type: CvType.CV_32F, scalar: Scalar(1))
        let ksize:Int32 = 3

        Imgproc.preCornerDetect(src: src, dst: dst, ksize: ksize)

        truth = Mat(rows: 4, cols: 4, type: CvType.CV_32F, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testPreCornerDetectMatMatIntInt() throws {
        let src = Mat(rows: 4, cols: 4, type: CvType.CV_32F, scalar: Scalar(1))
        let ksize:Int32 = 3

        Imgproc.preCornerDetect(src: src, dst: dst, ksize: ksize, borderType: .BORDER_REFLECT)

        truth = Mat(rows: 4, cols: 4, type: CvType.CV_32F, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testPyrDownMatMat() throws {
        let src = Mat(rows: 4, cols: 4, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [2, 1, 4, 2] as [Float])
        try src.put(row: 1, col: 0, data: [3, 2, 6, 8] as [Float])
        try src.put(row: 2, col: 0, data: [4, 6, 8, 10] as [Float])
        try src.put(row: 3, col: 0, data: [12, 32, 6, 18] as [Float])

        Imgproc.pyrDown(src: src, dst: dst)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [2.78125, 4.609375])
        try truth!.put(row: 1, col: 0, data: [8.546875, 8.8515625])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testPyrDownMatMatSize() throws {
        let src = Mat(rows: 4, cols: 4, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [2, 1, 4, 2] as [Float])
        try src.put(row: 1, col: 0, data: [3, 2, 6, 8] as [Float])
        try src.put(row: 2, col: 0, data: [4, 6, 8, 10] as [Float])
        try src.put(row: 3, col: 0, data: [12, 32, 6, 18] as [Float])
        let dstSize = Size(width: 2, height: 2)

        Imgproc.pyrDown(src: src, dst: dst, dstsize: dstSize)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [2.78125, 4.609375])
        try truth!.put(row: 1, col: 0, data: [8.546875, 8.8515625])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testPyrMeanShiftFilteringMatMatDoubleDouble() throws {
        let src = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_8UC3, scalar: Scalar(0))

        Imgproc.pyrMeanShiftFiltering(src: src, dst: dst, sp: 10, sr: 50)

        try assertMatEqual(src, dst)
    }

    func testPyrUpMatMat() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [2, 1] as [Float])
        try src.put(row: 1, col: 0, data: [3, 2] as [Float])

        Imgproc.pyrUp(src: src, dst: dst)

        truth = Mat(rows: 4, cols: 4, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [2,     1.75,  1.375, 1.25])
        try truth!.put(row: 1, col: 0, data: [2.25,  2,     1.625, 1.5])
        try truth!.put(row: 2, col: 0, data: [2.625, 2.375, 2,     1.875])
        try truth!.put(row: 3, col: 0, data: [2.75,  2.5,   2.125, 2])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testRemapMatMatMatMatInt() throws {
        // this test does something weird
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(2))
        let map1 = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        let map2 = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)

        try map1.put(row: 0, col: 0, data: [3, 6, 5] as [Float])
        try map2.put(row: 0, col: 0, data: [4, 8, 12] as [Float])

        Imgproc.remap(src: src, dst: dst, map1: map1, map2: map2, interpolation: InterpolationFlags.INTER_LINEAR.rawValue)

        truth = Mat(rows: 1, cols: 3, type: CvType.CV_32F, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testRemapMatMatMatMatIntIntScalar() throws {
        // this test does something weird
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(2))
        let map1 = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        let map2 = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)

        let sc = Scalar(0)

        try map1.put(row: 0, col: 0, data: [3, 6, 5, 0] as [Float])
        try map2.put(row: 0, col: 0, data: [4, 8, 12] as [Float])

        truth = Mat(rows: 1, cols: 3, type: CvType.CV_32F, scalar: Scalar(2))

        Imgproc.remap(src: src, dst: dst, map1: map1, map2: map2, interpolation: InterpolationFlags.INTER_LINEAR.rawValue, borderMode: .BORDER_REFLECT, borderValue: sc)
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testResizeMatMatSize() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_8UC1, scalar: Scalar(1))
        let dsize = Size(width: 1, height: 1)

        Imgproc.resize(src: src, dst: dst, dsize: dsize, fx: 0, fy: 0, interpolation: InterpolationFlags.INTER_LINEAR_EXACT.rawValue)

        truth = Mat(rows: 1, cols: 1, type: CvType.CV_8UC1, scalar: Scalar(1))
        try assertMatEqual(truth!, dst)
    }

    func testResizeMatMatSizeDoubleDoubleInt() throws {
        Imgproc.resize(src: gray255, dst: dst, dsize: Size(width: 2, height: 2), fx: 0, fy: 0, interpolation: InterpolationFlags.INTER_AREA.rawValue)

        truth = Mat(rows: 2, cols: 2, type: CvType.CV_8UC1, scalar: Scalar(255))
        try assertMatEqual(truth!, dst)
    }

    func testScharrMatMatIntIntInt() throws {
        let src = Mat.eye(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)

        Imgproc.Scharr(src: src, dst: dst, ddepth: CvType.CV_32F, dx: 1, dy: 0)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(0))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testScharrMatMatIntIntIntDoubleDouble() throws {
        let src = Mat.eye(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F)

        Imgproc.Scharr(src: src, dst: dst, ddepth: CvType.CV_32F, dx: 1, dy: 0, scale: 1.5, delta: 0.001)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(0.001))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testScharrMatMatIntIntIntDoubleDoubleInt() throws {
        let src = Mat.eye(rows: 3, cols: 3, type: CvType.CV_32F)

        Imgproc.Scharr(src: src, dst: dst, ddepth: CvType.CV_32F, dx: 1, dy: 0, scale: 1.5, delta: 0, borderType: .BORDER_REFLECT)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [-15, -19.5, -4.5])
        try truth!.put(row: 1, col: 0, data: [10.5, 0, -10.5])
        try truth!.put(row: 2, col: 0, data: [4.5, 19.5, 15])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testSepFilter2DMatMatIntMatMat() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(2))
        let kernelX = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        let kernelY = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        try kernelX.put(row: 0, col: 0, data: [4, 3, 7] as [Float])
        try kernelY.put(row: 0, col: 0, data: [9, 4, 2] as [Float])

        Imgproc.sepFilter2D(src: src, dst: dst, ddepth: CvType.CV_32F, kernelX: kernelX, kernelY: kernelY)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(420))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testSepFilter2DMatMatIntMatMatPointDouble() throws {
        let src = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32FC1, scalar: Scalar(2))
        let kernelX = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        try kernelX.put(row: 0, col: 0, data: [2, 2, 2] as [Float])
        let kernelY = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        try kernelY.put(row: 0, col: 0, data: [1, 1, 1] as [Float])

        Imgproc.sepFilter2D(src: src, dst: dst, ddepth: CvType.CV_32F, kernelX: kernelX, kernelY: kernelY, anchor: anchorPoint, delta: OpenCVTestCase.weakEPS)

        truth = Mat(rows: imgprocSz, cols: imgprocSz, type: CvType.CV_32F, scalar: Scalar(36 + OpenCVTestCase.weakEPS))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testSepFilter2DMatMatIntMatMatPointDoubleInt() throws {
        let kernelX = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        try kernelX.put(row: 0, col: 0, data: [2, 2, 2] as [Float])

        let kernelY = Mat(rows: 1, cols: 3, type: CvType.CV_32FC1)
        try kernelY.put(row: 0, col: 0, data: [1, 1, 1] as [Float])

        Imgproc.sepFilter2D(src: gray0, dst: dst, ddepth: CvType.CV_32F, kernelX: kernelX, kernelY: kernelY, anchor: anchorPoint, delta: OpenCVTestCase.weakEPS, borderType: .BORDER_REFLECT)

        truth = Mat(rows: 10, cols: 10, type: CvType.CV_32F, scalar: Scalar(OpenCVTestCase.weakEPS))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testSobelMatMatIntIntInt() throws {
        Imgproc.Sobel(src: gray255, dst: dst, ddepth: CvType.CV_8U, dx: 1, dy: 0)

        try assertMatEqual(gray0, dst)
    }

    func testSobelMatMatIntIntIntIntDoubleDouble() throws {
        Imgproc.Sobel(src: gray255, dst: dst, ddepth: CvType.CV_8U, dx: 1, dy: 0, ksize: 3, scale: 2, delta: 0.001)
        try assertMatEqual(gray0, dst)
    }

    func testSobelMatMatIntIntIntIntDoubleDoubleInt() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [2, 0, 1] as [Float])
        try src.put(row: 1, col: 0, data: [6, 4, 3] as [Float])
        try src.put(row: 2, col: 0, data: [1, 0, 2] as [Float])

        Imgproc.Sobel(src: src, dst: dst, ddepth: CvType.CV_32F, dx: 1, dy: 0, ksize: 3, scale: 2, delta: 0, borderType: .BORDER_REPLICATE)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [-16, -12, 4] as [Float])
        try truth!.put(row: 1, col: 0, data: [-14, -12, 2] as [Float])
        try truth!.put(row: 2, col: 0, data: [-10, 0, 10] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testThreshold() throws {
        Imgproc.threshold(src: makeMask(gray0.clone(), vals: [10]), dst: dst, thresh: 5, maxval: 255, type: .THRESH_TRUNC)
        try assertMatEqual(makeMask(gray0.clone(), vals: [5]), dst)

        Imgproc.threshold(src: makeMask(gray2.clone(), vals: [10]), dst: dst, thresh: 1, maxval: 255, type: .THRESH_BINARY)
        try assertMatEqual(gray255, dst)

        Imgproc.threshold(src: makeMask(gray2.clone(), vals: [10]), dst: dst, thresh: 3, maxval: 255, type: .THRESH_BINARY_INV)
        try assertMatEqual(makeMask(gray255.clone(), vals: [0]), dst)
    }

    func testWarpAffineMatMatMatSize() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [2, 0, 1] as [Float])
        try src.put(row: 1, col: 0, data: [6, 4, 3] as [Float])
        try src.put(row: 2, col: 0, data: [1, 0, 2] as [Float])
        let M = Mat(rows: 2, cols: 3, type: CvType.CV_32F)
        try M.put(row: 0, col: 0, data: [1, 0, 1] as [Float])
        try M.put(row: 1, col: 0, data: [0, 1, 1] as [Float])

        Imgproc.warpAffine(src: src, dst: dst, M: M, dsize: Size(width: 3, height: 3))

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [0, 0, 0] as [Float])
        try truth!.put(row: 1, col: 0, data: [0, 2, 0] as [Float])
        try truth!.put(row: 2, col: 0, data: [0, 6, 4] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testWarpAffineMatMatMatSizeInt() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [2, 4, 1] as [Float])
        try src.put(row: 1, col: 0, data: [6, 4, 3] as [Float])
        try src.put(row: 2, col: 0, data: [0, 2, 2] as [Float])
        let M = Mat(rows: 2, cols: 3, type: CvType.CV_32F)
        try M.put(row: 0, col: 0, data: [1, 0, 0] as [Float])
        try M.put(row: 1, col: 0, data: [0, 0, 1] as [Float])

        Imgproc.warpAffine(src: src, dst: dst, M: M, dsize: Size(width: 2, height: 2), flags: InterpolationFlags.WARP_INVERSE_MAP.rawValue)

        truth = Mat(rows: 2, cols: 2, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [6, 4] as [Float])
        try truth!.put(row: 1, col: 0, data: [6, 4] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testWarpPerspectiveMatMatMatSize() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [2, 4, 1] as [Float])
        try src.put(row: 1, col: 0, data: [0, 4, 5] as [Float])
        try src.put(row: 2, col: 0, data: [1, 2, 2] as [Float])
        let M = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try M.put(row: 0, col: 0, data: [1, 0, 1] as [Float])
        try M.put(row: 1, col: 0, data: [0, 1, 1] as [Float])
        try M.put(row: 2, col: 0, data: [0, 0, 1] as [Float])

        Imgproc.warpPerspective(src: src, dst: dst, M: M, dsize: Size(width: 3, height: 3))

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [0, 0, 0] as [Float])
        try truth!.put(row: 1, col: 0, data: [0, 2, 4] as [Float])
        try truth!.put(row: 2, col: 0, data: [0, 0, 4] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testWatershed() throws {
        let image = Mat.eye(rows: 4, cols: 4, type: CvType.CV_8UC(3))
        let markers = Mat(rows: 4, cols: 4, type: CvType.CV_32SC1, scalar: Scalar(0))

        Imgproc.watershed(image: image, markers: markers)

        truth = Mat(rows: 4, cols: 4, type: CvType.CV_32SC1)
        try truth!.put(row: 0, col: 0, data: [-1, -1, -1, -1] as [Int32])
        try truth!.put(row: 1, col: 0, data: [-1, 0, 0, -1] as [Int32])
        try truth!.put(row: 2, col: 0, data: [-1, 0, 0, -1] as [Int32])
        try truth!.put(row: 3, col: 0, data: [-1, -1, -1, -1] as [Int32])
        try assertMatEqual(truth!, markers)
    }

    func testGetTextSize() {
        let text = "iOS all the way"
        let fontScale:Int32 = 2
        let thickness:Int32 = 3
        var baseLine:Int32 = 0

        Imgproc.getTextSize(text: text, fontFace: .FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale: Double(fontScale), thickness: thickness, baseLine: &baseLine)
        let res = Imgproc.getTextSize(text: text, fontFace: .FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale: Double(fontScale), thickness: thickness, baseLine: &baseLine)

        XCTAssertEqual(431, res.width)
        XCTAssertEqual(44, res.height)
        XCTAssertEqual(20, baseLine)
    }

    func testCircleMatPointIntScalar() {
        let gray0clone = gray0.clone()
        let center = Point(x: gray0clone.cols() / 2, y: gray0clone.rows() / 2)
        let radius = min(gray0clone.cols() / 4, gray0clone.rows() / 4)
        let color = Scalar(128)

        Imgproc.circle(img: gray0clone, center: center, radius: radius, color: color)

        XCTAssert(0 != Core.countNonZero(src: gray0clone))
    }

    func testCircleMatPointIntScalarInt() {
        let gray0clone = gray0.clone()
        let center = Point(x: gray0clone.cols() / 2, y: gray0clone.rows() / 2)
        let radius = min(gray0clone.cols() / 4, gray0clone.rows() / 4)
        let color = Scalar(128)

        Imgproc.circle(img: gray0clone, center: center, radius: radius, color: color, thickness: Core.FILLED)

        XCTAssert(0 != Core.countNonZero(src: gray0clone))
    }

    func testCircleMatPointIntScalarIntIntInt() {
        let gray0clone = gray0.clone()
        let center = Point(x: gray0clone.cols() / 2, y: gray0clone.rows() / 2)
        let center2 = Point(x: gray0clone.cols(), y: gray0clone.rows())
        let radius = min(gray0clone.cols() / 4, gray0clone.rows() / 4)
        let color128 = Scalar(128)
        let color0 = Scalar(0)

        Imgproc.circle(img: gray0clone, center: center2, radius: radius * 2, color: color128, thickness: 2, lineType: .LINE_4, shift: 1)
        XCTAssertFalse(0 == Core.countNonZero(src: gray0clone))

        Imgproc.circle(img: gray0clone, center: center, radius: radius, color: color0, thickness: 2, lineType: .LINE_4, shift: 0)

        XCTAssert(0 == Core.countNonZero(src: gray0clone))
    }

    func testClipLine() {
        let r = Rect(x: 10, y: 10, width: 10, height: 10)
        var pt1 = Point(x: 5, y: 15)
        var pt2 = Point(x: 25, y: 15)

        XCTAssert(Imgproc.clipLine(imgRect: r, pt1: pt1, pt2: pt2))

        var pt1Clipped = Point(x: 10, y: 15)
        var pt2Clipped = Point(x: 19, y: 15)
        XCTAssertEqual(pt1Clipped, pt1)
        XCTAssertEqual(pt2Clipped, pt2)

        pt1 = Point(x: 5, y: 5)
        pt2 = Point(x: 25, y: 5)
        pt1Clipped = Point(x: 5, y: 5)
        pt2Clipped = Point(x: 25, y: 5)

        XCTAssertFalse(Imgproc.clipLine(imgRect: r, pt1: pt1, pt2: pt2))

        XCTAssertEqual(pt1Clipped, pt1)
        XCTAssertEqual(pt2Clipped, pt2)
    }

    func testEllipse2Poly() {
        let center = Point(x: 4, y: 4)
        let axes = Size(width: 2, height: 2)
        let angle:Int32 = 30
        let arcStart:Int32 = 30
        let arcEnd:Int32 = 60
        let delta:Int32 = 2
        let pts = NSMutableArray()

        Imgproc.ellipse2Poly(center: center, axes: axes, angle: angle, arcStart: arcStart, arcEnd: arcEnd, delta: delta, pts: pts)

        let truth = [Point(x: 5, y: 6), Point(x: 4, y: 6)]
        XCTAssert(truth == pts as! [Point])
    }

    func testEllipseMatPointSizeDoubleDoubleDoubleScalar() {
        let gray0clone = gray0.clone()
        let center = Point(x: gray0clone.cols() / 2, y: gray0clone.rows() / 2)
        let axes = Size(width: 2, height: 2)
        let angle = 30.0, startAngle = 60.0, endAngle = 90.0

        Imgproc.ellipse(img: gray0clone, center: center, axes: axes, angle: angle, startAngle: startAngle, endAngle: endAngle, color: colorWhite)

        XCTAssert(0 != Core.countNonZero(src: gray0clone))
    }

    func testEllipseMatPointSizeDoubleDoubleDoubleScalarInt() {
        let gray0clone = gray0.clone()
        let center = Point(x: gray0clone.cols() / 2, y: gray0clone.rows() / 2)
        let axes = Size(width: 2, height: 2)
        let angle = 30.0, startAngle = 60.0, endAngle = 90.0

        Imgproc.ellipse(img: gray0clone, center: center, axes: axes, angle: angle, startAngle: startAngle, endAngle: endAngle, color: colorWhite, thickness: Core.FILLED)

        XCTAssert(0 != Core.countNonZero(src: gray0clone))
    }

    func testEllipseMatPointSizeDoubleDoubleDoubleScalarIntIntInt() {
        let gray0clone = gray0.clone()
        let center = Point(x: gray0clone.cols() / 2, y: gray0clone.rows() / 2)
        let axes = Size(width: 2, height: 2)
        let center2 = Point(x: gray0clone.cols(), y: gray0clone.rows())
        let axes2 = Size(width: 4, height: 4)
        let angle = 30.0, startAngle = 0.0, endAngle = 30.0

        Imgproc.ellipse(img: gray0clone, center: center, axes: axes, angle: angle, startAngle: startAngle, endAngle: endAngle, color: colorWhite, thickness: Core.FILLED, lineType: .LINE_4, shift: 0)

        XCTAssert(0 != Core.countNonZero(src: gray0clone))

        Imgproc.ellipse(img: gray0clone, center: center2, axes: axes2, angle: angle, startAngle: startAngle, endAngle: endAngle, color: colorBlack, thickness: Core.FILLED, lineType: .LINE_4, shift: 1)

        XCTAssertEqual(0, Core.countNonZero(src: gray0clone))
    }

    func testEllipseMatRotatedRectScalar() throws {
        let matSize:Int32 = 10
        let gray0 = Mat.zeros(matSize, cols: matSize, type: CvType.CV_8U)
        let center = Point2f(x: Float(matSize / 2), y: Float(matSize / 2))
        let size = Size2f(width: Float(matSize / 4), height: Float(matSize / 2))
        let box = RotatedRect(center: center, size: size, angle: 45)

        Imgproc.ellipse(img: gray0, box: box, color: Scalar(1))

        let truth = Mat(rows: matSize, cols: matSize, type: CvType.CV_8U)
        try truth.put(row: 0, col: 0, data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                                         0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
                                         0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                                         0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
                                         0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0] as [Int8])

        try assertMatEqual(truth, gray0)
    }

    func testEllipseMatRotatedRectScalarInt() {
        let gray0clone = gray0.clone()
        let center = Point2f(x: Float(OpenCVTestCase.matSize / 2), y: Float(OpenCVTestCase.matSize / 2))
        let size = Size2f(width: Float(OpenCVTestCase.matSize / 4), height: Float(OpenCVTestCase.matSize / 2))
        let box = RotatedRect(center: center, size: size, angle: 45)

        Imgproc.ellipse(img: gray0clone, box: box, color: Scalar(1), thickness: Core.FILLED)
        Imgproc.ellipse(img: gray0clone, box: box, color: Scalar(0))

        XCTAssert(0 < Core.countNonZero(src: gray0clone))
    }

    func testEllipseMatRotatedRectScalarIntInt() {
        let gray0clone = gray0.clone()
        let center = Point2f(x: Float(OpenCVTestCase.matSize / 2), y: Float(OpenCVTestCase.matSize / 2))
        let size = Size2f(width: 2, height: Float(OpenCVTestCase.matSize * 2 / 3))
        let box = RotatedRect(center: center, size: size, angle: 20)

        Imgproc.ellipse(img: gray0clone, box: box, color: Scalar(9), thickness: 1, lineType: .LINE_AA)
        Imgproc.ellipse(img: gray0clone, box: box, color: Scalar(0), thickness: 1, lineType: .LINE_4)

        XCTAssert(0 < Core.countNonZero(src: gray0clone))
    }

    func testPolylinesMatListOfListOfPointBooleanScalar() {
        let img = gray0.clone()
        let polyline = [[Point(x: 1, y: 1), Point(x: 7, y: 1), Point(x: 7, y: 6), Point(x: 1, y: 6)]]

        Imgproc.polylines(img: img, pts: polyline, isClosed: true, color: Scalar(100))

        XCTAssertEqual(22, Core.countNonZero(src: img))

        Imgproc.polylines(img: img, pts: polyline, isClosed: false, color: Scalar(0))

        XCTAssertEqual(4, Core.countNonZero(src: img))
    }

    func testPolylinesMatListOfListOfPointBooleanScalarInt() {
        let img = gray0.clone()
        let polyline = [[Point(x: 1, y: 1), Point(x: 7, y: 1), Point(x: 7, y: 6), Point(x: 1, y: 6)]]

        Imgproc.polylines(img: img, pts: polyline, isClosed: true, color: Scalar(100), thickness: 2)

        XCTAssertEqual(62, Core.countNonZero(src: img))
    }

    func testPolylinesMatListOfListOfPointBooleanScalarIntIntInt() {
        let img = gray0.clone()
        let polyline1 = [[Point(x: 1, y: 1), Point(x: 7, y: 1), Point(x: 7, y: 6), Point(x: 1, y: 6)]]
        let polyline2 = [[Point(x: 2, y: 2), Point(x: 14, y: 2), Point(x: 14, y: 12), Point(x: 2, y: 12)]]

        Imgproc.polylines(img: img, pts: polyline1, isClosed: true, color: Scalar(100), thickness: 2, lineType: .LINE_8, shift: 0)

        XCTAssert(Core.countNonZero(src: img) > 0)

        Imgproc.polylines(img: img, pts: polyline2, isClosed: true, color: Scalar(0), thickness: 2, lineType: .LINE_8, shift: 1)

        XCTAssertEqual(0, Core.countNonZero(src: img))
    }

    func testPutTextMatStringPointIntDoubleScalar() {
        let text = "Hello World"
        let labelSize = Size(width: 175, height: 22)
        let img = Mat(rows: 20 + labelSize.height, cols: 20 + labelSize.width, type: CvType.CV_8U, scalar: colorBlack)
        let origin = Point(x: 10, y: labelSize.height + 10)

        Imgproc.putText(img: img, text: text, org: origin, fontFace: .FONT_HERSHEY_SIMPLEX, fontScale: 1.0, color: colorWhite)

        XCTAssert(Core.countNonZero(src: img) > 0)
        // check that border is not corrupted
        Imgproc.rectangle(img: img, pt1: Point(x: 11, y: 11), pt2: Point(x: labelSize.width + 10, y: labelSize.height + 10), color: colorBlack, thickness: Core.FILLED)
        XCTAssertEqual(0, Core.countNonZero(src: img))
    }

    func testPutTextMatStringPointIntDoubleScalarInt() {
        let text = "Hello World"
        let labelSize = Size(width: 176, height: 22)
        let img = Mat(rows: 20 + labelSize.height, cols: 20 + labelSize.width, type: CvType.CV_8U, scalar: colorBlack)
        let origin = Point(x: 10, y: labelSize.height + 10)

        Imgproc.putText(img: img, text: text, org: origin, fontFace: .FONT_HERSHEY_SIMPLEX, fontScale: 1.0, color: colorWhite, thickness: 2)

        XCTAssert(Core.countNonZero(src: img) > 0)
        // check that border is not corrupted
        Imgproc.rectangle(img: img, pt1: Point(x: 10, y: 10), pt2: Point(x: labelSize.width + 10 + 1, y: labelSize.height + 10 + 1), color: colorBlack, thickness: Core.FILLED)
        XCTAssertEqual(0, Core.countNonZero(src: img))
    }

    func testPutTextMatStringPointIntDoubleScalarIntIntBoolean() {
        let text = "Hello World"
        let labelSize = Size(width: 175, height: 22)

        let img = Mat(rows: 20 + labelSize.height, cols: 20 + labelSize.width, type: CvType.CV_8U, scalar: colorBlack)
        let origin = Point(x: 10, y: 10)

        Imgproc.putText(img: img, text: text, org: origin, fontFace: .FONT_HERSHEY_SIMPLEX, fontScale: 1.0, color: colorWhite, thickness: 1, lineType: .LINE_8, bottomLeftOrigin: true)

        XCTAssert(Core.countNonZero(src: img) > 0)
        // check that border is not corrupted
        Imgproc.rectangle(img: img, pt1: Point(x: 10, y: 10), pt2: Point(x: labelSize.width + 9, y: labelSize.height + 9), color: colorBlack, thickness: Core.FILLED)
        XCTAssertEqual(0, Core.countNonZero(src: img))
    }

}
