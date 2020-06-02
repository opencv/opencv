//
//  CoreTest.swift
//
//  Created by Giles Payne on 2020/01/27.
//

import XCTest
import OpenCV

class CoreTest: OpenCVTestCase {

    override func setUp() {
        super.setUp()
    }

    override func tearDown() {
        super.tearDown()
    }

    func testAbsdiff() throws {
        Core.absdiff(src1: gray128, src2: gray255, dst: dst)

        try assertMatEqual(gray127, dst)
    }

    func testAddMatMatMat() throws {
        Core.add(src1: gray128, src2: gray128, dst: dst)

        try assertMatEqual(gray255, dst)
    }

    func testAddMatMatMatMatInt() throws {
        Core.add(src1: gray0, src2: gray1, dst: dst, mask: gray1, dtype: CvType.CV_32F)

        XCTAssertEqual(CvType.CV_32F, dst.depth())
        try assertMatEqual(gray1_32f, dst, OpenCVTestCase.EPS)
    }

    func testAddWeightedMatDoubleMatDoubleDoubleMat() throws {
        Core.addWeighted(src1: gray1, alpha: 120.0, src2: gray127, beta: 1.0, gamma: 10.0, dst: dst)

        try assertMatEqual(gray255, dst)
    }

    func testAddWeightedMatDoubleMatDoubleDoubleMatInt() throws {
        Core.addWeighted(src1: gray1, alpha: 126.0, src2: gray127, beta: 1.0, gamma: 2.0, dst: dst, dtype: CvType.CV_32F)

        XCTAssertEqual(CvType.CV_32F, dst.depth())
        try assertMatEqual(gray255_32f, dst, OpenCVTestCase.EPS)
    }

    func testBitwise_andMatMatMat() throws {
        Core.bitwise_and(src1: gray127, src2: gray3, dst: dst)

        try assertMatEqual(gray3, dst)
    }

    func testBitwise_andMatMatMatMat() throws {
        Core.bitwise_and(src1: gray3, src2: gray1, dst: dst, mask: gray255)

        try assertMatEqual(gray1, dst)
    }

    func testBitwise_notMatMat() throws {
        Core.bitwise_not(src: gray255, dst: dst)

        try assertMatEqual(gray0, dst)
    }

    func testBitwise_notMatMatMat() throws {
        Core.bitwise_not(src: gray0, dst: dst, mask: gray1)

        try assertMatEqual(gray255, dst)
    }

    func testBitwise_orMatMatMat() throws {
        Core.bitwise_or(src1: gray1, src2: gray2, dst: dst)

        try assertMatEqual(gray3, dst)
    }

    func testBitwise_orMatMatMatMat() throws {
        Core.bitwise_or(src1: gray127, src2: gray3, dst: dst, mask: gray255)

        try assertMatEqual(gray127, dst)
    }

    func testBitwise_xorMatMatMat() throws {
        Core.bitwise_xor(src1: gray3, src2: gray2, dst: dst)

        try assertMatEqual(gray1, dst)
    }

    func testBitwise_xorMatMatMatMat() throws {
        Core.bitwise_or(src1: gray127, src2: gray128, dst: dst, mask: gray255)

        try assertMatEqual(gray255, dst)
    }

    func testCalcCovarMatrixMatMatMatInt() throws {
        let covar = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_64FC1)
        let mean = Mat(rows: 1, cols: OpenCVTestCase.matSize, type: CvType.CV_64FC1)

        Core.calcCovarMatrix(samples: gray0_32f, covar: covar, mean: mean, flags: CovarFlags.COVAR_ROWS.rawValue | CovarFlags.COVAR_NORMAL.rawValue)

        try assertMatEqual(gray0_64f, covar, OpenCVTestCase.EPS)
        try assertMatEqual(gray0_64f_1d, mean, OpenCVTestCase.EPS)
    }

    func testCalcCovarMatrixMatMatMatIntInt() throws {
        let covar = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)
        let mean = Mat(rows: 1, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)

        Core.calcCovarMatrix(samples: gray0_32f, covar: covar, mean: mean, flags: CovarFlags.COVAR_ROWS.rawValue | CovarFlags.COVAR_NORMAL.rawValue, ctype: CvType.CV_32F)

        try assertMatEqual(gray0_32f, covar, OpenCVTestCase.EPS)
        try assertMatEqual(gray0_32f_1d, mean, OpenCVTestCase.EPS)
    }

    func testCartToPolarMatMatMatMat() throws {
        let x = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [3.0, 6.0, 5, 0] as [Float])
        let y = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try y.put(row: 0, col: 0, data: [4.0, 8.0, 12.0] as [Float])
        let dst_angle = Mat()

        Core.cartToPolar(x: x, y: y, magnitude: dst, angle: dst_angle)

        let expected_magnitude = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try expected_magnitude.put(row: 0, col: 0, data: [5.0, 10.0, 13.0] as [Float])

        let expected_angle = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try expected_angle.put(row: 0, col: 0, data: [atan2rad(4,3), atan2rad(8,6), atan2rad(12,5)])
        try assertMatEqual(expected_magnitude, dst, OpenCVTestCase.EPS)
        try assertMatEqual(expected_angle, dst_angle, OpenCVTestCase.EPS)
    }

    func testCartToPolarMatMatMatMatBoolean() throws {
        let x = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [3.0, 6.0, 5, 0] as [Float])
        let y = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try y.put(row: 0, col: 0, data: [4.0, 8.0, 12.0] as [Float])
        let dst_angle = Mat()

        Core.cartToPolar(x: x, y: y, magnitude: dst, angle: dst_angle, angleInDegrees: true)

        let expected_magnitude = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try expected_magnitude.put(row: 0, col: 0, data: [5.0, 10.0, 13.0] as [Float])
        let expected_angle = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try expected_angle.put(row: 0, col: 0, data:[atan2deg(4,3), atan2deg(8,6), atan2deg(12,5)])
        try assertMatEqual(expected_magnitude, dst, OpenCVTestCase.EPS)
        try assertMatEqual(expected_angle, dst_angle, OpenCVTestCase.EPS * 180/Double.pi)
    }


    func testCheckRangeMat() throws {
        let outOfRange = Mat(rows: 2, cols: 2, type: CvType.CV_64F)
        try outOfRange.put(row: 0, col: 0, data: [Double.nan, -Double.infinity, Double.infinity, 0])

        XCTAssert(Core.checkRange(a: grayRnd_32f))
        XCTAssert(Core.checkRange(a: Mat()))
        XCTAssertFalse(Core.checkRange(a: outOfRange))
    }

    func testCheckRangeMatBooleanPointDoubleDouble() throws {
        let inRange = Mat(rows: 2, cols: 3, type: CvType.CV_64F)
        try inRange.put(row: 0, col: 0, data: [14, 48, 76, 33, 5, 99] as [Double])

        XCTAssert(Core.checkRange(a: inRange, quiet: true, minVal: 5, maxVal: 100))

        let outOfRange = Mat(rows: 2, cols: 3, type: CvType.CV_64F)
        try inRange.put(row: 0, col: 0, data: [-4, 0, 6, 33, 4, 109] as [Double])

        XCTAssertFalse(Core.checkRange(a: outOfRange, quiet: true, minVal: 5, maxVal: 100))
    }

    func testCompare() throws {
        Core.compare(src1: gray0, src2: gray0, dst: dst, cmpop: .CMP_EQ)

        try assertMatEqual(dst, gray255)

        Core.compare(src1: gray0, src2: gray1, dst: dst, cmpop: .CMP_EQ)

        try assertMatEqual(dst, gray0)

        try grayRnd.put(row: 0, col: 0, data: [0, 0] as [Int8])

        Core.compare(src1: gray0, src2: grayRnd, dst: dst, cmpop: .CMP_GE)

        let expected = Int32(grayRnd.total()) - Core.countNonZero(src: grayRnd)
        XCTAssertEqual(expected, Core.countNonZero(src: dst))
    }

    func testCompleteSymmMat() throws {
        Core.completeSymm(m: grayRnd_32f)

        try assertMatEqual(grayRnd_32f, grayRnd_32f.t(), OpenCVTestCase.EPS)
    }

    func testCompleteSymmMatBoolean() throws {
        let grayRnd_32f2 = grayRnd_32f.clone()

        Core.completeSymm(m: grayRnd_32f, lowerToUpper: true)

        try assertMatEqual(grayRnd_32f, grayRnd_32f.t(), OpenCVTestCase.EPS)
        Core.completeSymm(m: grayRnd_32f2, lowerToUpper: false)
        try assertMatNotEqual(grayRnd_32f2, grayRnd_32f, OpenCVTestCase.EPS)
    }

    func testConvertScaleAbsMatMat() throws {
        Core.convertScaleAbs(src: gray0, dst: dst)

        try assertMatEqual(gray0, dst, OpenCVTestCase.EPS)

        Core.convertScaleAbs(src: gray_16u_256, dst: dst)

        try assertMatEqual(gray255, dst, OpenCVTestCase.EPS)
    }

    func testConvertScaleAbsMatMatDoubleDouble() throws {
        Core.convertScaleAbs(src: gray_16u_256, dst: dst, alpha: 2, beta: -513)

        try assertMatEqual(gray1, dst)
    }

    func testCountNonZero() throws {
        XCTAssertEqual(0, Core.countNonZero(src: gray0))
        let gray0copy = gray0.clone()

        try gray0copy.put(row: 0, col: 0, data: [-1] as [Int8])
        try gray0copy.put(row: gray0copy.rows() - 1, col: gray0copy.cols() - 1, data: [-1] as [Int8])

        XCTAssertEqual(2, Core.countNonZero(src: gray0copy))
    }

    func testCubeRoot() {
        let res:Float = Core.cubeRoot(val: -27.0)

        XCTAssertEqual(-3.0, res)
    }

    func testDctMatMat() throws {
        let m = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try m.put(row: 0, col: 0, data: [135.22211, 50.811096, 102.27016, 207.6682] as [Float])
        let dst1 = Mat()
        let dst2 = Mat()

        Core.dct(src: gray0_32f_1d, dst: dst1)
        Core.dct(src: m, dst: dst2)

        truth = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [247.98576, -61.252407, 94.904533, 14.013477] as [Float])
        try assertMatEqual(gray0_32f_1d, dst1, OpenCVTestCase.EPS)
        try assertMatEqual(truth!, dst2, OpenCVTestCase.EPS)
    }

    func testDctMatMatInt() throws {
        let m = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try m.put(row: 0, col: 0, data: [247.98576, -61.252407, 94.904533, 14.013477] as [Float])
        let dst1 = Mat()
        let dst2 = Mat()

        Core.dct(src: gray0_32f_1d, dst: dst1, flags:DftFlags.DCT_INVERSE.rawValue)
        Core.dct(src: m, dst: dst2, flags:DftFlags.DCT_INVERSE.rawValue)

        truth = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [135.22211, 50.811096, 102.27016, 207.6682] as [Float])
        try assertMatEqual(gray0_32f_1d, dst1, OpenCVTestCase.EPS)
        try assertMatEqual(truth!, dst2, OpenCVTestCase.EPS)
    }

    func testDeterminant() throws {
        let mat = Mat(rows: 2, cols: 2, type: CvType.CV_32F)
        try mat.put(row: 0, col: 0, data: [4.0] as [Float])
        try mat.put(row: 0, col: 1, data: [2.0] as [Float])
        try mat.put(row: 1, col: 0, data: [4.0] as [Float])
        try mat.put(row: 1, col: 1, data: [4.0] as [Float])

        let det = Core.determinant(mtx: mat)

        XCTAssertEqual(8.0, det)
    }

    func testDftMatMat() throws {
        Core.dft(src: gray0_32f_1d, dst: dst)

        try assertMatEqual(gray0_32f_1d, dst, OpenCVTestCase.EPS)
    }

    func testDftMatMatIntInt() throws {
        let src1 = Mat(rows: 2, cols: 4, type: CvType.CV_32F)
        try src1.put(row: 0, col: 0, data: [1, 2, 3, 4] as [Float])
        try src1.put(row: 1, col: 0, data: [1, 1, 1, 1] as [Float])
        let src2 = Mat(rows: 2, cols: 4, type: CvType.CV_32F)
        try src2.put(row: 0, col: 0, data: [1, 2, 3, 4] as [Float])
        try src2.put(row: 1, col: 0, data: [0, 0, 0, 0] as [Float])
        let dst1 = Mat()
        let dst2 = Mat()

        Core.dft(src: src1, dst: dst1, flags: DftFlags.DFT_REAL_OUTPUT.rawValue, nonzeroRows: 1)
        Core.dft(src: src2, dst: dst2, flags: DftFlags.DFT_REAL_OUTPUT.rawValue, nonzeroRows: 0)

        try assertMatEqual(dst2, dst1, OpenCVTestCase.EPS)
    }

    func testDivideDoubleMatMat() throws {
        Core.divide(scale: 4.0, src: gray2, dst: dst)

        try assertMatEqual(gray2, dst)

        Core.divide(scale: 4.0, src: gray0, dst: dst)

        try assertMatEqual(gray0, dst)
    }

    func testDivideDoubleMatMatInt() throws {
        Core.divide(scale: 9.0, src: gray3, dst: dst, dtype: CvType.CV_32F)

        try assertMatEqual(gray3_32f, dst, OpenCVTestCase.EPS)
    }

    func testDivideMatMatMat() throws {
        Core.divide(src1: gray9, src2: gray3, dst: dst)

        try assertMatEqual(gray3, dst)
    }

    func testDivideMatMatMatDouble() throws {
        Core.divide(src1: gray1, src2: gray2, dst: dst, scale: 6.0)

        try assertMatEqual(gray3, dst)
    }

    func testDivideMatMatMatDoubleInt() throws {
        Core.divide(src1: gray1, src2: gray2, dst: dst, scale: 6.0, dtype: CvType.CV_32F)

        try assertMatEqual(gray3_32f, dst, OpenCVTestCase.EPS)
    }

    func testEigen() throws {
        let src = Mat(rows: 3, cols: 3, type: CvType.CV_32FC1)
        try src.put(row: 0, col: 0, data: [2, 0, 0] as [Float])
        try src.put(row: 1, col: 0, data: [0, 6, 0] as [Float])
        try src.put(row: 2, col: 0, data: [0, 0, 4] as [Float])
        let eigenVals = Mat()
        let eigenVecs = Mat()

        Core.eigen(src: src, eigenvalues: eigenVals, eigenvectors: eigenVecs)

        let expectedEigenVals = Mat(rows: 3, cols: 1, type: CvType.CV_32FC1)
        try expectedEigenVals.put(row: 0, col: 0, data: [6, 4, 2] as [Float])
        try assertMatEqual(eigenVals, expectedEigenVals, OpenCVTestCase.EPS)

        // check by definition
        let EPS = 1e-3
        for i:Int32 in 0..<3 {
            let vec = eigenVecs.row(i).t()
            let lhs = Mat(rows: 3, cols: 1, type: CvType.CV_32FC1)
            Core.gemm(src1: src, src2: vec, alpha: 1.0, src3: Mat(), beta: 1.0, dst: lhs)
            let rhs = Mat(rows: 3, cols: 1, type: CvType.CV_32FC1)
            Core.gemm(src1: vec, src2: eigenVals.row(i), alpha: 1.0, src3: Mat(), beta: 1.0, dst: rhs)
            try assertMatEqual(lhs, rhs, EPS)
        }
    }

    func testExp() throws {
        Core.exp(src: gray0_32f, dst: dst)

        try assertMatEqual(gray1_32f, dst, OpenCVTestCase.EPS)
    }

    func testExtractChannel() throws {
        Core.extractChannel(src: rgba128, dst: dst, coi: 0)

        try assertMatEqual(gray128, dst)
    }

    func testFastAtan2() {
        let EPS: Float = 0.3

        let res = Core.fastAtan2(y: 50, x: 50)

        XCTAssertEqual(Float(45.0), res, accuracy:EPS)

        let res2 = Core.fastAtan2(y: 80, x: 20)

        XCTAssertEqual(atan2(80, 20) * 180 / Float.pi, res2, accuracy:EPS)
    }

    func testFillConvexPolyMatListOfPointScalar() {
        let polyline = [Point(x: 1, y: 1), Point(x: 5, y: 0), Point(x: 6, y: 8), Point(x: 0, y: 9)]
        dst = gray0.clone()

        Imgproc.fillConvexPoly(img: dst, points: polyline, color: Scalar(150))

        XCTAssert(0 < Core.countNonZero(src: dst))
        XCTAssert(dst.total() > Core.countNonZero(src: dst))
    }

    func testFillConvexPolyMatListOfPointScalarIntInt() {
        let polyline1 = [Point(x: 2, y: 1), Point(x: 5, y: 1), Point(x: 5, y: 7), Point(x: 2, y: 7)]
        let polyline2 = [Point(x: 4, y: 2), Point(x: 10, y: 2), Point(x: 10, y: 14), Point(x: 4, y: 14)]

        // current implementation of fixed-point version of fillConvexPoly
        // requires image to be at least 2-pixel wider in each direction than
        // contour
        Imgproc.fillConvexPoly(img: gray0, points: polyline1, color: colorWhite, lineType: .LINE_8, shift: 0)

        XCTAssert(0 < Core.countNonZero(src: gray0))
        XCTAssert(gray0.total() > Core.countNonZero(src: gray0))

        Imgproc.fillConvexPoly(img: gray0, points: polyline2, color: colorBlack, lineType: .LINE_8, shift: 1)

        XCTAssertEqual(0, Core.countNonZero(src: gray0))
    }

    func testFillPolyMatListOfListOfPointScalar() throws {
        let matSize = 10;
        let gray0 = Mat.zeros(Int32(matSize), cols: Int32(matSize), type: CvType.CV_8U)
        let polyline = [Point(x: 1, y: 4), Point(x: 1, y: 8), Point(x: 4, y: 1), Point(x: 7, y: 8), Point(x: 7, y: 4)]
        let polylines = [polyline]

        Imgproc.fillPoly(img: gray0, pts: polylines, color: Scalar(1))

        let truth:[Int8] =
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
              0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
              0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
              0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
              0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
              0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        let truthMat = Mat(size: gray0.size(), type: CvType.CV_8U)
        try truthMat.put(row:0, col:0, data:truth)

        try assertMatEqual(truthMat, gray0)
    }

    func testFillPolyMatListOfListOfPointScalarIntIntPoint() {
        let polyline1 = [Point(x: 1, y: 4), Point(x: 1, y: 8), Point(x: 4, y: 1), Point(x: 7, y: 8), Point(x: 7, y: 4)]
        let polyline2 = [Point(x: 0, y: 3), Point(x: 0, y: 7), Point(x: 3, y: 0), Point(x: 6, y: 7), Point(x: 6, y: 3)]

        let polylines1 = [polyline1]
        let polylines2 = [polyline2]

        Imgproc.fillPoly(img: gray0, pts: polylines1, color: Scalar(1), lineType: .LINE_8, shift: 0, offset: Point(x: 0, y: 0))

        XCTAssert(0 < Core.countNonZero(src: gray0))

        Imgproc.fillPoly(img: gray0, pts: polylines2, color: Scalar(0), lineType: .LINE_8, shift: 0, offset: Point(x: 1, y: 1))

        XCTAssertEqual(0, Core.countNonZero(src: gray0))
    }

    func testFlip() throws {
        let src = Mat(rows: 2, cols: 2, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [1.0] as [Float])
        try src.put(row: 0, col: 1, data: [2.0] as [Float])
        try src.put(row: 1, col: 0, data: [3.0] as [Float])
        try src.put(row: 1, col: 1, data: [4.0] as [Float])
        let dst1 = Mat()
        let dst2 = Mat()

        Core.flip(src: src, dst: dst1, flipCode: 0)
        Core.flip(src: src, dst: dst2, flipCode: 1)

        let dst_f1 = Mat(rows: 2, cols: 2, type: CvType.CV_32F)
        try dst_f1.put(row: 0, col: 0, data: [3.0] as [Float])
        try dst_f1.put(row: 0, col: 1, data: [4.0] as [Float])
        try dst_f1.put(row: 1, col: 0, data: [1.0] as [Float])
        try dst_f1.put(row: 1, col: 1, data: [2.0] as [Float])
        let dst_f2 = Mat(rows: 2, cols: 2, type: CvType.CV_32F)
        try dst_f2.put(row: 0, col: 0, data: [2.0] as [Float])
        try dst_f2.put(row: 0, col: 1, data: [1.0] as [Float])
        try dst_f2.put(row: 1, col: 0, data: [4.0] as [Float])
        try dst_f2.put(row: 1, col: 1, data: [3.0] as [Float])
        try assertMatEqual(dst_f1, dst1, OpenCVTestCase.EPS)
        try assertMatEqual(dst_f2, dst2, OpenCVTestCase.EPS)
    }

    func testGemmMatMatDoubleMatDoubleMat() throws {
        let m1 = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1)
        try m1.put(row: 0, col: 0, data: [1.0, 0.0] as [Float])
        try m1.put(row: 1, col: 0, data: [1.0, 0.0] as [Float])
        let m2 = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1)
        try m2.put(row: 0, col: 0, data: [1.0, 0.0] as [Float])
        try m2.put(row: 1, col: 0, data: [1.0, 0.0] as [Float])
        let dmatrix = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1);
        try dmatrix.put(row: 0, col: 0, data: [0.001, 0.001] as [Float])
        try dmatrix.put(row: 1, col: 0, data: [0.001, 0.001] as [Float])

        Core.gemm(src1: m1, src2: m2, alpha: 1.0, src3: dmatrix, beta: 1.0, dst: dst)

        let expected = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1)
        try expected.put(row: 0, col: 0, data: [1.001, 0.001] as [Float])
        try expected.put(row: 1, col: 0, data: [1.001, 0.001] as [Float])
        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testGemmMatMatDoubleMatDoubleMatInt() throws {
        let m1 = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1)
        try m1.put(row: 0, col: 0, data: [1.0, 0.0])
        try m1.put(row: 1, col: 0, data: [1.0, 0.0])
        let m2 = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1)
        try m2.put(row: 0, col: 0, data: [1.0, 0.0])
        try m2.put(row: 1, col: 0, data: [1.0, 0.0])
        let dmatrix = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1)
        try dmatrix.put(row: 0, col: 0, data: [0.001, 0.001])
        try dmatrix.put(row: 1, col: 0, data: [0.001, 0.001])

        Core.gemm(src1: m1, src2: m2, alpha: 1.0, src3: dmatrix, beta: 1.0, dst: dst, flags: GemmFlags.GEMM_1_T.rawValue)

        let expected = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1)
        try expected.put(row: 0, col: 0, data: [2.001, 0.001])
        try expected.put(row: 1, col: 0, data: [0.001, 0.001])
        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testGetCPUTickCount() {
        let cpuCountStart = Core.getCPUTickCount()
        Core.sum(src: gray255)
        let actualTickCount = Core.getCPUTickCount()

        let expectedTickCount = actualTickCount - cpuCountStart;
        XCTAssert(expectedTickCount > 0)
    }

    func testGetNumberOfCPUs() {
        let cpus = Core.getNumberOfCPUs()

        XCTAssert(ProcessInfo().processorCount <= cpus)
    }

    func testGetOptimalDFTSize() {
        XCTAssertEqual(1, Core.getOptimalDFTSize(vecsize: 0))
        XCTAssertEqual(135, Core.getOptimalDFTSize(vecsize: 133))
        XCTAssertEqual(15, Core.getOptimalDFTSize(vecsize: 13))
    }

    func testGetTickCount() {

        let startCount = Core.getTickCount()
        Core.divide(src1: gray2, src2: gray1, dst: dst)
        let endCount = Core.getTickCount()

        let count = endCount - startCount;
        XCTAssert(count > 0)
    }

    func testGetTickFrequency() {
        let freq1 = Core.getTickFrequency()
        Core.divide(src1: gray2, src2: gray1, dst: dst)
        let freq2 = Core.getTickFrequency()

        XCTAssert(0 < freq1)
        XCTAssertEqual(freq1, freq2)
    }

    func testHconcat() throws {
        let mats = [Mat.eye(rows: 3, cols: 3, type: CvType.CV_8U), Mat.zeros(3, cols: 2, type: CvType.CV_8U)]

        Core.hconcat(src: mats, dst: dst)

        try assertMatEqual(Mat.eye(rows: 3, cols: 5, type: CvType.CV_8U), dst)
    }

    func testIdctMatMat() throws {
        let mat = Mat(rows: 1, cols: 8, type: CvType.CV_32F)
        try mat.put(row: 0, col: 0, data: [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0])

        Core.idct(src: mat, dst: dst)

        truth = Mat(rows: 1, cols: 8, type: CvType.CV_32F)

        try truth!.put(row: 0, col: 0, data: [3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testIdctMatMatInt() throws {
        let mat = Mat(rows: 2, cols: 8, type: CvType.CV_32F)
        try mat.put(row: 0, col: 0, data: [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0])
        try mat.put(row: 1, col: 0, data: [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0])

        Core.idct(src: mat, dst: dst, flags: DftFlags.DCT_ROWS.rawValue)

        truth = Mat(rows: 2, cols: 8, type: CvType.CV_32F)

        try truth!.put(row: 0, col: 0, data: [3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115])
        try truth!.put(row: 1, col: 0, data: [3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testIdftMatMat() throws {
        let mat = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try mat.put(row: 0, col: 0, data: [1.0, 2.0, 3.0, 4.0])

        Core.idft(src: mat, dst: dst)

        truth = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [9, -9, 1, 3] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testIdftMatMatIntInt() throws {
        let mat = Mat(rows: 2, cols: 4, type: CvType.CV_32F)
        try mat.put(row: 0, col: 0, data: [1.0, 2.0, 3.0, 4.0] as [Float])
        try mat.put(row: 1, col: 0, data: [1.0, 2.0, 3.0, 4.0] as [Float])
        let dst = Mat()

        Core.idft(src: mat, dst: dst, flags: DftFlags.DFT_REAL_OUTPUT.rawValue, nonzeroRows: 1)

        truth = Mat(rows: 2, cols: 4, type: CvType.CV_32F)
        try truth!.put(row: 0, col: 0, data: [18, -18, 2, 6] as [Float])
        try truth!.put(row: 1, col: 0, data: [0, 0, 0, 0] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testInRange() throws {
        let gray0copy = gray0.clone()
        try gray0copy.put(row: 1, col: 1, data: [100, -105, -55] as [Int8])

        Core.inRange(src: gray0copy, lowerb: Scalar(120), upperb: Scalar(160), dst: dst)

        var vals = [Int8](repeating: 0, count: 3)
        try dst.get(row: 1, col: 1, data: &vals)

        XCTAssertEqual(0, vals[0])
        XCTAssertEqual(-1, vals[1])
        XCTAssertEqual(0, vals[2])
        XCTAssertEqual(1, Core.countNonZero(src: dst))
    }

    func testInsertChannel() throws {
        dst = rgba128.clone()
        Core.insertChannel(src: gray0, dst: dst, coi: 0)
        Core.insertChannel(src: gray0, dst: dst, coi: 1)
        Core.insertChannel(src: gray0, dst: dst, coi: 2)
        Core.insertChannel(src: gray0, dst: dst, coi: 3)

        try assertMatEqual(rgba0, dst)
    }

    func testInvertMatMat() throws {
        let src = Mat(rows: 2, cols: 2, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [1.0] as [Float])
        try src.put(row: 0, col: 1, data: [2.0] as [Float])
        try src.put(row: 1, col: 0, data: [1.5] as [Float])
        try src.put(row: 1, col: 1, data: [4.0] as [Float])

        Core.invert(src: src, dst: dst)

        truth = Mat(rows: 2, cols: 2, type: CvType.CV_32F)

        try truth!.put(row: 0, col: 0, data: [4.0] as [Float])
        try truth!.put(row: 0, col: 1, data: [-2.0] as [Float])
        try truth!.put(row: 1, col: 0, data: [-1.5] as [Float])
        try truth!.put(row: 1, col: 1, data: [1.0] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testInvertMatMatInt() throws {
        let src = Mat.eye(rows: 3, cols: 3, type: CvType.CV_32FC1)
        try src.put(row: 0, col: 2, data: [1] as [Float])

        let cond = Core.invert(src: src, dst: dst, flags: DecompTypes.DECOMP_SVD.rawValue)

        truth = Mat.eye(rows: 3, cols: 3, type: CvType.CV_32FC1)
        try truth!.put(row: 0, col: 2, data: [-1] as [Float])
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
        XCTAssertEqual(0.3819660544395447, cond, accuracy:OpenCVTestCase.EPS)
    }

    func testKmeansMatIntMatTermCriteriaIntInt() throws {
        let data = Mat(rows: 4, cols: 5, type: CvType.CV_32FC1)
        try data.put(row: 0, col: 0, data: [1, 2, 3, 4, 5] as [Float])
        try data.put(row: 1, col: 0, data: [2, 3, 4, 5, 6] as [Float])
        try data.put(row: 2, col: 0, data: [5, 4, 3, 2, 1] as [Float])
        try data.put(row: 3, col: 0, data: [6, 5, 4, 3, 2] as [Float])
        let criteria = TermCriteria(type: TermCriteria.eps, maxCount: 0, epsilon: OpenCVTestCase.EPS)
        let labels = Mat()

        Core.kmeans(data: data, K: 2, bestLabels: labels, criteria: criteria, attempts: 1, flags: KmeansFlags.KMEANS_PP_CENTERS.rawValue)

        var first_center = [Int32](repeating: 0, count: 1)
        try labels.get(row: 0, col: 0, data: &first_center)
        let c1 = first_center[0]
        let expected_labels = Mat(rows: 4, cols: 1, type: CvType.CV_32S)
        try expected_labels.put(row: 0, col: 0, data: [c1, c1, 1 - c1, 1 - c1])
        try assertMatEqual(expected_labels, labels)
    }

    func testKmeansMatIntMatTermCriteriaIntIntMat() throws {
        let data = Mat(rows: 4, cols: 5, type: CvType.CV_32FC1)
        try data.put(row: 0, col: 0, data: [1, 2, 3, 4, 5] as [Float])
        try data.put(row: 1, col: 0, data: [2, 3, 4, 5, 6] as [Float])
        try data.put(row: 2, col: 0, data: [5, 4, 3, 2, 1] as [Float])
        try data.put(row: 3, col: 0, data: [6, 5, 4, 3, 2] as [Float])
        let criteria = TermCriteria(type:TermCriteria.eps, maxCount: 0, epsilon: OpenCVTestCase.EPS)
        let labels = Mat()
        let centers = Mat()

        Core.kmeans(data: data, K: 2, bestLabels: labels, criteria: criteria, attempts: 6, flags: KmeansFlags.KMEANS_RANDOM_CENTERS.rawValue, centers: centers)

        var first_center = [Int32](repeating: 0, count: 1)
        try labels.get(row: 0, col: 0, data: &first_center)
        let c1 = first_center[0]
        let expected_labels = Mat(rows: 4, cols: 1, type: CvType.CV_32S)
        try expected_labels.put(row: 0, col: 0, data: [c1, c1, 1 - c1, 1 - c1])
        let expected_centers = Mat(rows: 2, cols: 5, type: CvType.CV_32FC1)
        try expected_centers.put(row: c1, col: 0, data: [1.5, 2.5, 3.5, 4.5, 5.5] as [Float])
        try expected_centers.put(row: 1 - c1, col: 0, data: [5.5, 4.5, 3.5, 2.5, 1.5] as [Float])
        try assertMatEqual(expected_labels, labels)
        try assertMatEqual(expected_centers, centers, OpenCVTestCase.EPS)
    }

    func testLineMatPointPointScalar() {
        let nPoints = min(gray0.cols(), gray0.rows())
        let point1 = Point(x: 0, y: 0)
        let point2 = Point(x: nPoints, y: nPoints)
        let color = Scalar(255)

        Imgproc.line(img: gray0, pt1: point1, pt2: point2, color: color)

        XCTAssert(nPoints == Core.countNonZero(src: gray0))
    }

    func testLineMatPointPointScalarInt() {
        let nPoints = min(gray0.cols(), gray0.rows())
        let point1 = Point(x: 0, y: 0)
        let point2 = Point(x: nPoints, y: nPoints)

        Imgproc.line(img: gray0, pt1: point1, pt2: point2, color: colorWhite, thickness: 1)

        XCTAssert(nPoints == Core.countNonZero(src: gray0))
    }

    func testLineMatPointPointScalarIntIntInt() {
        let nPoints = min(gray0.cols(), gray0.rows())
        let point1 = Point(x: 3, y: 4)
        let point2 = Point(x: nPoints, y: nPoints)
        let point1_4 = Point(x: 3 * 4, y: 4 * 4)
        let point2_4 = Point(x: nPoints * 4, y: nPoints * 4)

        Imgproc.line(img: gray0, pt1: point2, pt2: point1, color: colorWhite, thickness: 2, lineType: .LINE_8, shift: 0)

        XCTAssertFalse(0 == Core.countNonZero(src: gray0))

        Imgproc.line(img: gray0, pt1: point2_4, pt2: point1_4, color: colorBlack, thickness: 2, lineType: .LINE_8, shift: 2)

        XCTAssertEqual(0, Core.countNonZero(src: gray0))
    }

    func testLog() throws {
        let mat = Mat(rows: 1, cols: 4, type: CvType.CV_32FC1)
        try mat.put(row: 0, col: 0, data: [1.0, 10.0, 100.0, 1000.0])

        Core.log(src: mat, dst: dst)

        let expected = Mat(rows: 1, cols: 4, type: CvType.CV_32FC1)
        try expected.put(row: 0, col: 0, data: [0, 2.3025851, 4.6051702, 6.9077554])
        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testLUTMatMatMat() throws {
        let lut = Mat(rows: 1, cols: 256, type: CvType.CV_8UC1)
        lut.setTo(scalar: Scalar(0))

        Core.LUT(src: grayRnd, lut: lut, dst: dst)

        try assertMatEqual(gray0, dst)

        lut.setTo(scalar: Scalar(255))

        Core.LUT(src: grayRnd, lut: lut, dst: dst)

        try assertMatEqual(gray255, dst)
    }

    func testMagnitude() throws {
        let x = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        let y = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [3.0, 5.0, 9.0, 6.0])
        try y.put(row: 0, col: 0, data: [4.0, 12.0, 40.0, 8.0])

        Core.magnitude(x: x, y: y, magnitude: dst)

        let out = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try out.put(row: 0, col: 0, data: [5.0, 13.0, 41.0, 10.0])
        try assertMatEqual(out, dst, OpenCVTestCase.EPS)

        Core.magnitude(x: gray0_32f, y: gray255_32f, magnitude: dst)

        try assertMatEqual(gray255_32f, dst, OpenCVTestCase.EPS)
    }

    func testMahalanobis() {
        Core.setRNGSeed(seed: 45)
        let src = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)
        Core.randu(dst: src, low: -128, high: 128)

        var covar = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)
        let mean = Mat(rows: 1, cols: OpenCVTestCase.matSize, type: CvType.CV_32F)
        Core.calcCovarMatrix(samples: src, covar: covar, mean: mean, flags: CovarFlags.COVAR_ROWS.rawValue | CovarFlags.COVAR_NORMAL.rawValue, ctype: CvType.CV_32F)
        covar = covar.inv()

        let line1 = src.row(0)
        let line2 = src.row(1)

        var d = Core.Mahalanobis(v1: line1, v2: line1, icovar: covar)

        XCTAssertEqual(0.0, d)

        d = Core.Mahalanobis(v1: line1, v2: line2, icovar: covar)

        XCTAssert(d > 0.0)
    }

    func testMax() throws {
        Core.max(src1: gray0, src2: gray255, dst: dst)

        try assertMatEqual(gray255, dst)

        let x = Mat(rows: 1, cols: 1, type: CvType.CV_32F)
        let y = Mat(rows: 1, cols: 1, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [23.0])
        try y.put(row: 0, col: 0, data: [4.0])

        Core.max(src1: x, src2: y, dst: dst)

        let truth = Mat(rows: 1, cols: 1, type: CvType.CV_32F)
        try truth.put(row: 0, col: 0, data: [23.0])
        try assertMatEqual(truth, dst, OpenCVTestCase.EPS)
    }

    func testMeanMat() {
        let mean = Core.mean(src: makeMask(gray128))

        assertScalarEqual(Scalar(64), mean, OpenCVTestCase.EPS)
    }

    func testMeanMatMat() {
        let mask1 = makeMask(gray1.clone())
        let mask2 = makeMask(gray0, vals: [1])

        let mean1 = Core.mean(src: grayRnd, mask: mask1)
        let mean2 = Core.mean(src: grayRnd, mask: mask2)
        let mean = Core.mean(src: grayRnd, mask: gray1)

        assertScalarEqual(mean, Scalar(0.5 * (mean1.val[0].doubleValue + mean2.val[0].doubleValue)), OpenCVTestCase.EPS)
    }

    func testMeanStdDevMatMatMat() {
        let mean = DoubleVector()
        let stddev = DoubleVector()
        Core.meanStdDev(src: rgbLena, mean: mean, stddev: stddev)

        let expectedMean = [105.3989906311035, 99.56269836425781, 179.7303047180176]
        let expectedDev = [33.74205485167219, 52.8734582803278, 49.01569488056406]

        assertArrayEquals(expectedMean as [NSNumber], mean.array as [NSNumber], OpenCVTestCase.EPS)
        assertArrayEquals(expectedDev as [NSNumber], stddev.array as [NSNumber], OpenCVTestCase.EPS)
    }

    func testMeanStdDevMatMatMatMat() {
        var submat = grayRnd.submat(rowStart: 0, rowEnd: grayRnd.rows() / 2, colStart: 0, colEnd: grayRnd.cols() / 2)
        submat.setTo(scalar: Scalar(33))
        let mask = gray0.clone()
        submat = mask.submat(rowStart: 0, rowEnd: mask.rows() / 2, colStart: 0, colEnd: mask.cols() / 2)
        submat.setTo(scalar: Scalar(1))
        let mean = DoubleVector()
        let stddev = DoubleVector()

        Core.meanStdDev(src: grayRnd, mean: mean, stddev: stddev, mask: mask)

        let expectedMean = [33]
        let expectedDev = [0]

        assertArrayEquals(expectedMean as [NSNumber], mean.array as [NSNumber], OpenCVTestCase.EPS)
        assertArrayEquals(expectedDev as [NSNumber], stddev.array as [NSNumber], OpenCVTestCase.EPS)
    }

    func testMerge() throws {
        let src1 = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(1))
        let src2 = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(2))
        let src3 = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(3))
        let srcArray = [src1, src2, src3]

        Core.merge(mv: srcArray, dst: dst)

        truth = Mat(rows: 2, cols: 2, type: CvType.CV_32FC3, scalar: Scalar(1, 2, 3))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testMin() throws {
        Core.min(src1: gray0, src2: gray255, dst: dst)

        try assertMatEqual(gray0, dst)
    }

    func testMinMaxLocMat() throws {
        let minVal:Double = 1
        let maxVal:Double = 10
        let minLoc = Point(x: gray3.cols() / 4, y: gray3.rows() / 2)
        let maxLoc = Point(x: gray3.cols() / 2, y: gray3.rows() / 4)
        let gray3copy = gray3.clone()
        try gray3copy.put(row: minLoc.y, col: minLoc.x, data: [minVal])
        try gray3copy.put(row: maxLoc.y, col: maxLoc.x, data: [maxVal])

        let mmres = Core.minMaxLoc(gray3copy)

        XCTAssertEqual(minVal, mmres.minVal)
        XCTAssertEqual(maxVal, mmres.maxVal)
        assertPointEquals(minLoc, mmres.minLoc)
        assertPointEquals(maxLoc, mmres.maxLoc)
    }

    func testMinMaxLocMatMat() throws {
        let src = Mat(rows: 4, cols: 4, type: CvType.CV_8U)
        try src.put(row: 0, col: 0, data: [2, 4, 27, 3] as [Int8])
        try src.put(row: 1, col: 0, data: [0, 8, 7, -126] as [Int8])
        try src.put(row: 2, col: 0, data: [13, 4, 13, 4] as [Int8])
        try src.put(row: 3, col: 0, data: [6, 4, 2, 13] as [Int8])
        let mask = Mat(rows: 4, cols: 4, type: CvType.CV_8U, scalar: Scalar(0))
        mask.submat(rowStart: 1, rowEnd: 3, colStart: 1, colEnd: 4).setTo(scalar: Scalar(1))

        let res = Core.minMaxLoc(src, mask: mask)

        XCTAssertEqual(4.0, res.minVal)
        XCTAssertEqual(130.0, res.maxVal)
        assertPointEquals(Point(x: 1, y: 2), res.minLoc)
        assertPointEquals(Point(x: 3, y: 1), res.maxLoc)
    }

    func testMixChannels() throws {
        let rgba0Copy = rgba0.clone()
        rgba0Copy.setTo(scalar: Scalar(10, 20, 30, 40))
        let src = [rgba0Copy]
        let dst = [gray3, gray2, gray1, gray0, getMat(CvType.CV_8UC3, vals: [0, 0, 0])]
        let fromTo = IntVector([
                3, 0,
                3, 1,
                2, 2,
                0, 3,
                2, 4,
                1, 5,
                0, 6])

        Core.mixChannels(src: src, dst: dst, fromTo: fromTo)

        try assertMatEqual(getMat(CvType.CV_8U, vals: [40]), dst[0])
        try assertMatEqual(getMat(CvType.CV_8U, vals: [40]), dst[1])
        try assertMatEqual(getMat(CvType.CV_8U, vals: [30]), dst[2])
        try assertMatEqual(getMat(CvType.CV_8U, vals: [10]), dst[3])
        try assertMatEqual(getMat(CvType.CV_8UC3, vals: [30, 20, 10]), dst[4])
    }

    func testMulSpectrumsMatMatMatInt() throws {
        let src1 = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try src1.put(row: 0, col: 0, data: [1.0, 2.0, 3.0, 4.0])

        let src2 = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try src2.put(row: 0, col: 0, data: [1.0, 2.0, 3.0, 4.0])

        Core.mulSpectrums(a: src1, b: src2, c: dst, flags: DftFlags.DFT_ROWS.rawValue)

        let expected = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try expected.put(row: 0, col: 0, data: [1, -5, 12, 16] as [Float])

        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testMulSpectrumsMatMatMatIntBoolean() throws {
        let src1 = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try src1.put(row: 0, col: 0, data: [1.0, 2.0, 3.0, 4.0])
        let src2 = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try src2.put(row: 0, col: 0, data: [1.0, 2.0, 3.0, 4.0])

        Core.mulSpectrums(a: src1, b: src2, c: dst, flags: DftFlags.DFT_ROWS.rawValue, conjB: true)

        let expected = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try expected.put(row: 0, col: 0, data: [1, 13, 0, 16] as [Float])
        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testMultiplyMatMatMat() throws {
        Core.multiply(src1: gray0, src2: gray255, dst: dst)

        try assertMatEqual(gray0, dst)
    }

    func testMultiplyMatMatMatDouble() throws {
        Core.multiply(src1: gray1, src2: gray1, dst: dst, scale: 2.0)

        try assertMatEqual(gray2, dst)

    }

    func testMultiplyMatMatMatDoubleInt() throws {
        Core.multiply(src1: gray1, src2: gray2, dst: dst, scale: 1.5, dtype: CvType.CV_32F)

        try assertMatEqual(gray3_32f, dst, OpenCVTestCase.EPS)
    }

    func testMulTransposedMatMatBoolean() throws {
        Core.mulTransposed(src: grayE_32f, dst: dst, aTa: true)

        try assertMatEqual(grayE_32f, dst, OpenCVTestCase.EPS)
    }

    func testMulTransposedMatMatBooleanMatDouble() throws {
        Core.mulTransposed(src: grayE_32f, dst: dst, aTa: true, delta: gray0_32f, scale: 2)

        truth = gray0_32f;
        truth!.diag().setTo(scalar: Scalar(2))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testMulTransposedMatMatBooleanMatDoubleInt() throws {
        let a = getMat(CvType.CV_32F, vals: [1])

        Core.mulTransposed(src: a, dst: dst, aTa: true, delta: gray0_32f, scale: 3, dtype: CvType.CV_64F)

        try assertMatEqual(getMat(CvType.CV_64F, vals: [3 * a.rows()] as [NSNumber]), dst, OpenCVTestCase.EPS)
    }

    func testNormalizeMatMat() throws {
        let m = gray0.clone()
        m.diag().setTo(scalar: Scalar(2))

        Core.normalize(src: m, dst: dst)

        try assertMatEqual(gray0, dst)
    }

    func testNormalizeMatMatDoubleDoubleInt() throws {
        let src = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [1.0, 2.0, 3.0, 4.0])

        Core.normalize(src: src, dst: dst, alpha: 1.0, beta: 2.0, norm_type: .NORM_INF)

        let expected = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try expected.put(row: 0, col: 0, data: [0.25, 0.5, 0.75, 1])
        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testNormalizeMatMatDoubleDoubleIntInt() throws {
        let src = Mat(rows: 1, cols: 5, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [0, 1, 2, 3, 4] as [Float])

        Core.normalize(src: src, dst: dst, alpha: 1, beta: 2, norm_type: .NORM_MINMAX, dtype: CvType.CV_64F)

        let expected = Mat(rows: 1, cols: 5, type: CvType.CV_64F)
        try expected.put(row: 0, col: 0, data: [1, 1.25, 1.5, 1.75, 2])
        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testNormalizeMatMatDoubleDoubleIntIntMat() throws {
        let src = Mat(rows: 1, cols: 5, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [0, 1, 2, 3, 4] as [Float])
        let mask = Mat(rows: 1, cols: 5, type: CvType.CV_8U)
        try mask.put(row: 0, col: 0, data: [1, 0, 0, 0, 1] as [Int8])
        dst = src.clone()

        Core.normalize(src: src, dst: dst, alpha: 1, beta: 2, norm_type: .NORM_MINMAX, dtype: CvType.CV_32F, mask: mask)

        let expected = Mat(rows: 1, cols: 5, type: CvType.CV_32F)
        try expected.put(row: 0, col: 0, data: [1, 1, 2, 3, 2] as [Float])
        try assertMatEqual(expected, dst, OpenCVTestCase.EPS)
    }

    func testNormMat() throws {
        let n = Core.norm(src1: gray1)

        XCTAssertEqual(10, n)
    }

    func testNormMatInt() throws {
        let n = Core.norm(src1: gray127, normType: .NORM_INF)

        XCTAssertEqual(127, n)
    }

    func testNormMatIntMat() throws {
        let n = Core.norm(src1: gray3, normType: .NORM_L1, mask: gray0)

        XCTAssertEqual(0.0, n)
    }

    func testNormMatMat() throws {
        let n = Core.norm(src1: gray0, src2: gray1)

        XCTAssertEqual(10.0, n)
    }

    func testNormMatMatInt() throws {
        let n = Core.norm(src1: gray127, src2: gray1, normType: .NORM_INF)

        XCTAssertEqual(126.0, n)
    }

    func testNormMatMatIntMat() throws {
        let n = Core.norm(src1: gray3, src2: gray0, normType: .NORM_L1, mask: makeMask(gray0.clone(), vals: [1]))

        XCTAssertEqual(150.0, n)
    }

    func testPCABackProject() throws {
        let mean = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try mean.put(row: 0, col: 0, data: [2, 4, 4, 8] as [Float])
        let vectors = Mat(rows: 1, cols: 4, type: CvType.CV_32F, scalar: Scalar(0))
        try vectors.put(row: 0, col: 0, data: [0.2, 0.4, 0.4, 0.8])
        let data = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try data.put(row: 0, col: 0, data: [-5, 0, -10] as [Float])
        let result = Mat()

        Core.PCABackProject(data: data, mean: mean, eigenvectors: vectors, result: result)

        let truth = Mat(rows: 3, cols: 4, type: CvType.CV_32F)
        try truth.put(row: 0, col: 0, data: [1, 2, 2, 4] as [Float])
        try truth.put(row: 1, col: 0, data: [2, 4, 4, 8] as [Float])
        try truth.put(row: 2, col: 0, data: [0, 0, 0, 0] as [Float])
        try assertMatEqual(truth, result, OpenCVTestCase.EPS)
    }

    func testPCAComputeMatMatMat() throws {
        let data = Mat(rows: 3, cols: 4, type: CvType.CV_32F)
        try data.put(row: 0, col: 0, data: [1, 2, 2, 4] as [Float])
        try data.put(row: 1, col: 0, data: [2, 4, 4, 8] as [Float])
        try data.put(row: 2, col: 0, data: [3, 6, 6, 12] as [Float])
        let mean = Mat()
        let vectors = Mat()

        Core.PCACompute(data: data, mean: mean, eigenvectors: vectors)
        let mean_truth = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try mean_truth.put(row: 0, col: 0, data: [2, 4, 4, 8] as [Float])
        let vectors_truth = Mat(rows: 3, cols: 4, type: CvType.CV_32F, scalar: Scalar(0))
        try vectors_truth.put(row: 0, col: 0, data: [0.2, 0.4, 0.4, 0.8] as [Float])
        try assertMatEqual(mean_truth, mean, OpenCVTestCase.EPS)

        // eigenvectors are normalized (length = 1),
        // but direction is unknown (v and -v are both eigen vectors)
        // so this direct check doesn't work:
        // try assertMatEqual(vectors_truth, vectors, OpenCVTestCase.EPS)
        for i in 0..<1 {
            let vec0 = vectors_truth.row(Int32(i))
            let vec1 = vectors.row(Int32(i))
            let vec1_ = Mat()
            Core.subtract(src1: Mat(rows: 1, cols: 4, type: CvType.CV_32F, scalar: Scalar(0)), src2: vec1, dst: vec1_)
            let scale1 = Core.norm(src1: vec0, src2: vec1)
            let scale2 = Core.norm(src1: vec0, src2: vec1_)
            XCTAssert(min(scale1, scale2) < OpenCVTestCase.EPS)
        }
    }

    func testPCAComputeMatMatMatInt() throws {
        let data = Mat(rows: 3, cols: 4, type: CvType.CV_32F)
        try data.put(row: 0, col: 0, data: [1, 2, 2, 4] as [Float])
        try data.put(row: 1, col: 0, data: [2, 4, 4, 8] as [Float])
        try data.put(row: 2, col: 0, data: [3, 6, 6, 12] as [Float])
        let mean = Mat()
        let vectors = Mat()

        Core.PCACompute(data:data, mean:mean, eigenvectors:vectors, maxComponents:1)

        let mean_truth = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try mean_truth.put(row: 0, col: 0, data: [2, 4, 4, 8] as [Float])
        let vectors_truth = Mat(rows: 1, cols: 4, type: CvType.CV_32F, scalar: Scalar(0))
        try vectors_truth.put(row: 0, col: 0, data: [0.2, 0.4, 0.4, 0.8] as [Float])
        try assertMatEqual(mean_truth, mean, OpenCVTestCase.EPS)
        // eigenvectors are normalized (length = 1),
        // but direction is unknown (v and -v are both eigen vectors)
        // so this direct check doesn't work:
        // try assertMatEqual(vectors_truth, vectors, OpenCVTestCase.EPS)
        for i in 0..<1 {
            let vec0 = vectors_truth.row(Int32(i))
            let vec1 = vectors.row(Int32(i))
            let vec1_ = Mat()
            Core.subtract(src1: Mat(rows: 1, cols: 4, type: CvType.CV_32F, scalar: Scalar(0)), src2: vec1, dst: vec1_)
            let scale1 = Core.norm(src1: vec0, src2: vec1)
            let scale2 = Core.norm(src1: vec0, src2: vec1_)
            XCTAssert(min(scale1, scale2) < OpenCVTestCase.EPS)
        }
    }

    func testPCAProject() throws {
        let mean = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try mean.put(row: 0, col: 0, data: [2, 4, 4, 8] as [Float])
        let vectors = Mat(rows: 1, cols: 4, type: CvType.CV_32F, scalar: Scalar(0))
        try vectors.put(row: 0, col: 0, data: [0.2, 0.4, 0.4, 0.8] as [Float])
        let data = Mat(rows: 3, cols: 4, type: CvType.CV_32F)
        try data.put(row: 0, col: 0, data: [1, 2, 2, 4] as [Float])
        try data.put(row: 1, col: 0, data: [2, 4, 4, 8] as [Float])
        try data.put(row: 2, col: 0, data: [0, 0, 0, 0] as [Float])
        let result = Mat()

        Core.PCAProject(data: data, mean: mean, eigenvectors: vectors, result: result)

        let truth = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try truth.put(row: 0, col: 0, data: [-5, 0, -10] as [Float])
        try assertMatEqual(truth, result, OpenCVTestCase.EPS)
    }

    func testPerspectiveTransform() throws {
        let src = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC2)
        Core.randu(dst: src, low: 0, high: 256)
        let transformMatrix = Mat.eye(rows: 3, cols: 3, type: CvType.CV_32F)

        Core.perspectiveTransform(src: src, dst: dst, m: transformMatrix)
        try assertMatEqual(src, dst, OpenCVTestCase.EPS)
    }

    func testPerspectiveTransform3D() throws {
        let src = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC3)
        Core.randu(dst: src, low: 0, high: 256)
        let transformMatrix = Mat.eye(rows: 4, cols: 4, type: CvType.CV_32F)

        Core.perspectiveTransform(src: src, dst: dst, m: transformMatrix)

        try assertMatEqual(src, dst, OpenCVTestCase.EPS)
    }

    func testPhaseMatMatMat() throws {
        let x = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [10.0, 10.0, 20.0, 5.0] as [Float])
        let y = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try y.put(row: 0, col: 0, data: [20.0, 15.0, 20.0, 20.0] as [Float])
        let gold = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try gold.put(row: 0, col: 0, data: [atan2rad(20, 10), atan2rad(15, 10), atan2rad(20, 20), atan2rad(20, 5)])

        Core.phase(x: x, y: y, angle: dst)

        try assertMatEqual(gold, dst, OpenCVTestCase.EPS)
    }

    func testPhaseMatMatMatBoolean() throws {
        let x = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [10.0, 10.0, 20.0, 5.0] as [Float])
        let y = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try y.put(row: 0, col: 0, data: [20.0, 15.0, 20.0, 20.0] as [Float])
        let gold = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try gold.put(row: 0, col: 0, data: [atan2deg(20, 10), atan2deg(15, 10), atan2deg(20, 20), atan2deg(20, 5)])

        Core.phase(x: x, y: y, angle: dst, angleInDegrees: true)

        try assertMatEqual(gold, dst, OpenCVTestCase.EPS * 180 / Double.pi)
    }

    func testPolarToCartMatMatMatMat() throws {
        let magnitude = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try magnitude.put(row: 0, col: 0, data: [5.0, 10.0, 13.0])
        let angle = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try angle.put(row: 0, col: 0, data: [0.92729962, 0.92729962, 1.1759995])
        let xCoordinate = Mat()
        let yCoordinate = Mat()

        Core.polarToCart(magnitude: magnitude, angle: angle, x: xCoordinate, y: yCoordinate)

        let x = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [3.0, 6.0, 5, 0])
        let y = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try y.put(row: 0, col: 0, data: [4.0, 8.0, 12.0])
        try assertMatEqual(x, xCoordinate, OpenCVTestCase.EPS)
        try assertMatEqual(y, yCoordinate, OpenCVTestCase.EPS)
    }

    func testPolarToCartMatMatMatMatBoolean() throws {
        let magnitude = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try magnitude.put(row: 0, col: 0, data: [5.0, 10.0, 13.0])
        let angle = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try angle.put(row: 0, col: 0, data: [0.92729962, 0.92729962, 1.1759995])
        let xCoordinate = Mat()
        let yCoordinate = Mat()

        Core.polarToCart(magnitude: magnitude, angle: angle, x: xCoordinate, y: yCoordinate, angleInDegrees: true)

        let x = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try x.put(row: 0, col: 0, data: [4.9993458, 9.9986916, 12.997262])
        let y = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
        try y.put(row: 0, col: 0, data: [0.080918625, 0.16183725, 0.26680708])
        try assertMatEqual(x, xCoordinate, OpenCVTestCase.EPS)
        try assertMatEqual(y, yCoordinate, OpenCVTestCase.EPS)
    }

    func testPow() throws {
        Core.pow(src: gray2, power: 7, dst: dst)

        try assertMatEqual(gray128, dst)
    }

    func testRandn() {
        Core.randn(dst: gray0, mean: 100, stddev: 23)

        XCTAssertEqual(100, Core.mean(src: gray0).val[0] as! Double, accuracy:23 / 2)
    }

    func testRandShuffleMat() throws {
        let original = Mat(rows: 1, cols: 10, type: CvType.CV_32F)
        try original.put(row: 0, col: 0, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as [Float])
        let shuffled = original.clone()

        Core.randShuffle(dst: shuffled)

        try assertMatNotEqual(original, shuffled, OpenCVTestCase.EPS)
        let dst1 = Mat()
        let dst2 = Mat()
        Core.sort(src: original, dst: dst1, flags: SortFlags.SORT_ASCENDING.rawValue)
        Core.sort(src: shuffled, dst: dst2, flags: SortFlags.SORT_ASCENDING.rawValue)
        try assertMatEqual(dst1, dst2, OpenCVTestCase.EPS)
    }

    func testRandShuffleMatDouble() throws {
        let original = Mat(rows: 1, cols: 10, type: CvType.CV_32F)
        try original.put(row: 0, col: 0, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as [Float])
        let shuffled = original.clone()

        Core.randShuffle(dst: shuffled, iterFactor: 10)

        try assertMatNotEqual(original, shuffled, OpenCVTestCase.EPS)
        let dst1 = Mat()
        let dst2 = Mat()
        Core.sort(src: original, dst: dst1, flags: SortFlags.SORT_ASCENDING.rawValue)
        Core.sort(src: shuffled, dst: dst2, flags: SortFlags.SORT_ASCENDING.rawValue)
        try assertMatEqual(dst1, dst2, OpenCVTestCase.EPS)
    }

    func testRandu() {
        Core.randu(dst: gray0, low: 3, high: 23)
        XCTAssert(Core.checkRange(a: gray0, quiet: true, minVal: 3, maxVal: 23))
    }

    func testRectangleMatPointPointScalar() {
        let bottomRight = Point(x: gray0.cols() / 2, y: gray0.rows() / 2)
        let topLeft = Point(x: 0, y: 0)
        let color = Scalar(128)

        Imgproc.rectangle(img: gray0, pt1: bottomRight, pt2: topLeft, color: color)

        XCTAssert(0 != Core.countNonZero(src: gray0))
    }

    func testRectangleMatPointPointScalarInt() {
        let bottomRight = Point(x: gray0.cols(), y: gray0.rows())
        let topLeft = Point(x: 0, y: 0)
        let color = Scalar(128)

        Imgproc.rectangle(img: gray0, pt1: bottomRight, pt2: topLeft, color: color, thickness: 2)
        Imgproc.rectangle(img: gray0, pt1: bottomRight, pt2: topLeft, color: colorBlack)

        XCTAssert(0 != Core.countNonZero(src: gray0))
    }

    func testRectangleMatPointPointScalarIntInt() {
        let bottomRight = Point(x: gray0.cols() / 2, y: gray0.rows() / 2)
        let topLeft = Point(x: 0, y: 0)
        let color = Scalar(128)

        Imgproc.rectangle(img: gray0, pt1: bottomRight, pt2: topLeft, color: color, thickness: 2, lineType: .LINE_AA, shift: 0)
        Imgproc.rectangle(img: gray0, pt1: bottomRight, pt2: topLeft, color: colorBlack, thickness: 2, lineType: .LINE_4, shift: 0)

        XCTAssert(0 != Core.countNonZero(src: gray0))
    }

    func testRectangleMatPointPointScalarIntIntInt() {
        let bottomRight1 = Point(x: gray0.cols(), y: gray0.rows())
        let bottomRight2 = Point(x: gray0.cols() / 2, y: gray0.rows() / 2)
        let topLeft = Point(x: 0, y: 0)
        let color = Scalar(128)

        Imgproc.rectangle(img: gray0, pt1: bottomRight1, pt2: topLeft, color: color, thickness: 2, lineType: .LINE_8, shift: 1)

        XCTAssert(0 != Core.countNonZero(src: gray0))

        Imgproc.rectangle(img: gray0, pt1: bottomRight2, pt2: topLeft, color: colorBlack, thickness: 2, lineType: .LINE_8, shift: 0)

        XCTAssertEqual(0, Core.countNonZero(src: gray0))
    }

    func testReduceMatMatIntInt() throws {
        let src = Mat(rows: 2, cols: 2, type: CvType.CV_32F)

        try src.put(row: 0, col: 0, data: [1, 0] as [Float])
        try src.put(row: 1, col: 0, data: [3, 0] as [Float])

        Core.reduce(src: src, dst: dst, dim: 0, rtype: Int32(Core.REDUCE_AVG))

        let out = Mat(rows: 1, cols: 2, type: CvType.CV_32F)
        try out.put(row: 0, col: 0, data: [2, 0] as [Float])
        try assertMatEqual(out, dst, OpenCVTestCase.EPS)
    }

    func testReduceMatMatIntIntInt() throws {
        let src = Mat(rows: 2, cols: 2, type: CvType.CV_32F)
        try src.put(row: 0, col: 0, data: [1, 0] as [Float])
        try src.put(row: 1, col: 0, data: [2, 3] as [Float])

        Core.reduce(src: src, dst: dst, dim: 1, rtype: Int32(Core.REDUCE_SUM), dtype: CvType.CV_64F)

        let out = Mat(rows: 2, cols: 1, type: CvType.CV_64F)
        try out.put(row: 0, col: 0, data: [1, 5] as [Double])
        try assertMatEqual(out, dst, OpenCVTestCase.EPS)
    }

    func testRepeat() throws {
        let src = Mat(rows: 1, cols: 2, type: CvType.CV_32F, scalar: Scalar(0))

        Core.repeat(src: src, ny: OpenCVTestCase.matSize, nx: OpenCVTestCase.matSize / 2, dst: dst)

        try assertMatEqual(gray0_32f, dst, OpenCVTestCase.EPS)
    }

    func testScaleAdd() throws {
        Core.scaleAdd(src1: gray3, alpha: 2.0, src2: gray3, dst: dst)

        try assertMatEqual(gray9, dst)
    }

    func testSetIdentityMat() throws {
        Core.setIdentity(mtx: gray0_32f)

        try assertMatEqual(grayE_32f, gray0_32f, OpenCVTestCase.EPS)
    }

    func testSetIdentityMatScalar() throws {
        let m = gray0_32f;

        Core.setIdentity(mtx: m, s: Scalar(5))

        truth = Mat(size: m.size(), type: m.type(), scalar: Scalar(0))
        truth!.diag().setTo(scalar: Scalar(5))
        try assertMatEqual(truth!, m, OpenCVTestCase.EPS)
    }

    func testSolveCubic() throws {
        let coeffs = Mat(rows: 1, cols: 4, type: CvType.CV_32F)
        try coeffs.put(row: 0, col: 0, data: [1, 6, 11, 6] as [Float])

        XCTAssertEqual(3, Core.solveCubic(coeffs: coeffs, roots: dst))

        let roots = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try roots.put(row: 0, col: 0, data: [-3, -1, -2] as [Float])
        try assertMatEqual(roots, dst, OpenCVTestCase.EPS)
    }

    func testSolveMatMatMat() throws {
        let a = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try a.put(row: 0, col: 0, data: [1, 1, 1] as [Float])
        try a.put(row: 1, col: 0, data: [1, -2, 2] as [Float])
        try a.put(row: 2, col: 0, data: [1, 2, 1] as [Float])
        let b = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try b.put(row: 0, col: 0, data: [0, 4, 2] as [Float])

        XCTAssert(Core.solve(src1: a, src2: b, dst: dst))

        let res = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try res.put(row: 0, col: 0, data: [-12, 2, 10] as [Float])
        try assertMatEqual(res, dst, OpenCVTestCase.EPS)
    }

    func testSolveMatMatMatInt() throws {
        let a = Mat(rows: 3, cols: 3, type: CvType.CV_32F)
        try a.put(row: 0, col: 0, data: [1, 1, 1] as [Float])
        try a.put(row: 1, col: 0, data: [1, -2, 2] as [Float])
        try a.put(row: 2, col: 0, data: [1, 2, 1] as [Float])
        let b = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try b.put(row: 0, col: 0, data: [0, 4, 2] as [Float])

        XCTAssert(Core.solve(src1: a, src2: b, dst: dst, flags: DecompTypes.DECOMP_QR.rawValue | DecompTypes.DECOMP_NORMAL.rawValue))

        let res = Mat(rows: 3, cols: 1, type: CvType.CV_32F)
        try res.put(row: 0, col: 0, data: [-12, 2, 10] as [Float])
        try assertMatEqual(res, dst, OpenCVTestCase.EPS)
    }

    func testSolvePolyMatMat() throws {
        let coeffs = Mat(rows: 4, cols: 1, type: CvType.CV_32F)
        try coeffs.put(row: 0, col: 0, data: [-6, 11, -6, 1] as [Float])
        let roots = Mat()

        XCTAssertGreaterThanOrEqual(1e-6, abs(Core.solvePoly(coeffs: coeffs, roots: roots)))

        truth = Mat(rows: 3, cols: 1, type: CvType.CV_32FC2)
        try truth!.put(row: 0, col: 0, data: [1, 0, 2, 0, 3, 0] as [Float])
        try assertMatEqual(truth!, roots, OpenCVTestCase.EPS)
    }

    func testSolvePolyMatMatInt() throws {
        let coeffs = Mat(rows: 4, cols: 1, type: CvType.CV_32F)
        try coeffs.put(row: 0, col: 0, data: [-6, 11, -6, 1] as [Float])
        let roots = Mat()

        XCTAssertEqual(10.198039027185569, Core.solvePoly(coeffs: coeffs, roots: roots, maxIters: 1))

        truth = Mat(rows: 3, cols: 1, type: CvType.CV_32FC2)
        try truth!.put(row: 0, col: 0, data: [1, 0, -1, 2, -2, 12] as [Float])
        try assertMatEqual(truth!, roots, OpenCVTestCase.EPS)
    }

    func testSort() {
        var submat = gray0.submat(rowStart: 0, rowEnd: gray0.rows() / 2, colStart: 0, colEnd: gray0.cols() / 2)
        submat.setTo(scalar: Scalar(1.0))

        Core.sort(src: gray0, dst: dst, flags: SortFlags.SORT_EVERY_ROW.rawValue)

        submat = dst.submat(rowStart: 0, rowEnd: dst.rows() / 2, colStart: dst.cols() / 2, colEnd: dst.cols())
        XCTAssert(submat.total() == Core.countNonZero(src: submat))

        Core.sort(src: gray0, dst: dst, flags: SortFlags.SORT_EVERY_COLUMN.rawValue)

        submat = dst.submat(rowStart: dst.rows() / 2, rowEnd: dst.rows(), colStart: 0, colEnd: dst.cols() / 2)

        XCTAssert(submat.total() == Core.countNonZero(src: submat))
    }

    func testSortIdx() throws {
        let a = Mat.eye(rows: 3, cols: 3, type: CvType.CV_8UC1)
        let b = Mat()

        Core.sortIdx(src: a, dst: b, flags: SortFlags.SORT_EVERY_ROW.rawValue | SortFlags.SORT_ASCENDING.rawValue)

        truth = Mat(rows: 3, cols: 3, type: CvType.CV_32SC1)
        try truth!.put(row: 0, col: 0, data: [1, 2, 0] as [Int32])
        try truth!.put(row: 1, col: 0, data: [0, 2, 1] as [Int32])
        try truth!.put(row: 2, col: 0, data: [0, 1, 2] as [Int32])
        try assertMatEqual(truth!, b)
    }

    func testSplit() throws {
        let m = getMat(CvType.CV_8UC3, vals: [1, 2, 3])
        let cois = NSMutableArray()

        Core.split(m: m, mv: cois)

        try assertMatEqual(gray1, cois[0] as! Mat)
        try assertMatEqual(gray2, cois[1] as! Mat)
        try assertMatEqual(gray3, cois[2] as! Mat)
    }

    func testSqrt() throws {
        Core.sqrt(src: gray9_32f, dst: dst)

        try assertMatEqual(gray3_32f, dst, OpenCVTestCase.EPS)

        let rgba144 = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC4, scalar: Scalar.all(144))
        let rgba12 = Mat(rows: OpenCVTestCase.matSize, cols: OpenCVTestCase.matSize, type: CvType.CV_32FC4, scalar: Scalar.all(12))

        Core.sqrt(src: rgba144, dst: dst)

        try assertMatEqual(rgba12, dst, OpenCVTestCase.EPS)
    }

    func testSubtractMatMatMat() throws {
        Core.subtract(src1: gray128, src2: gray1, dst: dst)

        try assertMatEqual(gray127, dst)
    }

    func testSubtractMatMatMatMat() throws {
        let mask = makeMask(gray1.clone())
        dst = gray128.clone()

        Core.subtract(src1: gray128, src2: gray1, dst: dst, mask: mask)

        try assertMatEqual(makeMask(gray127, vals: [128]), dst)
    }

    func testSubtractMatMatMatMatInt() throws {
        Core.subtract(src1: gray3, src2: gray2, dst: dst, mask: gray1, dtype: CvType.CV_32F)

        try assertMatEqual(gray1_32f, dst, OpenCVTestCase.EPS)
    }

    func testSumElems() throws {
        let src = Mat(rows: 4, cols: 4, type: CvType.CV_8U, scalar: Scalar(10))

        let res1 = Core.sum(src: src)

        assertScalarEqual(Scalar(160), res1, OpenCVTestCase.EPS)
    }

    func testSVBackSubst() throws {
        let w = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(2))
        let u = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(4))
        let vt = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(2))
        let rhs = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(1))

        Core.SVBackSubst(w: w, u: u, vt: vt, rhs: rhs, dst: dst)

        let truth = Mat(rows: 2, cols: 2, type: CvType.CV_32FC1, scalar: Scalar(16))
        try assertMatEqual(truth, dst, OpenCVTestCase.EPS)
    }

    func testSVDecompMatMatMatMat() throws {
        let src = Mat(rows: 1, cols: 4, type: CvType.CV_32FC1)
        try src.put(row: 0, col: 0, data: [1, 4, 8, 6] as [Float])
        let w = Mat()
        let u = Mat()
        let vt = Mat()

        Core.SVDecomp(src: src, w: w, u: u, vt: vt)

        let truthW = Mat(rows: 1, cols: 1, type: CvType.CV_32FC1, scalar: Scalar(10.816654))
        let truthU = Mat(rows: 1, cols: 1, type: CvType.CV_32FC1, scalar: Scalar(1))
        let truthVT = Mat(rows: 1, cols: 4, type: CvType.CV_32FC1)
        try truthVT.put(row: 0, col: 0, data: [0.09245003, 0.36980012, 0.73960024, 0.5547002])
        try assertMatEqual(truthW, w, OpenCVTestCase.EPS)
        try assertMatEqual(truthU, u, OpenCVTestCase.EPS)
        try assertMatEqual(truthVT, vt, OpenCVTestCase.EPS)
    }

    func testSVDecompMatMatMatMatInt() throws {
        let src = Mat(rows: 1, cols: 4, type: CvType.CV_32FC1)
        try src.put(row: 0, col: 0, data: [1, 4, 8, 6] as [Float])
        let w = Mat()
        let u = Mat()
        let vt = Mat()

        Core.SVDecomp(src: src, w: w, u: u, vt: vt, flags: Int32(Core.SVD_NO_UV))

        let truthW = Mat(rows: 1, cols: 1, type: CvType.CV_32FC1, scalar: Scalar(10.816654))
        try assertMatEqual(truthW, w, OpenCVTestCase.EPS)
        XCTAssert(u.empty())
        XCTAssert(vt.empty())
    }

    func testTrace() {
        let s = Core.trace(mtx: gray1)

        XCTAssertEqual(Scalar(Double(OpenCVTestCase.matSize)), s)
    }

    func testTransform() throws {
        let src = Mat(rows: 2, cols: 2, type: CvType.CV_32F, scalar: Scalar(55))
        let m = Mat.eye(rows: 2, cols: 2, type: CvType.CV_32FC1)

        Core.transform(src: src, dst: dst, m: m)

        truth = Mat(rows: 2, cols: 2, type: CvType.CV_32FC2, scalar: Scalar(55, 1))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testTranspose() {
        gray0.submat(rowStart: 0, rowEnd: gray0.rows() / 2, colStart: 0, colEnd: gray0.cols()).setTo(scalar: Scalar(1))
        let destination = getMat(CvType.CV_8U, vals: [0])

        Core.transpose(src: gray0, dst: destination)

        let subdst = destination.submat(rowStart: 0, rowEnd: destination.rows(), colStart: 0, colEnd: destination.cols() / 2)
        XCTAssert(subdst.total() == Core.countNonZero(src: subdst))
    }

    func testVconcat() throws {
        let mats = [Mat.eye(rows: 3, cols: 3, type: CvType.CV_8U), Mat.zeros(2, cols: 3, type: CvType.CV_8U)]

        Core.vconcat(src: mats, dst: dst)

        try assertMatEqual(Mat.eye(rows: 5, cols: 3, type: CvType.CV_8U), dst)

    }

    func testCopyMakeBorderMatMatIntIntIntIntInt() throws {
        let src = Mat(rows: 2, cols: 2, type: CvType.CV_32F, scalar: Scalar(1))
        let border: Int32 = 2

        Core.copyMakeBorder(src: src, dst: dst, top: border, bottom: border, left: border, right: border, borderType: .BORDER_REPLICATE)

        truth = Mat(rows: 6, cols: 6, type: CvType.CV_32F, scalar: Scalar(1))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testCopyMakeBorderMatMatIntIntIntIntIntScalar() throws {
        let src = Mat(rows: 2, cols: 2, type: CvType.CV_32F, scalar: Scalar(1))

        let value = Scalar(0)
        let border: Int32 = 2

        Core.copyMakeBorder(src: src, dst: dst, top: border, bottom: border, left: border, right: border, borderType: .BORDER_REPLICATE, value: value)
        // TODO_: write better test (use Core.BORDER_CONSTANT)

        truth = Mat(rows: 6, cols: 6, type: CvType.CV_32F, scalar: Scalar(1))
        try assertMatEqual(truth!, dst, OpenCVTestCase.EPS)
    }

    func testBorderInterpolate() {
        let val1 = Core.borderInterpolate(p: 100, len: 150, borderType: .BORDER_REFLECT_101)
        XCTAssertEqual(100, val1)

        let val2 = Core.borderInterpolate(p: -5, len: 10, borderType: .BORDER_WRAP)
        XCTAssertEqual(5, val2)
    }

    func atan2deg(_ y:Double, _ x:Double) -> Double {
        var res = atan2(y, x)
        if (res < 0) {
            res = Double.pi * 2 + res
        }
        return res * 180 / Double.pi
    }

    func atan2rad(_ y:Double, _ x:Double) -> Double {
        var res = atan2(y, x)
        if (res < 0) {
            res = Double.pi * 2 + res
        }
        return res
    }
}
