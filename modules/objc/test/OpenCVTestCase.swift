//
//  OpenCVTestCase.swift
//  StitchAppTests
//
//  Created by Giles Payne on 2020/01/19.
//  Copyright © 2020 Xtravision. All rights reserved.
//

import XCTest
import StitchApp

enum OpenCVTestError: Error {
    case unsupportedOperationError(String)
}

open class OpenCVTestCase: XCTestCase {

    //change to 'true' to unblock fail on fail("Not yet implemented")
    static let passNYI = true

    static let isTestCaseEnabled = true

    static let XFEATURES2D = "xfeatures2d"
    static let DEFAULT_FACTORY = "create"

    static let matSize: Int32 = 10
    static let EPS = 0.001
    static let weakEPS = 0.5

    var dst: Mat = Mat()
    var truth: Mat? = nil

    let colorBlack = Scalar(0)
    let colorWhite = Scalar(255, 255, 255)

    // Naming notation: <channels info>_[depth]_[dimensions]_value
    // examples: gray0 - single channel 8U 2d Mat filled with 0
    // grayRnd - single channel 8U 2d Mat filled with random numbers
    // gray0_32f_1d

    let gray0 = Mat(rows:matSize, cols:matSize, type:CvType.CV_8U, scalar:Scalar(0))
    let gray1 = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U, scalar: Scalar(1))
    let gray2 = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U, scalar: Scalar(2))
    let gray3 = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U, scalar: Scalar(3))
    let gray9 = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U, scalar: Scalar(9))
    let gray127 = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U, scalar: Scalar(127))
    let gray128 = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U, scalar: Scalar(128))
    let gray255 = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U, scalar: Scalar(255))
    let grayRnd = Mat(rows:matSize, cols:matSize, type: CvType.CV_8U)

    let gray_16u_256 = Mat(rows: matSize, cols: matSize, type: CvType.CV_16U, scalar: Scalar(256))
    let gray_16s_1024 = Mat(rows: matSize, cols: matSize, type: CvType.CV_16S, scalar: Scalar(1024))

    let gray0_32f = Mat(rows: matSize, cols: matSize, type: CvType.CV_32F, scalar: Scalar(0.0))
    let gray1_32f = Mat(rows: matSize, cols: matSize, type: CvType.CV_32F, scalar: Scalar(1.0))
    let gray3_32f = Mat(rows: matSize, cols: matSize, type: CvType.CV_32F, scalar: Scalar(3.0))
    let gray9_32f = Mat(rows: matSize, cols: matSize, type: CvType.CV_32F, scalar: Scalar(9.0))
    let gray255_32f = Mat(rows: matSize, cols: matSize, type: CvType.CV_32F, scalar: Scalar(255.0))
    let grayE_32f = Mat.eye(rows: matSize, cols: matSize, type: CvType.CV_32FC1)
    let grayRnd_32f = Mat(rows: matSize, cols: matSize, type: CvType.CV_32F)

    let gray0_64f = Mat(rows: matSize, cols: matSize, type: CvType.CV_64F, scalar: Scalar(0.0))
    let gray0_32f_1d = Mat(rows: 1, cols: matSize, type: CvType.CV_32F, scalar: Scalar(0.0))
    let gray0_64f_1d = Mat(rows: 1, cols: matSize, type: CvType.CV_64F, scalar: Scalar(0.0))

    let rgba0 = Mat(rows: matSize, cols: matSize, type: CvType.CV_8UC4, scalar: Scalar.all(0))
    let rgba128 = Mat(rows: matSize, cols: matSize, type: CvType.CV_8UC4, scalar: Scalar.all(128))
    
    let rgbLena = Imgcodecs.imread(Bundle(for: OpenCVTestCase.self).path(forResource:"lena", ofType:"png", inDirectory:"resources")!)
    let grayChess = Imgcodecs.imread(Bundle(for: OpenCVTestCase.self).path(forResource:"chessboard", ofType:"jpg", inDirectory:"resources")!)

    let gray255_32f_3d = Mat(sizes: [matSize, matSize, matSize] as [NSNumber], type: CvType.CV_32F, scalar: Scalar(255.0))

    let v1 = Mat(rows: 1, cols: 3, type: CvType.CV_32F)
    let v2 = Mat(rows: 1, cols: 3, type: CvType.CV_32F)

    override open func setUp() {
        //Core.setErrorVerbosity(false)
        Core.randu(grayRnd, low: 0, high: 255)
        Core.randu(grayRnd_32f, low:0, high: 255)
        v1.put(row: 0,col: 0, data: [1.0, 3.0, 2.0])
        v2.put(row: 0,col: 0, data: [2.0, 1.0, 3.0])
        
    }

    override open func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    func assertMatEqual(_ expected:Mat, _ actual:Mat) throws {
        try compareMats(expected, actual, true)
    }

    func assertMatNotEqual(_ expected:Mat, _ actual:Mat) throws {
        try compareMats(expected, actual, false)
    }

    func assertMatEqual(_ expected:Mat, _ actual:Mat, _ eps:Double) throws {
        try compareMats(expected, actual, eps, true);
    }

    func assertMatNotEqual(_ expected:Mat, _ actual:Mat, _ eps:Double) throws {
        try compareMats(expected, actual, eps, false);
    }

    func assertSizeEquals(_ expected:CVSize,_ actual: CVSize,_ eps: Double) {
        let msg = "expected:<\(expected)> but was:<\(actual)>"
        XCTAssertEqual(expected.width, actual.width, accuracy:eps, msg)
        XCTAssertEqual(expected.height, actual.height, accuracy:eps, msg)
    }

    func assertPointEquals(_ expected:CVPoint, _ actual: CVPoint, _ eps: Double) {
        let msg = "expected:<\(expected)> but was:<\(actual)>"
        XCTAssertEqual(expected.x, actual.x, accuracy:eps, msg)
        XCTAssertEqual(expected.y, actual.y, accuracy:eps, msg)
    }
    
    func compareMats(_ expected:Mat, _ actual:Mat, _ isEqualityMeasured:Bool) throws {
        if expected.type() != actual.type() || !dimensionsEqual(expected, actual) {
            throw OpenCVTestError.unsupportedOperationError("Incompatible matrices")
        }

        if (expected.depth() == CvType.CV_32F || expected.depth() == CvType.CV_64F) {
            if isEqualityMeasured {
                throw OpenCVTestError.unsupportedOperationError("Floating-point Mats must not be checked for exact match. Use assertMatEqual(expected:Mat, actual:Mat, eps:Double) instead.")
            } else {
                throw OpenCVTestError.unsupportedOperationError("Floating-point Mats must not be checked for exact match. Use assertMatNotEqual(expected:Mat, actual:Mat, eps:Double) instead.")
            }
        }

        let diff = Mat()
        Core.absdiff(expected, src2: actual, dst: diff);
        let reshaped = diff.reshape(channels: 1)
        let mistakes = Core.countNonZero(reshaped)

        if isEqualityMeasured {
            XCTAssertTrue(mistakes == 0, "Mats are different in \(mistakes) points")
        } else {
            XCTAssertFalse(mistakes == 0, "Mats are equal")
        }
    }

    func compareMats(_ expected: Mat, _ actual:Mat, _ eps:Double, _ isEqualityMeasured:Bool) throws {
        if expected.type() != actual.type() || !dimensionsEqual(expected, actual) {
            throw OpenCVTestError.unsupportedOperationError("Incompatible matrices")
        }

        let diff = Mat()
        Core.absdiff(expected, src2: actual, dst: diff)
        let maxDiff = Core.norm(diff, normType: NormTypes.NORM_INF.rawValue)

        if isEqualityMeasured {
            XCTAssertTrue(maxDiff <= eps, "Max difference between expected and actual Mats is \(maxDiff), that bigger than \(eps)")
        } else {
            XCTAssertFalse(maxDiff <= eps, "Max difference between expected and actual Mats is \(maxDiff), that less than \(eps)")
        }
    }

    func dimensionsEqual(_ expected: Mat, _ actual: Mat) -> Bool {
        if expected.dims() != actual.dims() {
            return false
        }
        if expected.dims() > 2 {
            return (0..<expected.dims()).allSatisfy { expected.size($0) == actual.size($0) }
        } else {
            return expected.cols() == actual.cols() && expected.rows() == actual.rows();
        }
    }

    func makeMask(_ mat:Mat, vals:[Double] = []) -> Mat {
        mat.submat(rowStart: 0, rowEnd: mat.rows(), colStart: 0, colEnd: mat.cols() / 2).setTo(scalar: Scalar(vals: vals as [NSNumber]))
        return mat
    }
}
