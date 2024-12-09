//
//  ScalarTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class ScalarTest: OpenCVTestCase {

    let s1 = Scalar(1.0)
    let s2 = Scalar.all(1.0)

    func testAll() {
        let dstScalar = Scalar.all(2.0)
        let truth = Scalar(2.0, 2.0, 2.0, 2.0)
        XCTAssertEqual(truth, dstScalar)
    }

    func testClone() {
        let dstScalar = s2.clone()
        XCTAssertEqual(s2, dstScalar)
    }

    func testConj() {
        let dstScalar = s2.conj()
        let truth = Scalar(1, -1, -1, -1)
        XCTAssertEqual(truth, dstScalar)
    }

    func testEqualsObject() {
        let dstScalar = s2.clone()
        XCTAssert(s2 == dstScalar)

        XCTAssertFalse(s2 == s1)
    }

    func testHashCode() {
        XCTAssertEqual(s2.hash(), s2.hash())
    }

    func testIsReal() {
        XCTAssert(s1.isReal())

        XCTAssertFalse(s2.isReal())
    }

    func testMulScalar() {
        let dstScalar = s2.mul(s1)
        XCTAssertEqual(s1, dstScalar)
    }

    func testMulScalarDouble() {
        let multiplier = 2.0
        let dstScalar = s2.mul(s1, scale: multiplier)
        let truth = Scalar(2)
        XCTAssertEqual(truth, dstScalar)
    }

    func testScalarDouble() {
        let truth = Scalar(1)
        XCTAssertEqual(truth, s1)
    }

    func testScalarDoubleArray() {
        let vals: [Double] = [2.0, 4.0, 5.0, 3.0]
        let dstScalar = Scalar(vals:vals as [NSNumber])

        let truth = Scalar(2.0, 4.0, 5.0, 3.0)
        XCTAssertEqual(truth, dstScalar)
    }

    func testScalarDoubleDouble() {
        let dstScalar = Scalar(2, 5)
        let truth = Scalar(2.0, 5.0, 0.0, 0.0)
        XCTAssertEqual(truth, dstScalar)
    }

    func testScalarDoubleDoubleDouble() {
        let dstScalar = Scalar(2.0, 5.0, 5.0)
        let truth = Scalar(2.0, 5.0, 5.0, 0.0)
        XCTAssertEqual(truth, dstScalar);
    }

    func testScalarDoubleDoubleDoubleDouble() {
        let dstScalar = Scalar(2.0, 5.0, 5.0, 9.0)
        let truth = Scalar(2.0, 5.0, 5.0, 9.0)
        XCTAssertEqual(truth, dstScalar)
    }

    func testToString() {
        let actual = "\(s2)"
        let expected = "Scalar [1.000000, 1.000000, 1.000000, 1.000000]"
        XCTAssertEqual(expected, actual)
    }

}
