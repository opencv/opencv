//
//  TermCriteriaTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class TermCriteriaTest: OpenCVTestCase {

    let tc2 = TermCriteria(type: 2, maxCount: 4, epsilon: EPS)

    func testClone() {
        let tc1 = tc2.clone()
        XCTAssertEqual(tc2, tc1)
    }

    func testEqualsObject() {
        var tc1 = TermCriteria()
        XCTAssertFalse(tc2 == tc1)

        tc1 = tc2.clone()
        XCTAssert(tc2 == tc1)
    }

    func testHashCode() {
        XCTAssertEqual(tc2.hash(), tc2.hash())
    }

    func testSet() {
        let tc1 = TermCriteria()
        let vals1:[Double] = []

        tc1.set(vals: vals1 as [NSNumber])

        XCTAssertEqual(0, tc1.type)
        XCTAssertEqual(0, tc1.maxCount)
        XCTAssertEqual(0.0, tc1.epsilon)

        let vals2 = [9, 8, 0.002]
        tc2.set(vals: vals2 as [NSNumber])

        XCTAssertEqual(9, tc2.type)
        XCTAssertEqual(8, tc2.maxCount)
        XCTAssertEqual(0.002, tc2.epsilon)
    }

    func testTermCriteria() {
        let tc1 = TermCriteria()

        XCTAssertNotNil(tc1)
        XCTAssertEqual(0, tc1.type)
        XCTAssertEqual(0, tc1.maxCount)
        XCTAssertEqual(0.0, tc1.epsilon)
    }

    func testTermCriteriaDoubleArray() {
        let vals = [ 3, 2, 0.007]
        let tc1 = TermCriteria(vals: vals as [NSNumber])

        XCTAssertEqual(3, tc1.type)
        XCTAssertEqual(2, tc1.maxCount)
        XCTAssertEqual(0.007, tc1.epsilon)
    }

    func testTermCriteriaIntIntDouble() {
        let tc1 = TermCriteria(type: 2, maxCount: 4, epsilon: OpenCVTestCase.EPS)

        XCTAssertNotNil(tc1)
        XCTAssertEqual(2, tc1.type)
        XCTAssertEqual(4, tc1.maxCount)
        XCTAssertEqual(OpenCVTestCase.EPS, tc1.epsilon)
    }

    func testToString() {
        let actual = "\(tc2)"
        let expected = "TermCriteria { type: 2, maxCount: 4, epsilon: 0.001000}"
        XCTAssertEqual(expected, actual)
    }

}
