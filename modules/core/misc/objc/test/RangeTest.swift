//
//  RangeTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class RangeTest: OpenCVTestCase {

    let r1 = Range(start: 1, end: 11)
    let r2 = Range(start: 1, end: 1)

    func testAll() {
        let range = Range.all()
        XCTAssertEqual(Int32.min, range.start)
        XCTAssertEqual(Int32.max, range.end)
    }

    func testClone() {
        let dstRange = r1.clone()
        XCTAssertEqual(r1, dstRange)
    }

    func testEmpty() {
        var flag = r1.empty()
        XCTAssertFalse(flag)

        flag = r2.empty()
        XCTAssert(flag)
    }

    func testEqualsObject() {
        XCTAssertFalse(r2 == r1)

        let range = r1.clone()
        XCTAssert(r1 == range)
    }

    func testHashCode() {
        XCTAssertEqual(r1.hash(), r1.hash())
    }

    func testIntersection() {
        let range = r1.intersection(r2)
        XCTAssertEqual(r2, range)
    }

    func testRange() {
        let range = Range()

        XCTAssertNotNil(range)
        XCTAssertEqual(0, range.start)
        XCTAssertEqual(0, range.end)
    }

    func testRangeDoubleArray() {
        let vals:[Double] = [2, 4]
        let r = Range(vals: vals as [NSNumber])

        XCTAssert(2 == r.start);
        XCTAssert(4 == r.end);
    }

    func testRangeIntInt() {
        let r1 = Range(start: 12, end: 13)

        XCTAssertNotNil(r1);
        XCTAssertEqual(12, r1.start);
        XCTAssertEqual(13, r1.end);
    }

    func testSet() {
        let vals1:[Double] = []
        r1.set(vals: vals1 as [NSNumber])
        XCTAssertEqual(0, r1.start)
        XCTAssertEqual(0, r1.end)

        let vals2 = [6, 10]
        r2.set(vals: vals2 as [NSNumber])
        XCTAssertEqual(6, r2.start)
        XCTAssertEqual(10, r2.end)
    }

    func testShift() {
        let delta:Int32 = 1
        let range = Range().shift(delta)
        XCTAssertEqual(r2, range)
    }

    func testSize() {
        XCTAssertEqual(10, r1.size())

        XCTAssertEqual(0, r2.size())
    }

    func testToString() {
        let actual = "\(r1)"
        let expected = "Range {1, 11}"
        XCTAssertEqual(expected, actual)
    }

}
