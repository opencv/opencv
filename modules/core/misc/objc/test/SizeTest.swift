//
//  SizeTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class SizeTest: OpenCVTestCase {

    let sz1 = Size2d(width: 10.0, height: 10.0)
    let sz2 = Size2d(width: -1, height: -1)

    func testArea() {
        let area = sz1.area()
        XCTAssertEqual(100.0, area);
    }

    func testClone() {
        let dstSize = sz1.clone()
        XCTAssertEqual(sz1, dstSize)
    }

    func testEqualsObject() {
        XCTAssertFalse(sz1 == sz2);

        let sz2 = sz1.clone();
        XCTAssertTrue(sz1 == sz2);
    }

    func testHashCode() {
        XCTAssertEqual(sz1.hash(), sz1.hash());
    }

    func testSet() {
        let vals1:[Double] = []
        sz2.set(vals: vals1 as [NSNumber])
        XCTAssertEqual(0, sz2.width);
        XCTAssertEqual(0, sz2.height);

        let vals2:[Double] = [9, 12]
        sz1.set(vals: vals2 as [NSNumber]);
        XCTAssertEqual(9, sz1.width);
        XCTAssertEqual(12, sz1.height);
    }

    func testSize() {
        let dstSize = Size2d()

        XCTAssertNotNil(dstSize)
        XCTAssertEqual(0, dstSize.width)
        XCTAssertEqual(0, dstSize.height)
    }

    func testSizeDoubleArray() {
        let vals:[Double] = [10, 20]
        let sz2 = Size2d(vals: vals as [NSNumber])

        XCTAssertEqual(10, sz2.width)
        XCTAssertEqual(20, sz2.height)
    }

    func testSizeDoubleDouble() {
        XCTAssertNotNil(sz1)

        XCTAssertEqual(10.0, sz1.width)
        XCTAssertEqual(10.0, sz1.height)
    }

    func testSizePoint() {
        let p = Point2d(x: 2, y: 4)
        let sz1 = Size2d(point: p)

        XCTAssertNotNil(sz1)
        XCTAssertEqual(2.0, sz1.width)
        XCTAssertEqual(4.0, sz1.height)
    }

    func testToString() {
        let actual = "\(sz1)"
        let expected = "Size2d {10.000000,10.000000}"
        XCTAssertEqual(expected, actual);
    }

}
