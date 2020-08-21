//
//  Point3Test.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class Point3Test: OpenCVTestCase {

    let p1 = Point3i(x: 2, y: 2, z: 2)
    let p2 = Point3i(x: 1, y: 1, z: 1)

    func testClone() {
        let truth = Point3i(x: 1, y: 1, z: 1)
        let p1 = truth.clone()
        XCTAssertEqual(truth, p1)
    }

    func testCross() {
        let dstPoint = p1.cross(p2)
        let truth = Point3i(x: 0, y: 0, z: 0)
        XCTAssertEqual(truth, dstPoint)
    }

    func testDot() {
        let result = p1.dot(p2)
        XCTAssertEqual(6.0, result)
    }

    func testEqualsObject() {
        var flag = p1 == p1
        XCTAssert(flag)

        flag = p1 == p2
        XCTAssertFalse(flag)
    }

    func testHashCode() {
        XCTAssertEqual(p1.hash(), p1.hash())
    }

    func testPoint3() {
        let p1 = Point3i()

        XCTAssertNotNil(p1)
        XCTAssert(0 == p1.x)
        XCTAssert(0 == p1.y)
        XCTAssert(0 == p1.z)
    }

    func testPoint3DoubleArray() {
        let vals:[Double] = [1, 2, 3]
        let p1 = Point3i(vals: vals as [NSNumber])

        XCTAssert(1 == p1.x)
        XCTAssert(2 == p1.y)
        XCTAssert(3 == p1.z)
    }

    func testPoint3DoubleDoubleDouble() {
        let p1 = Point3i(x: 1, y: 2, z: 3)

        XCTAssertEqual(1, p1.x)
        XCTAssertEqual(2, p1.y)
        XCTAssertEqual(3, p1.z)
    }

    func testPoint3Point() {
        let p = Point(x: 2, y: 3)
        let p1 = Point3i(point: p)

        XCTAssertEqual(2, p1.x)
        XCTAssertEqual(3, p1.y)
        XCTAssertEqual(0, p1.z)
    }

    func testSet() {
        let vals1:[Double] = []
        p1.set(vals: vals1 as [NSNumber]);

        XCTAssertEqual(0, p1.x)
        XCTAssertEqual(0, p1.y)
        XCTAssertEqual(0, p1.z)

        let vals2 = [3, 6, 10]
        p1.set(vals: vals2 as [NSNumber])

        XCTAssertEqual(3, p1.x)
        XCTAssertEqual(6, p1.y)
        XCTAssertEqual(10, p1.z)
    }

    func testToString() {
        let actual = "\(p1)"
        let expected = "Point3i {2,2,2}"
        XCTAssertEqual(expected, actual)
    }

}
