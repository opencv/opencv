//
//  PointTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class PointTest: OpenCVTestCase {

    let p1 = Point2d(x: 2, y: 2)
    let p2 = Point2d(x: 1, y: 1)

    func testClone() {
        let truth = Point2d(x: 1, y: 1)
        let dstPoint = truth.clone()
        XCTAssertEqual(truth, dstPoint);
    }

    func testDot() {
        let result = p1.dot(p2);
        XCTAssertEqual(4.0, result)
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

    func testInside() {
        let rect =  Rect2d(x: 0, y: 0, width: 5, height: 3)
        XCTAssert(p1.inside(rect))

        let p2 = Point2d(x: 3, y: 3)
        XCTAssertFalse(p2.inside(rect))
    }

    func testPoint() {
        let p = Point2d()

        XCTAssertNotNil(p)
        XCTAssertEqual(0.0, p.x)
        XCTAssertEqual(0.0, p.y)
    }

    func testPointDoubleArray() {
        let vals:[Double] =  [2, 4]
        let p = Point2d(vals: vals as [NSNumber])

        XCTAssertEqual(2.0, p.x);
        XCTAssertEqual(4.0, p.y);
    }

    func testPointDoubleDouble() {
        let p1 = Point2d(x: 7, y: 5)

        XCTAssertNotNil(p1)
        XCTAssertEqual(7.0, p1.x);
        XCTAssertEqual(5.0, p1.y);
    }

    func testSet() {
        let vals1:[Double] = []
        p1.set(vals: vals1 as [NSNumber])
        XCTAssertEqual(0.0, p1.x)
        XCTAssertEqual(0.0, p1.y)

        let vals2 = [ 6, 10 ]
        p2.set(vals: vals2 as [NSNumber])
        XCTAssertEqual(6.0, p2.x)
        XCTAssertEqual(10.0, p2.y)
    }

    func testToString() {
        let actual = "\(p1)"
        let expected = "Point2d {2.000000,2.000000}"
        XCTAssertEqual(expected, actual)
    }

}
