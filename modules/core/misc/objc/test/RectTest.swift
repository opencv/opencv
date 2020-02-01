//
//  RectTest.swift
//  StitchAppTests
//
//  Created by Giles Payne on 2020/01/31.
//  Copyright © 2020 Xtravision. All rights reserved.
//

import XCTest
import StitchApp

class RectTest: OpenCVTestCase {

    let r = CVRect()
    let rect = CVRect(x: 0, y: 0, width: 10, height: 10)

    func testArea() {
        let area = rect.area()
        XCTAssertEqual(100.0, area)
    }

    func testBr() {
        let p_br = rect.br()
        let truth = CVPoint(x: 10, y: 10)
        XCTAssertEqual(truth, p_br)
    }

    func testClone() {
        let r = rect.clone()
        XCTAssertEqual(rect, r)
    }

    func testContains() {
        let rect = CVRect(x: 0, y: 0, width: 10, height: 10)

        let p_inner = CVPoint(x: 5, y: 5)
        let p_outer = CVPoint(x: 5, y: 55)
        let p_bl = CVPoint(x: 0, y: 0)
        let p_br = CVPoint(x: 10, y: 0)
        let p_tl = CVPoint(x: 0, y: 10)
        let p_tr = CVPoint(x: 10, y: 10)

        XCTAssert(rect.contains(p_inner))
        XCTAssert(rect.contains(p_bl))

        XCTAssertFalse(rect.contains(p_outer))
        XCTAssertFalse(rect.contains(p_br))
        XCTAssertFalse(rect.contains(p_tl))
        XCTAssertFalse(rect.contains(p_tr))
    }

    func testEqualsObject() {
        var flag = rect == r
        XCTAssertFalse(flag)

        let r = rect.clone()
        flag = rect == r
        XCTAssert(flag)
    }

    func testHashCode() {
        XCTAssertEqual(rect.hash(), rect.hash())
    }

    func testRect() {
        let r = CVRect()

        XCTAssertEqual(0, r.x)
        XCTAssertEqual(0, r.y)
        XCTAssertEqual(0, r.width)
        XCTAssertEqual(0, r.height)
    }

    func testRectDoubleArray() {
        let vals:[Double] = [1, 3, 5, 2]
        let r = CVRect(vals: vals as [NSNumber])
        
        XCTAssertEqual(1, r.x)
        XCTAssertEqual(3, r.y)
        XCTAssertEqual(5, r.width)
        XCTAssertEqual(2, r.height)
    }

    func testRectIntIntIntInt() {
        let rect = CVRect(x: 1, y: 3, width: 5, height: 2)

        XCTAssertNotNil(rect)
        XCTAssertEqual(1, rect.x)
        XCTAssertEqual(3, rect.y)
        XCTAssertEqual(5, rect.width)
        XCTAssertEqual(2, rect.height)
    }

    func testRectPointPoint() {
        let p1 = CVPoint(x:4, y:4)
        let p2 = CVPoint(x: 2, y: 3)

        let r = CVRect(point: p1, point: p2)
        XCTAssertNotNil(r);
        XCTAssertEqual(2, r.x);
        XCTAssertEqual(3, r.y);
        XCTAssertEqual(2, r.width);
        XCTAssertEqual(1, r.height);
    }

    func testRectPointSize() {
        let p1 = CVPoint(x: 4, y: 4)
        let sz = CVSize(width: 3, height: 1)
        let r = CVRect(point: p1, size: sz)

        XCTAssertEqual(4, r.x)
        XCTAssertEqual(4, r.y)
        XCTAssertEqual(3, r.width)
        XCTAssertEqual(1, r.height)
    }

    func testSet() {
        let vals1:[Double] = []
        let r1 = CVRect(vals:vals1 as [NSNumber])

        XCTAssertEqual(0, r1.x)
        XCTAssertEqual(0, r1.y)
        XCTAssertEqual(0, r1.width)
        XCTAssertEqual(0, r1.height)

        let vals2:[Double] = [2, 2, 10, 5]
        let r = CVRect(vals: vals2 as [NSNumber])

        XCTAssertEqual(2, r.x)
        XCTAssertEqual(2, r.y)
        XCTAssertEqual(10, r.width)
        XCTAssertEqual(5, r.height)
    }

    func testSize() {
        let s1 = CVSize(width: 0, height: 0)
        XCTAssertEqual(s1, r.size())

        let s2 = CVSize(width: 10, height: 10)
        XCTAssertEqual(s2, rect.size())
    }

    func testTl() {
        let p_tl = rect.tl()
        let truth = CVPoint(x: 0, y: 0)
        XCTAssertEqual(truth, p_tl)
    }

    func testToString() {
        let actual = "\(rect)"
        let expected = "CVRect {0,0,10,10}"
        XCTAssertEqual(expected, actual);
    }

}
