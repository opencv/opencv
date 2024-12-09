//
//  RotatedRectTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class RotatedRectTest: OpenCVTestCase {

    let angle: Double = 40
    let center = Point2f(x: Float(OpenCVTestCase.matSize / 2), y: Float(OpenCVTestCase.matSize / 2))
    let size = Size2f(width: Float(OpenCVTestCase.matSize / 4), height: Float(OpenCVTestCase.matSize / 2))

    func testBoundingRect() {
        let size = Size2f(width: Float(OpenCVTestCase.matSize / 2), height: Float(OpenCVTestCase.matSize / 2));
        XCTAssertEqual(size.height, size.width);
        let length = size.height;

        let angle: Double = 45
        let rr = RotatedRect(center: center, size: size, angle: angle)

        let r = rr.boundingRect()
        let halfDiagonal = length * sqrt(2) / 2

        XCTAssert(Float(r.x) == floor(center.x - halfDiagonal) && Float(r.y) == floor(center.y - halfDiagonal))

        XCTAssert((r.br().x >= ceil(center.x + halfDiagonal)) && (r.br().y >= ceil(center.y + halfDiagonal)))

        XCTAssert((r.br().x - ceil(center.x + halfDiagonal)) <= 1 && (r.br().y - ceil(center.y + halfDiagonal)) <= 1)
    }

    func testClone() {
        let rrect = RotatedRect(center: center, size: size, angle: angle)
        let clone = rrect.clone();

        XCTAssertNotNil(clone)
        XCTAssert(rrect.center == clone.center)
        XCTAssert(rrect.size == clone.size)
        XCTAssert(rrect.angle == clone.angle)
    }

    func testEqualsObject() {
        let center2 = Point2f(x: Float(OpenCVTestCase.matSize / 3), y: Float(OpenCVTestCase.matSize) / 1.5)
        let size2 = Size2f(width: Float(OpenCVTestCase.matSize / 2), height: Float(OpenCVTestCase.matSize / 4))
        let angle2:Double = 0

        let rrect1 = RotatedRect(center: center, size: size, angle: angle)
        let rrect2 = RotatedRect(center: center2, size: size2, angle: angle2)
        let rrect3 = rrect1
        let clone1 = rrect1.clone()
        let clone2 = rrect2.clone()

        XCTAssert(rrect1 == rrect3)
        XCTAssertFalse(rrect1 == rrect2)

        XCTAssert(rrect2 == clone2)
        clone2.angle = 10
        XCTAssertFalse(rrect2 == clone2)

        XCTAssert(rrect1 == clone1)

        clone1.center.x += 1
        XCTAssertFalse(rrect1 == clone1)

        clone1.center.x -= 1
        XCTAssert(rrect1 == clone1)

        clone1.size.width += 1
        XCTAssertFalse(rrect1 == clone1)

        XCTAssertFalse(rrect1 == size)
    }

    func testHashCode() {
        let rr = RotatedRect(center: center, size: size, angle: angle)
        XCTAssertEqual(rr.hash(), rr.hash())
    }

    func testPoints() {
        let rrect = RotatedRect(center: center, size: size, angle: angle);

        let p = rrect.points()

        let is_p0_irrational = (100 * p[0].x != round(100 * p[0].x)) && (100 * p[0].y != round(100 * p[0].y))
        let is_p1_irrational = (100 * p[1].x != round(100 * p[1].x)) && (100 * p[1].y != round(100 * p[1].y));
        let is_p2_irrational = (100 * p[2].x != round(100 * p[2].x)) && (100 * p[2].y != round(100 * p[2].y));
        let is_p3_irrational = (100 * p[3].x != round(100 * p[3].x)) && (100 * p[3].y != round(100 * p[3].y));

        XCTAssert(is_p0_irrational && is_p1_irrational && is_p2_irrational && is_p3_irrational)

        XCTAssert(abs((p[0].x + p[2].x) / 2 - center.x) + abs((p[0].y + p[2].y) / 2 - center.y) < OpenCVTestCase.FEPS, "Symmetric points 0 and 2")

        XCTAssert(abs((p[1].x + p[3].x) / 2 - center.x) + abs((p[1].y + p[3].y) / 2 - center.y) < OpenCVTestCase.FEPS, "Symmetric points 1 and 3")

        XCTAssert(abs((p[1].x - p[0].x) * (p[2].x - p[1].x) +
            (p[1].y - p[0].y) * (p[2].y - p[1].y)) < OpenCVTestCase.FEPS, "Orthogonal vectors 01 and 12")

        XCTAssert(abs((p[2].x - p[1].x) * (p[3].x - p[2].x) +
            (p[2].y - p[1].y) * (p[3].y - p[2].y)) < OpenCVTestCase.FEPS, "Orthogonal vectors 12 and 23");

        XCTAssert(abs((p[3].x - p[2].x) * (p[0].x - p[3].x) +
            (p[3].y - p[2].y) * (p[0].y - p[3].y)) < OpenCVTestCase.FEPS, "Orthogonal vectors 23 and 30")

        XCTAssert(abs((p[0].x - p[3].x) * (p[1].x - p[0].x) +
            (p[0].y - p[3].y) * (p[1].y - p[0].y)) < OpenCVTestCase.FEPS, "Orthogonal vectors 30 and 01")

        XCTAssert(abs((p[1].x - p[0].x) * (p[1].x - p[0].x) +
            (p[1].y - p[0].y) * (p[1].y - p[0].y) - size.height * size.height) < OpenCVTestCase.FEPS, "Length of the vector 01")

        XCTAssert(abs((p[1].x - p[2].x) * (p[1].x - p[2].x) +
            (p[1].y - p[2].y) * (p[1].y - p[2].y) - size.width * size.width) < OpenCVTestCase.FEPS, "Length of the vector 21")

        XCTAssert(abs((p[2].x - p[1].x) / size.width - Float(cos(angle * Double.pi / 180))) < OpenCVTestCase.FEPS, "Angle of the vector 21 with the axes");
    }

    func testRotatedRect() {
        let rr = RotatedRect()

        XCTAssertNotNil(rr)
        XCTAssertNotNil(rr.center)
        XCTAssertNotNil(rr.size)
        XCTAssertEqual(0.0, rr.angle)
    }

    func testRotatedRectDoubleArray() {
        let vals = [1.5, 2.6, 3.7, 4.2, 5.1]
        let rr = RotatedRect(vals: vals as [NSNumber])

        XCTAssertNotNil(rr)
        XCTAssertEqual(1.5, rr.center.x)
        XCTAssertEqual(2.6, rr.center.y)
        XCTAssertEqual(3.7, rr.size.width)
        XCTAssertEqual(4.2, rr.size.height)
        XCTAssertEqual(5.1, rr.angle)
    }

    func testRotatedRectPointSizeDouble() {
        let rr = RotatedRect(center: center, size: size, angle: 40);

        XCTAssertNotNil(rr)
        XCTAssertNotNil(rr.center)
        XCTAssertNotNil(rr.size)
        XCTAssertEqual(40.0, rr.angle);
    }

    func testSet() {
        let vals1: [Double] = []
        let r1 = RotatedRect(center: center, size: size, angle: 40);

        r1.set(vals: vals1 as [NSNumber])

        XCTAssertEqual(0, r1.angle)
        assertPoint2fEquals(Point2f(x: 0, y: 0), r1.center, OpenCVTestCase.FEPS)
        assertSize2fEquals(Size2f(width: 0, height: 0), r1.size, OpenCVTestCase.FEPS)

        let vals2 = [1, 2, 3, 4, 5]
        let r2 = RotatedRect(center: center, size: size, angle: 40)

        r2.set(vals: vals2 as [NSNumber])

        XCTAssertEqual(5, r2.angle)
        assertPoint2fEquals(Point2f(x: 1, y: 2), r2.center, OpenCVTestCase.FEPS)
        assertSize2fEquals(Size2f(width: 3, height: 4), r2.size, OpenCVTestCase.FEPS)
    }

    func testToString() {
        let actual = "\(RotatedRect(center: Point2f(x:1, y:2), size: Size2f(width:10, height:12), angle:4.5))"
        let expected = "RotatedRect {Point2f {1.000000,2.000000},Size2f {10.000000,12.000000},4.500000}"
        XCTAssertEqual(expected, actual);
    }

}
