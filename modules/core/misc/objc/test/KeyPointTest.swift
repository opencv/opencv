//
//  KeyPointTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class KeyPointTest: OpenCVTestCase {

    let angle:Float = 30
    let classId:Int32 = 1
    let octave:Int32 = 1
    let response:Float = 2.0
    let size:Float = 3.0
    let x:Float = 1.0
    let y:Float = 2.0

    func testKeyPoint() {
        let keyPoint = KeyPoint()
        assertPoint2fEquals(Point2f(x: 0, y: 0), keyPoint.pt, OpenCVTestCase.FEPS)
    }

    func testKeyPointFloatFloatFloat() {
        let keyPoint = KeyPoint(x: x, y: y, size: size)
        assertPoint2fEquals(Point2f(x: 1, y: 2), keyPoint.pt, OpenCVTestCase.FEPS)
    }

    func testKeyPointFloatFloatFloatFloat() {
        let keyPoint = KeyPoint(x: x, y: y, size: size, angle: 10.0)
        XCTAssertEqual(10.0, keyPoint.angle);
    }

    func testKeyPointFloatFloatFloatFloatFloat() {
        let keyPoint = KeyPoint(x: x, y: y, size: size, angle: 1.0, response: 1.0)
        XCTAssertEqual(1.0, keyPoint.response)
    }

    func testKeyPointFloatFloatFloatFloatFloatInt() {
        let keyPoint = KeyPoint(x: x, y: y, size: size, angle: 1.0, response: 1.0, octave: 1)
        XCTAssertEqual(1, keyPoint.octave)
    }

    func testKeyPointFloatFloatFloatFloatFloatIntInt() {
        let keyPoint = KeyPoint(x: x, y: y, size: size, angle: 1.0, response: 1.0, octave: 1, classId: 1)
        XCTAssertEqual(1, keyPoint.classId)
    }

    func testToString() {
        let keyPoint = KeyPoint(x: x, y: y, size: size, angle: angle, response: response, octave: octave, classId: classId)

        let actual = "\(keyPoint)"

        let expected = "KeyPoint { pt: Point2f {1.000000,2.000000}, size: 3.000000, angle: 30.000000, response: 2.000000, octave: 1, classId: 1}"
        XCTAssertEqual(expected, actual)
    }

}
