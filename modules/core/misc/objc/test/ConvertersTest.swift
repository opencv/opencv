//
//  ConvertersTest.swift
//
//  Created by Giles Payne on 2020/06/01.
//

import XCTest
import OpenCV

class ConvertersTest: OpenCVTestCase {

    func testPoint2iToMat() {
        let pointsIn = [Point(x:3, y:4), Point(x:6, y:7), Point(x:7, y:6), Point(x:-78, y:14), Point(x:-93, y:700)]
        let pointsOut = Converters.Mat_to_vector_Point(Converters.vector_Point_to_Mat(pointsIn))
        XCTAssertEqual(pointsIn, pointsOut)
    }

    func testPoint2fToMat() {
        let pointsIn = [Point2f(x:3.8, y:4.2), Point2f(x:6.01, y:7), Point2f(x:7, y:6), Point2f(x:-78, y:14), Point2f(x:-93, y:700)]
        let pointsOut = Converters.Mat_to_vector_Point2f(Converters.vector_Point2f_to_Mat(pointsIn))
        XCTAssertEqual(pointsIn, pointsOut)
    }

    func testPoint2dToMat() {
        let pointsIn = [Point2d(x:3.80004, y:73.2), Point2d(x:16.01, y:7.1111), Point2d(x:3.14, y:6), Point2d(x:-78, y:14)]
        let pointsOut = Converters.Mat_to_vector_Point2d(Converters.vector_Point2d_to_Mat(pointsIn))
        XCTAssertEqual(pointsIn, pointsOut)
    }

    func testPoint3iToMat() {
        let pointsIn = [Point3i(x:3, y:4, z:2), Point3i(x:6, y:7, z:1), Point3i(x:7, y:6, z:9), Point3i(x:-78, y:14, z:0), Point3i(x:-93, y:700, z:54)]
        let pointsOut = Converters.Mat_to_vector_Point3i(Converters.vector_Point3i_to_Mat(pointsIn))
        XCTAssertEqual(pointsIn, pointsOut)
    }

    func testPoint3fToMat() {
        let pointsIn = [Point3f(x:3.8, y:4.2, z:1200), Point3f(x:6.01, y:7, z: 12), Point3f(x:7, y:6, z:8.88128), Point3f(x:-78, y:14, z:-1), Point3f(x:-93, y:700, z:200)]
        let pointsOut = Converters.Mat_to_vector_Point3f(Converters.vector_Point3f_to_Mat(pointsIn))
        XCTAssertEqual(pointsIn, pointsOut)
    }

    func testPoint3dToMat() {
        let pointsIn = [Point3d(x:3.80004, y:73.2, z:1), Point3d(x:16.01, y:7.1111, z:2), Point3d(x:3.14, y:6, z:3), Point3d(x:-78, y:14, z:4)]
        let pointsOut = Converters.Mat_to_vector_Point3d(Converters.vector_Point3d_to_Mat(pointsIn))
        XCTAssertEqual(pointsIn, pointsOut)
    }

    func testFloatToMat() {
        let floatsIn:[Float] = [23.8, 999.89, 93, 0.9, 12]
        let floatsOut = Converters.Mat_to_vector_float(Converters.vector_float_to_Mat(floatsIn as [NSNumber])) as! [Float]
        XCTAssertEqual(floatsIn, floatsOut)
    }

    func testIntToMat() {
        let intsIn:[Int32] = [23, 999, -93, 0, 4412]
        let intsOut = Converters.Mat_to_vector_int(Converters.vector_int_to_Mat(intsIn as [NSNumber])) as! [Int32]
        XCTAssertEqual(intsIn, intsOut)
    }

    func testCharToMat() {
        let charsIn:[Int8] = [23, -23, 93, 0, -127]
        let charsOut = Converters.Mat_to_vector_char(Converters.vector_char_to_Mat(charsIn as [NSNumber])) as! [Int8]
        XCTAssertEqual(charsIn, charsOut)
    }

    func testUCharToMat() {
        let ucharsIn:[UInt8] = [23, 190, 93, 0, 255]
        let ucharsOut = Converters.Mat_to_vector_uchar(Converters.vector_uchar_to_Mat(ucharsIn as [NSNumber])) as! [UInt8]
        XCTAssertEqual(ucharsIn, ucharsOut)
    }

    func testDoubleToMat() {
        let doublesIn:[Double] = [23.8, 999.89, 93, 0.9, 12]
        let doublesOut = Converters.Mat_to_vector_double(Converters.vector_double_to_Mat(doublesIn as [NSNumber])) as! [Double]
        XCTAssertEqual(doublesIn, doublesOut)
    }

    func testRectToMat() {
        let rectsIn = [Rect(x: 0, y: 0, width: 3, height: 4), Rect(x: 10, y: 23, width: 7, height: 6), Rect(x: 0, y: 1111110, width: 1, height: 4000)]
        let rectsOut = Converters.Mat_to_vector_Rect(Converters.vector_Rect_to_Mat(rectsIn))
        XCTAssertEqual(rectsIn, rectsOut)
    }

    func testRect2dToMat() {
        let rectsIn = [Rect2d(x: 0.001, y: 0.00001, width: 3.2, height: 4.556555555), Rect2d(x: 10.009, y: -6623, width: 7.9, height: 6), Rect2d(x: 0, y: 1111.33110, width: 0.99999, height: 3999.999)]
        let rectsOut = Converters.Mat_to_vector_Rect2d(Converters.vector_Rect2d_to_Mat(rectsIn))
        XCTAssertEqual(rectsIn, rectsOut)
    }

    func testKeyPointToMat() {
        let keyPointsIn = [KeyPoint(x: 8.99, y: 9.00, size: 3, angle: 3.23, response: 0.001, octave: 3, classId: 5), KeyPoint(x: 58.99, y: 9.488, size: 3.4, angle: 2.223, response: 0.006, octave: 4, classId: 7), KeyPoint(x: 7, y: 9.003, size: 12, angle: -3.23, response: 0.02, octave: 1, classId: 8)]
        let keyPointsOut = Converters.Mat_to_vector_KeyPoint(Converters.vector_KeyPoint_to_Mat(keyPointsIn))
        XCTAssertEqual(keyPointsIn, keyPointsOut)
    }

    func testDMatchToMat() {
        let dmatchesIn = [DMatch(queryIdx: 2, trainIdx: 4, distance: 0.7), DMatch(queryIdx: 3, trainIdx: 7, distance: 0.1), DMatch(queryIdx: 4, trainIdx: 8, distance: 0.01)]
        let dmatchesOut = Converters.Mat_to_vector_DMatch(Converters.vector_DMatch_to_Mat(dmatchesIn))
        XCTAssertEqual(dmatchesIn, dmatchesOut)
    }

    func testRotatedRectToMat() {
        let rectsIn = [RotatedRect(center: Point2f(x: 0.4, y: 0.9), size: Size2f(width: 3.0, height: 8.9), angle: 0.3342)]
        let rectsOut = Converters.Mat_to_vector_RotatedRect(Converters.vector_RotatedRect_to_Mat(rectsIn))
        XCTAssertEqual(rectsIn[0].center, rectsOut[0].center)
        XCTAssertEqual(rectsIn[0].size, rectsOut[0].size)
        XCTAssertEqual(rectsIn[0].angle, rectsOut[0].angle, accuracy: OpenCVTestCase.EPS)
    }
}
