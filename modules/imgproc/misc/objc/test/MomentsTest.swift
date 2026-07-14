//
//  MomentsTest.swift
//
//  Created by Giles Payne on 2020/02/10.
//

import XCTest
import OpenCV

class MomentsTest: XCTestCase {

    func testAll() {
        let data = Mat(rows: 3,cols: 3, type: CvType.CV_8UC1, scalar: Scalar(1))
        data.row(1).setTo(scalar: Scalar(5))
        let res = Imgproc.moments(array: data)
        XCTAssertEqual(res.m00, 21.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m10, 21.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m01, 21.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m20, 35.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m11, 21.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m02, 27.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m30, 63.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m21, 35.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m12, 27.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.m03, 39.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.mu20, 14.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.mu11, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.mu02, 6.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.mu30, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.mu21, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.mu12, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.mu03, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.nu20, 0.031746031746031744, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.nu11, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.nu02, 0.013605442176870746, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.nu30, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.nu21, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.nu12, 0.0, accuracy: OpenCVTestCase.EPS);
        XCTAssertEqual(res.nu03, 0.0, accuracy: OpenCVTestCase.EPS);
    }

}
