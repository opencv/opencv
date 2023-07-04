//
//  CvTypeTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class CvTypeTest: OpenCVTestCase {

    func testMakeType() {
        XCTAssertEqual(CvType.CV_8UC4, CvType.make(CvType.CV_8U, channels: 4))
    }

    func testCV_8UC() {
        XCTAssertEqual(CvType.CV_8UC4, CvType.CV_8UC(4))
    }

    func testCV_8SC() {
        XCTAssertEqual(CvType.CV_8SC4, CvType.CV_8SC(4))
    }

    func testCV_16UC() {
        XCTAssertEqual(CvType.CV_16UC4, CvType.CV_16UC(4))
    }

    func testCV_16SC() {
        XCTAssertEqual(CvType.CV_16SC4, CvType.CV_16SC(4))
    }

    func testCV_32SC() {
        XCTAssertEqual(CvType.CV_32SC4, CvType.CV_32SC(4))
    }

    func testCV_32FC() {
        XCTAssertEqual(CvType.CV_32FC4, CvType.CV_32FC(4))
    }

    func testCV_64FC() {
        XCTAssertEqual(CvType.CV_64FC4, CvType.CV_64FC(4))
    }

    func testCV_16FC() {
        XCTAssertEqual(CvType.CV_16FC1, CvType.CV_16FC(1))
        XCTAssertEqual(CvType.CV_16FC2, CvType.CV_16FC(2))
        XCTAssertEqual(CvType.CV_16FC3, CvType.CV_16FC(3))
        XCTAssertEqual(CvType.CV_16FC4, CvType.CV_16FC(4))
    }

    func testChannels() {
        XCTAssertEqual(1, CvType.channels(CvType.CV_64F))
    }

    func testDepth() {
        XCTAssertEqual(CvType.CV_64F, CvType.depth(CvType.CV_64FC3))
    }

    func testIsInteger() {
        XCTAssertFalse(CvType.isInteger(CvType.CV_32FC3));
        XCTAssert(CvType.isInteger(CvType.CV_16S));
    }

    func testELEM_SIZE() {
        XCTAssertEqual(3 * 8, CvType.elemSize(CvType.CV_64FC3));
        XCTAssertEqual(3 * 2, CvType.elemSize(CvType.CV_16FC3));
    }

    func testTypeToString() {
        XCTAssertEqual("CV_32FC1", CvType.type(toString: CvType.CV_32F));
        XCTAssertEqual("CV_32FC3", CvType.type(toString: CvType.CV_32FC3));
        XCTAssertEqual("CV_32FC(128)", CvType.type(toString: CvType.CV_32FC(128)));
    }

}
