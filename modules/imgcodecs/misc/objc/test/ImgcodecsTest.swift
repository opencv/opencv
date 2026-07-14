//
//  Imgcodecs.swift
//
//  Created by Giles Payne on 2020/02/10.
//

import XCTest
import OpenCV

class ImgcodecsTest: OpenCVTestCase {

    let LENA_PATH = Bundle(for: ImgcodecsTest.self).path(forResource:"lena", ofType:"png", inDirectory:"resources")!

    func testImencodeStringMatListOfByte() {
        var buff = [UInt8]()
        XCTAssert(Imgcodecs.imencode(ext: ".jpg", img: gray127, buf: &buff))
        XCTAssertFalse(0 == buff.count)
    }

    func testImencodeStringMatListOfByteListOfInteger() {
        let params40:[Int32] = [ImwriteFlags.IMWRITE_JPEG_QUALITY.rawValue, 40]
        let params90:[Int32] = [ImwriteFlags.IMWRITE_JPEG_QUALITY.rawValue, 90]

        var buff40 = [UInt8]()
        var buff90 = [UInt8]()

        XCTAssert(Imgcodecs.imencode(ext: ".jpg", img: rgbLena, buf: &buff40, params: params40))
        XCTAssert(Imgcodecs.imencode(ext: ".jpg", img: rgbLena, buf: &buff90, params: params90))

        XCTAssert(buff40.count > 0)
        XCTAssert(buff40.count < buff90.count)
    }

    func testImreadString() {
        dst = Imgcodecs.imread(filename: LENA_PATH)
        XCTAssertFalse(dst.empty())
        XCTAssertEqual(3, dst.channels())
        XCTAssert(512 == dst.cols())
        XCTAssert(512 == dst.rows())
    }

    func testImreadStringInt() {
        dst = Imgcodecs.imread(filename: LENA_PATH, flags: 0)
        XCTAssertFalse(dst.empty());
        XCTAssertEqual(1, dst.channels());
        XCTAssert(512 == dst.cols());
        XCTAssert(512 == dst.rows());
    }

}
