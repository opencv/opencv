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
        let buff = ByteVector()
        XCTAssert(Imgcodecs.imencode(ext: ".jpg", img: gray127, buf: buff))
        XCTAssertFalse(0 == buff.length)
    }

    func testImencodeStringMatListOfByteListOfInteger() {
        let params40 = IntVector([ImwriteFlags.IMWRITE_JPEG_QUALITY.rawValue, 40])
        let params90 = IntVector([ImwriteFlags.IMWRITE_JPEG_QUALITY.rawValue, 90])

        let buff40 = ByteVector()
        let buff90 = ByteVector()

        XCTAssert(Imgcodecs.imencode(ext: ".jpg", img: rgbLena, buf: buff40, params: params40))
        XCTAssert(Imgcodecs.imencode(ext: ".jpg", img: rgbLena, buf: buff90, params: params90))

        XCTAssert(buff40.length > 0)
        XCTAssert(buff40.length < buff90.length)
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
