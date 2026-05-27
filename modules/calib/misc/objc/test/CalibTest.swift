//
//  Calib3dTest.swift
//
//  Created by Giles Payne on 2020/05/26.
//

import XCTest
import OpenCV

class CalibTest: OpenCVTestCase {

    func testConstants()
    {
        // calib3d.hpp: some constants have conflict with constants from 'fisheye' namespace
        XCTAssertEqual(1, Calib.CALIB_USE_INTRINSIC_GUESS)
        XCTAssertEqual(2, Calib.CALIB_FIX_ASPECT_RATIO)
        XCTAssertEqual(4, Calib.CALIB_FIX_PRINCIPAL_POINT)
        XCTAssertEqual(8, Calib.CALIB_ZERO_TANGENT_DIST)
        XCTAssertEqual(16, Calib.CALIB_FIX_FOCAL_LENGTH)
        XCTAssertEqual(32, Calib.CALIB_FIX_K1)
        XCTAssertEqual(64, Calib.CALIB_FIX_K2)
        XCTAssertEqual(128, Calib.CALIB_FIX_K3)
        XCTAssertEqual(0x0800, Calib.CALIB_FIX_K4)
        XCTAssertEqual(0x1000, Calib.CALIB_FIX_K5)
        XCTAssertEqual(0x2000, Calib.CALIB_FIX_K6)
        XCTAssertEqual(0x4000, Calib.CALIB_RATIONAL_MODEL)
        XCTAssertEqual(0x8000, Calib.CALIB_THIN_PRISM_MODEL)
        XCTAssertEqual(0x10000, Calib.CALIB_FIX_S1_S2_S3_S4)
        XCTAssertEqual(0x40000, Calib.CALIB_TILTED_MODEL)
        XCTAssertEqual(0x80000, Calib.CALIB_FIX_TAUX_TAUY)
        XCTAssertEqual(0x100000, Calib.CALIB_USE_QR)
        XCTAssertEqual(0x200000, Calib.CALIB_FIX_TANGENT_DIST)
        XCTAssertEqual(0x100, Calib.CALIB_FIX_INTRINSIC)
        XCTAssertEqual(0x200, Calib.CALIB_SAME_FOCAL_LENGTH)
        XCTAssertEqual(0x400, Calib.CALIB_ZERO_DISPARITY)
        XCTAssertEqual((1 << 17), Calib.CALIB_USE_LU)
        XCTAssertEqual((1 << 22), Calib.CALIB_USE_EXTRINSIC_GUESS)
    }

}
