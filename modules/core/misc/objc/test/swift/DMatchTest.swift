//
//  DMatchTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import OpenCV

class DMatchTest: OpenCVTestCase {

    func testDMatchIntIntFloat() {
        let dm1 = DMatch(queryIdx: 1, trainIdx: 4, distance: 4.0)

        XCTAssertEqual(1, dm1.queryIdx)
        XCTAssertEqual(4, dm1.trainIdx)
        XCTAssertEqual(4.0, dm1.distance)
    }

    func testDMatchIntIntIntFloat() {
        let dm2 = DMatch(queryIdx: 2, trainIdx: 6, imgIdx: -1, distance: 8.0)

        XCTAssertEqual(2, dm2.queryIdx)
        XCTAssertEqual(6, dm2.trainIdx)
        XCTAssertEqual(-1, dm2.imgIdx)
        XCTAssertEqual(8.0, dm2.distance)
    }

    func testLessThan() {
        let dm1 = DMatch(queryIdx: 1, trainIdx: 4, distance: 4.0)
        let dm2 = DMatch(queryIdx: 2, trainIdx: 6, imgIdx: -1, distance: 8.0)
        XCTAssert(dm1.lessThan(dm2))
    }

    func testToString() {
        let dm2 = DMatch(queryIdx: 2, trainIdx: 6, imgIdx: -1, distance: 8.0)

        let actual = "\(dm2)"

        let expected = "DMatch { queryIdx: 2, trainIdx: 6, imgIdx: -1, distance: 8.000000}"
        XCTAssertEqual(expected, actual)
    }

}
