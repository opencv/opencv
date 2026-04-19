//
//  FeaturesTest.swift
//
//  Created by Alexander Smorkalov on 2026/13/04.
//

import XCTest
import OpenCV

class ImgprocTest: OpenCVTestCase {
    func testGoodFeaturesToTrackMatListOfPointIntDoubleDouble() {
        let src = gray0
        Imgproc.rectangle(img: src, pt1: Point(x: 2, y: 2), pt2: Point(x: 8, y: 8), color: Scalar(100), thickness: -1)
        var lp = [Point]()

        Imgproc.goodFeaturesToTrack(image: src, corners: &lp, maxCorners: 100, qualityLevel: 0.01, minDistance: 3)

        XCTAssertEqual(4, lp.count)
    }

    func testGoodFeaturesToTrackMatListOfPointIntDoubleDoubleMatIntBooleanDouble() {
        let src = gray0
        Imgproc.rectangle(img: src, pt1: Point(x: 2, y: 2), pt2: Point(x: 8, y: 8), color: Scalar(100), thickness: -1)
        var lp = [Point]()

        Imgproc.goodFeaturesToTrack(image: src, corners: &lp, maxCorners: 100, qualityLevel: 0.01, minDistance: 3, mask: gray1, blockSize: 4, gradientSize: 3, useHarrisDetector: true, k: 0)

        XCTAssertEqual(4, lp.count)
    }
}
