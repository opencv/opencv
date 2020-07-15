//
//  Subdiv2DTest.swift
//
//  Created by Giles Payne on 2020/02/10.
//

import XCTest
import OpenCV

class Subdiv2DTest: OpenCVTestCase {

    func testGetTriangleList() {
        let s2d = Subdiv2D(rect: Rect(x: 0, y: 0, width: 50, height: 50))
        s2d.insert(pt: Point2f(x: 10, y: 10))
        s2d.insert(pt: Point2f(x: 20, y: 10))
        s2d.insert(pt: Point2f(x: 20, y: 20))
        s2d.insert(pt: Point2f(x: 10, y: 20))
        let triangles = NSMutableArray()
        s2d.getTriangleList(triangleList: triangles)
        XCTAssertEqual(2, triangles.count)
    }

}
