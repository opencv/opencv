import XCTest
import OpenCV

class ObjdetectTest: OpenCVTestCase {

    var size = Size()

    override func setUp() {
        super.setUp()
        size = Size(width: 3, height: 3)
    }

    override func tearDown() {
        super.tearDown()
    }

    func testFindChessboardCornersMatSizeMat() {
        let patternSize = Size(width: 9, height: 6)
        let corners = MatOfPoint2f()
        Calib.findChessboardCorners(image: grayChess, patternSize: patternSize, corners: corners)
        XCTAssertFalse(corners.empty())
    }

    func testFindChessboardCornersMatSizeMatInt() {
        let patternSize = Size(width: 9, height: 6)
        let corners = MatOfPoint2f()
        Calib.findChessboardCorners(image: grayChess, patternSize: patternSize, corners: corners, flags: Calib.CALIB_CB_ADAPTIVE_THRESH + Calib.CALIB_CB_NORMALIZE_IMAGE + Calib.CALIB_CB_FAST_CHECK)
        XCTAssertFalse(corners.empty())
    }

    func testFind4QuadCornerSubpix() {
        let patternSize = Size(width: 9, height: 6)
        let corners = MatOfPoint2f()
        let region_size = Size(width: 5, height: 5)
        Calib.findChessboardCorners(image: grayChess, patternSize: patternSize, corners: corners)
        Calib.find4QuadCornerSubpix(img: grayChess, corners: corners, region_size: region_size)
        XCTAssertFalse(corners.empty())
    }

    func testFindCirclesGridMatSizeMat() {
        let size = 300
        let img = Mat(rows:Int32(size), cols:Int32(size), type:CvType.CV_8U)
        img.setTo(scalar: Scalar(255))
        let centers = Mat()

        XCTAssertFalse(Calib.findCirclesGrid(image: img, patternSize: Size(width: 5, height: 5), centers: centers))

        for i in 0..<5 {
            for j in 0..<5 {
                let x = Int32(size * (2 * i + 1) / 10)
                let y = Int32(size * (2 * j + 1) / 10)
                let pt = Point(x: x, y: y)
                Imgproc.circle(img: img, center: pt, radius: 10, color: Scalar(0), thickness: -1)
            }
        }

        XCTAssert(Calib.findCirclesGrid(image: img, patternSize:Size(width:5, height:5), centers:centers))

        XCTAssertEqual(25, centers.rows())
        XCTAssertEqual(1, centers.cols())
        XCTAssertEqual(CvType.CV_32FC2, centers.type())
    }

    func testFindCirclesGridMatSizeMatInt() {
        let size:Int32 = 300
        let img = Mat(rows:size, cols: size, type: CvType.CV_8U)
        img.setTo(scalar: Scalar(255))
        let centers = Mat()

        XCTAssertFalse(Calib.findCirclesGrid(image: img, patternSize: Size(width: 3, height: 5), centers: centers, flags: Calib.CALIB_CB_CLUSTERING | Calib.CALIB_CB_ASYMMETRIC_GRID))

        let step = size * 2 / 15
        let offsetx = size / 6
        let offsety = (size - 4 * step) / 2
        for i:Int32 in 0...2 {
            for j:Int32 in 0...4 {
                let pt = Point(x: offsetx + (2 * i + j % 2) * step, y: offsety + step * j)
                Imgproc.circle(img: img, center: pt, radius: 10, color: Scalar(0), thickness: -1)
            }
        }

        XCTAssert(Calib.findCirclesGrid(image: img, patternSize: Size(width: 3, height: 5), centers: centers, flags: Calib.CALIB_CB_CLUSTERING | Calib.CALIB_CB_ASYMMETRIC_GRID))

        XCTAssertEqual(15, centers.rows())
        XCTAssertEqual(1, centers.cols())
        XCTAssertEqual(CvType.CV_32FC2, centers.type())
    }
}
