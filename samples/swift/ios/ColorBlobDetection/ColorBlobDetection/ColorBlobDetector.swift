//
//  ColorBlobDetector.swift
//
//  Created by Giles Payne on 2020/04/04.
//

import OpenCV

public class ColorBlobDetector {
    // Lower and Upper bounds for range checking in HSV color space
    var lowerBound = Scalar(0.0)
    var upperBound = Scalar(0.0)
    // Minimum contour area in percent for contours filtering
    static let minContourArea = 0.1
    // Color radius for range checking in HSV color space
    var colorRadius = Scalar(25.0, 50.0, 50.0, 0.0)
    let spectrum = Mat()
    let contours = NSMutableArray()

    // Cache
    let pyrDownMat = Mat()
    let hsvMat = Mat()
    let mask = Mat()
    let dilatedMask = Mat()
    let hierarchy = Mat()

    func setHsvColor(hsvColor:Scalar) {
        let minH = (hsvColor.val[0].doubleValue >= colorRadius.val[0].doubleValue) ? hsvColor.val[0].doubleValue - colorRadius.val[0].doubleValue : 0
        let maxH = (hsvColor.val[0].doubleValue + colorRadius.val[0].doubleValue <= 255) ? hsvColor.val[0].doubleValue + colorRadius.val[0].doubleValue : 255

        lowerBound = Scalar(minH, hsvColor.val[1].doubleValue - colorRadius.val[1].doubleValue, hsvColor.val[2].doubleValue - colorRadius.val[2].doubleValue, 0)
        upperBound = Scalar(maxH, hsvColor.val[1].doubleValue + colorRadius.val[1].doubleValue, hsvColor.val[2].doubleValue + colorRadius.val[2].doubleValue, 255)

        let spectrumHsv = Mat(rows: 1, cols: (Int32)(maxH-minH), type:CvType.CV_8UC3);

        for j:Int32 in 0..<Int32(maxH - minH) {
            let tmp:[Double] = [Double(Int32(minH) + j), 255, 255]
            try! spectrumHsv.put(row: 0, col: j, data: tmp)
        }

        Imgproc.cvtColor(src: spectrumHsv, dst: spectrum, code: .COLOR_HSV2RGB_FULL, dstCn: 4)
    }

    func process(rgbaImage:Mat) {
        Imgproc.pyrDown(src: rgbaImage, dst: pyrDownMat)
        Imgproc.pyrDown(src: pyrDownMat, dst: pyrDownMat)

        Imgproc.cvtColor(src: pyrDownMat, dst: hsvMat, code: .COLOR_RGB2HSV_FULL)

        Core.inRange(src: hsvMat, lowerb: lowerBound, upperb: upperBound, dst: mask)
        Imgproc.dilate(src: mask, dst: dilatedMask, kernel: Mat())

        let contoursTmp = NSMutableArray()

        Imgproc.findContours(image: dilatedMask, contours: contoursTmp, hierarchy: hierarchy, mode: .RETR_EXTERNAL, method: .CHAIN_APPROX_SIMPLE)

        // Find max contour area
        var maxArea = 0.0
        for contour in contoursTmp {
            let contourMat = MatOfPoint(array: (contour as! NSMutableArray) as! [Point])
            let area = Imgproc.contourArea(contour: contourMat)
            maxArea = max(area, maxArea)
        }

        // Filter contours by area and resize to fit the original image size
        contours.removeAllObjects()
        for contour in contoursTmp {
            let contourMat = MatOfPoint(array: (contour as! NSMutableArray) as! [Point])
            if (Imgproc.contourArea(contour: contourMat) > ColorBlobDetector.minContourArea * maxArea) {
                Core.multiply(src1: contourMat, srcScalar: Scalar(4.0,4.0), dst: contourMat)
                contours.add(NSMutableArray(array: contourMat.toArray()))
            }
        }
    }
}
