//
//  ViewController.swift
//
//  Created by Giles Payne on 2020/03/02.
//

import UIKit
import OpenCV

class ViewController: UIViewController, CvVideoCameraDelegate2 {

    var isColorSelected = false
    var rgba: Mat? = nil
    let detector = ColorBlobDetector()
    let spectrum = Mat()
    var blobColorRgba = Scalar(255.0)
    var blobColorHsv = Scalar(255.0)
    let SPECTRUM_SIZE = Size(width: 200, height: 64)
    let CONTOUR_COLOR = Scalar(255.0, 0.0, 0.0, 255.0)
    var cameraHolderWidth: CGFloat = 0
    var cameraHolderHeight: CGFloat = 0

    func processImage(_ image: Mat!) {
        rgba = image
        if isColorSelected {
            detector.process(rgbaImage: image)
            let contours = detector.contours
            NSLog("Contours count: \(contours.count))")
            Imgproc.drawContours(image: image, contours: contours as! [[Point]], contourIdx: -1, color: CONTOUR_COLOR)

            let colorLabel = image.submat(rowStart: 4, rowEnd: 68, colStart: 4, colEnd: 68)
            colorLabel.setTo(scalar: blobColorRgba)

            let spectrumLabel = image.submat(rowStart: 4, rowEnd: 4 + spectrum.rows(), colStart: 70, colEnd: 70 + spectrum.cols())
            spectrum.copy(to: spectrumLabel)
        }
    }

    var camera: CvVideoCamera2? = nil

    @IBOutlet weak var cameraHolder: UIView!
    override func viewDidLoad() {
        super.viewDidLoad()
        camera = CvVideoCamera2(parentView: cameraHolder)
        camera?.rotateVideo = true
        camera?.delegate = self
        camera?.start()
    }

    override func viewDidLayoutSubviews() {
        if UIDevice.current.orientation.isLandscape {
            cameraHolderWidth = cameraHolder.bounds.height
            cameraHolderHeight = cameraHolder.bounds.width
        } else {
            cameraHolderWidth = cameraHolder.bounds.width
            cameraHolderHeight = cameraHolder.bounds.height
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let aRgba = rgba, touches.count == 1 {
            let touch = touches.first!
            let cols = CGFloat(aRgba.cols())
            let rows = CGFloat(aRgba.rows())

            let orientation = UIDevice.current.orientation
            var x = touch.location(in: cameraHolder).x
            var y = touch.location(in: cameraHolder).y
            if orientation == .landscapeLeft {
                let tempX = x
                x = cameraHolder.bounds.height - y
                y = tempX
            } else if orientation == .landscapeRight {
                let tempY = y
                y = cameraHolder.bounds.width - x
                x = tempY
            }

            x = x * (cols / cameraHolderWidth)
            y = y * (rows / cameraHolderHeight)

            if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) {
                return
            }

            let touchedRect = Rect()

            touchedRect.x = (x>4) ? Int32(x)-4 : 0;
            touchedRect.y = (y>4) ? Int32(y)-4 : 0;

            touchedRect.width = (x+4 < cols) ? Int32(x) + 4 - touchedRect.x : Int32(cols) - touchedRect.x;
            touchedRect.height = (y+4 < rows) ? Int32(y) + 4 - touchedRect.y : Int32(rows) - touchedRect.y;

            let touchedRegionRgba = aRgba.submat(roi: touchedRect)

            let touchedRegionHsv = Mat()
            Imgproc.cvtColor(src: touchedRegionRgba, dst: touchedRegionHsv, code: .COLOR_RGB2HSV_FULL)

            // Calculate average color of touched region
            blobColorHsv = Core.sum(src: touchedRegionHsv)
            let pointCount = touchedRect.width*touchedRect.height
            blobColorHsv = blobColorHsv.mul(Scalar.all(1.0/Double(pointCount)))

            blobColorRgba = convertScalarHsv2Rgba(hsvColor: blobColorHsv)

            NSLog("Touched rgba color: (\(blobColorRgba.val[0]), \(blobColorRgba.val[1]), \( blobColorRgba.val[2]), \(blobColorRgba.val[3])")

            detector.setHsvColor(hsvColor: blobColorHsv)

            Imgproc.resize(src: detector.spectrum, dst: spectrum, dsize: SPECTRUM_SIZE, fx: 0, fy: 0, interpolation: InterpolationFlags.INTER_LINEAR_EXACT.rawValue)

            isColorSelected = true
        }
    }

    func convertScalarHsv2Rgba(hsvColor:Scalar) -> Scalar {
        let pointMatRgba = Mat()
        let pointMatHsv = Mat(rows: 1, cols: 1, type: CvType.CV_8UC3, scalar: hsvColor)
        Imgproc.cvtColor(src: pointMatHsv, dst: pointMatRgba, code: .COLOR_HSV2RGB_FULL, dstCn: 4)
        let elementData = pointMatRgba.get(row: 0, col: 0)
        return Scalar(vals: elementData as [NSNumber])
    }

}
