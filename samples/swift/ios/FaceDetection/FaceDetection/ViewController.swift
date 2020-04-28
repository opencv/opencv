//
//  ViewController.swift
//
//  Created by Giles Payne on 2020/03/02.
//

import UIKit
import OpenCV

extension Rect {
    func rotateClockwise(parentHeight:Int32) {
        let tmpX = self.x
        self.x = parentHeight - (self.y + self.height)
        self.y = tmpX
        swapDims()
    }

    func rotateCounterclockwise(parentWidth:Int32) {
        let tmpY = self.y
        self.y = parentWidth - (self.x + self.width)
        self.x = tmpY
        swapDims()
    }

    func swapDims() {
        let tmpWidth = self.width
        self.width = self.height
        self.height = tmpWidth
    }
}

class ViewController: UIViewController, CvVideoCameraDelegate2 {

    let swiftDetector = CascadeClassifier(filename: Bundle(for: ViewController.self).path(forResource:"lbpcascade_frontalface", ofType:"xml")!)
    let nativeDetector = DetectionBasedTracker(cascadeName: Bundle(for: ViewController.self).path(forResource:"lbpcascade_frontalface", ofType:"xml")!, minFaceSize: 0)
    var rgba: Mat? = nil
    var gray: Mat = Mat()
    var relativeFaceSize: Float = 0.2
    var absoluteFaceSize: Int32 = 0
    let FACE_RECT_COLOR = Scalar(0.0, 255.0, 0.0, 255.0)
    let FACE_RECT_THICKNESS: Int32 = 4

    func processImage(_ image: Mat!) {
        let orientation = UIDevice.current.orientation
        switch orientation {
        case .landscapeLeft:
            rgba = Mat()
            Core.rotate(src: image, dst: rgba!, rotateCode: .ROTATE_90_COUNTERCLOCKWISE)
        case .landscapeRight:
            rgba = Mat()
            Core.rotate(src: image, dst: rgba!, rotateCode: .ROTATE_90_CLOCKWISE)
        default:
            rgba = image
        }

        Imgproc.cvtColor(src: rgba!, dst: gray, code: .COLOR_RGB2GRAY)

        if (absoluteFaceSize == 0) {
            let height = gray.rows()
            if (round(Float(height) * relativeFaceSize) > 0) {
                absoluteFaceSize = Int32(round(Float(height) * relativeFaceSize))
            }
        }

        let faces = NSMutableArray()

        swiftDetector.detectMultiScale(image: gray, objects: faces, scaleFactor: 1.1, minNeighbors: Int32(2), flags: Int32(2), minSize: Size(width: absoluteFaceSize, height: absoluteFaceSize), maxSize: Size())
        //nativeDetector!.detect(gray, faces: faces)

        for face in faces as! [Rect] {
            if orientation == .landscapeLeft {
                face.rotateClockwise(parentHeight: gray.rows())
            } else if orientation == .landscapeRight {
                face.rotateCounterclockwise(parentWidth: gray.cols())
            }
            Imgproc.rectangle(img: image, pt1: face.tl(), pt2: face.br(), color: FACE_RECT_COLOR, thickness: FACE_RECT_THICKNESS)
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
}
