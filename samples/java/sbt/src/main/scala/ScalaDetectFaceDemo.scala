import org.opencv.core.Core
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import reflect._

/*
 * Detects faces in an image, draws boxes around them, and writes the results
 * to "scalaFaceDetection.png".
 */
object ScalaDetectFaceDemo {
  def run() {
    println(s"\nRunning ${classTag[this.type].toString.replace("$", "")}")

    // Create a face detector from the cascade file in the resources directory.
    val faceDetector = new CascadeClassifier(getClass.getResource("/lbpcascade_frontalface.xml").getPath)
    val image = Imgcodecs.imread(getClass.getResource("/AverageMaleFace.jpg").getPath)

    // Detect faces in the image.
    // MatOfRect is a special container class for Rect.
    val faceDetections = new MatOfRect
    faceDetector.detectMultiScale(image, faceDetections)

    println(s"Detected ${faceDetections.toArray.size} faces")

    // Draw a bounding box around each face.
    for (rect <- faceDetections.toArray) {
      Imgproc.rectangle(
        image,
        new Point(rect.x, rect.y),
        new Point(rect.x + rect.width,
          rect.y + rect.height),
        new Scalar(0, 255, 0))
    }

    // Save the visualized detection.
    val filename = "scalaFaceDetection.png"
    println(s"Writing ${filename}")
    assert(Imgcodecs.imwrite(filename, image))
  }
}
