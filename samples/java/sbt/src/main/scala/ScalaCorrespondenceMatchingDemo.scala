import org.opencv.highgui.Highgui
import org.opencv.features2d.DescriptorExtractor
import org.opencv.features2d.Features2d
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Mat
import org.opencv.features2d.FeatureDetector
import org.opencv.features2d.DescriptorMatcher
import org.opencv.core.MatOfDMatch
import reflect._

/*
 * Finds corresponding points between a pair of images using local descriptors.
 * The correspondences are visualized in the image "scalaCorrespondences.png",
 * which is written to disk.
 */
object ScalaCorrespondenceMatchingDemo {
  def run() {
    println(s"\nRunning ${classTag[this.type].toString.replace("$", "")}")

    // Detects keypoints and extracts descriptors in a given image of type Mat.
    def detectAndExtract(mat: Mat) = {
      // A special container class for KeyPoint.
      val keyPoints = new MatOfKeyPoint
      // We're using the SURF detector.
      val detector = FeatureDetector.create(FeatureDetector.SURF)
      detector.detect(mat, keyPoints)

      println(s"There were ${keyPoints.toArray.size} KeyPoints detected")

      // Let's just use the best KeyPoints.
      val sorted = keyPoints.toArray.sortBy(_.response).reverse.take(50)
      // There isn't a constructor that takes Array[KeyPoint], so we unpack
      // the array and use the constructor that can take any number of
      // arguments.
      val bestKeyPoints: MatOfKeyPoint = new MatOfKeyPoint(sorted: _*)

      // We're using the SURF descriptor.
      val extractor = DescriptorExtractor.create(DescriptorExtractor.SURF)
      val descriptors = new Mat
      extractor.compute(mat, bestKeyPoints, descriptors)

      println(s"${descriptors.rows} descriptors were extracted, each with dimension ${descriptors.cols}")

      (bestKeyPoints, descriptors)
    }

    // Load the images from the |resources| directory.
    val leftImage = Highgui.imread(getClass.getResource("/img1.png").getPath)
    val rightImage = Highgui.imread(getClass.getResource("/img2.png").getPath)

    // Detect KeyPoints and extract descriptors.
    val (leftKeyPoints, leftDescriptors) = detectAndExtract(leftImage)
    val (rightKeyPoints, rightDescriptors) = detectAndExtract(rightImage)

    // Match the descriptors.
    val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
    // A special container class for DMatch.
    val dmatches = new MatOfDMatch
    // The backticks are because "match" is a keyword in Scala.
    matcher.`match`(leftDescriptors, rightDescriptors, dmatches)

    // Visualize the matches and save the visualization.
    val correspondenceImage = new Mat
    Features2d.drawMatches(leftImage, leftKeyPoints, rightImage, rightKeyPoints, dmatches, correspondenceImage)
    val filename = "scalaCorrespondences.png"
    println(s"Writing ${filename}")
    assert(Highgui.imwrite(filename, correspondenceImage))
  }
}
