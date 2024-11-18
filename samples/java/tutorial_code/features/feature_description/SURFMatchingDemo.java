import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features.DescriptorMatcher;
import org.opencv.features.Features;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.xfeatures2d.SURF;

class SURFMatching {
    public void run(String[] args) {
        String filename1 = args.length > 1 ? args[0] : "../data/box.png";
        String filename2 = args.length > 1 ? args[1] : "../data/box_in_scene.png";
        Mat img1 = Imgcodecs.imread(filename1, Imgcodecs.IMREAD_GRAYSCALE);
        Mat img2 = Imgcodecs.imread(filename2, Imgcodecs.IMREAD_GRAYSCALE);
        if (img1.empty() || img2.empty()) {
            System.err.println("Cannot read images!");
            System.exit(0);
        }

        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        double hessianThreshold = 400;
        int nOctaves = 4, nOctaveLayers = 3;
        boolean extended = false, upright = false;
        SURF detector = SURF.create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint(), keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat(), descriptors2 = new Mat();
        detector.detectAndCompute(img1, new Mat(), keypoints1, descriptors1);
        detector.detectAndCompute(img2, new Mat(), keypoints2, descriptors2);

        //-- Step 2: Matching descriptor vectors with a brute force matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors1, descriptors2, matches);

        //-- Draw matches
        Mat imgMatches = new Mat();
        Features.drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

        HighGui.imshow("Matches", imgMatches);
        HighGui.waitKey(0);

        System.exit(0);
    }
}

public class SURFMatchingDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new SURFMatching().run(args);
    }
}
