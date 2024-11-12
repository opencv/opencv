import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features.DescriptorMatcher;
import org.opencv.features.Features;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.xfeatures2d.SURF;

class SURFFLANNMatching {
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

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        //-- Filter matches using the Lowe's ratio test
        float ratioThresh = 0.7f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);

        //-- Draw matches
        Mat imgMatches = new Mat();
        Features.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar.all(-1),
                Scalar.all(-1), new MatOfByte(), Features.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

        //-- Show detected matches
        HighGui.imshow("Good Matches", imgMatches);
        HighGui.waitKey(0);

        System.exit(0);
    }
}

public class SURFFLANNMatchingDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new SURFFLANNMatching().run(args);
    }
}
