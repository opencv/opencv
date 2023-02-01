package org.opencv.test.features2d;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.DMatch;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.core.KeyPoint;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.features2d.Feature2D;

public class Features2dTest extends OpenCVTestCase {

    public void testDrawKeypointsMatListOfKeyPointMat() {
        fail("Not yet implemented");
    }

    public void testDrawKeypointsMatListOfKeyPointMatScalar() {
        fail("Not yet implemented");
    }

    public void testDrawKeypointsMatListOfKeyPointMatScalarInt() {
        fail("Not yet implemented");
    }

    public void testDrawMatches2MatListOfKeyPointMatListOfKeyPointListOfListOfDMatchMat() {
        fail("Not yet implemented");
    }

    public void testDrawMatches2MatListOfKeyPointMatListOfKeyPointListOfListOfDMatchMatScalar() {
        fail("Not yet implemented");
    }

    public void testDrawMatches2MatListOfKeyPointMatListOfKeyPointListOfListOfDMatchMatScalarScalar() {
        fail("Not yet implemented");
    }

    public void testDrawMatches2MatListOfKeyPointMatListOfKeyPointListOfListOfDMatchMatScalarScalarListOfListOfByte() {
        fail("Not yet implemented");
    }

    public void testDrawMatches2MatListOfKeyPointMatListOfKeyPointListOfListOfDMatchMatScalarScalarListOfListOfByteInt() {
        fail("Not yet implemented");
    }

    public void testDrawMatchesMatListOfKeyPointMatListOfKeyPointListOfDMatchMat() {
        fail("Not yet implemented");
    }

    public void testDrawMatchesMatListOfKeyPointMatListOfKeyPointListOfDMatchMatScalar() {
        fail("Not yet implemented");
    }

    public void testDrawMatchesMatListOfKeyPointMatListOfKeyPointListOfDMatchMatScalarScalar() {
        fail("Not yet implemented");
    }

    public void testDrawMatchesMatListOfKeyPointMatListOfKeyPointListOfDMatchMatScalarScalarListOfByte() {
        fail("Not yet implemented");
    }

    public void testDrawMatchesMatListOfKeyPointMatListOfKeyPointListOfDMatchMatScalarScalarListOfByteInt() {
        fail("Not yet implemented");
    }

    public void testPTOD()
    {
        String detectorCfg = "%YAML:1.0\n---\nhessianThreshold: 4000.\noctaves: 3\noctaveLayers: 4\nupright: 0\n";
        String extractorCfg = "%YAML:1.0\n---\nnOctaves: 4\nnOctaveLayers: 2\nextended: 0\nupright: 0\n";

        Feature2D detector = createClassInstance(XFEATURES2D+"SURF", DEFAULT_FACTORY, null, null);
        Feature2D extractor = createClassInstance(XFEATURES2D+"SURF", DEFAULT_FACTORY, null, null);
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);

        String detectorCfgFile = OpenCVTestRunner.getTempFileName("yml");
        writeFile(detectorCfgFile, detectorCfg);
        detector.read(detectorCfgFile);

        String extractorCfgFile = OpenCVTestRunner.getTempFileName("yml");
        writeFile(extractorCfgFile, extractorCfg);
        extractor.read(extractorCfgFile);

        Mat imgTrain = Imgcodecs.imread(OpenCVTestRunner.LENA_PATH, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        Mat imgQuery = imgTrain.submat(new Range(0, imgTrain.rows() - 100), Range.all());

        MatOfKeyPoint trainKeypoints = new MatOfKeyPoint();
        MatOfKeyPoint queryKeypoints = new MatOfKeyPoint();

        detector.detect(imgTrain, trainKeypoints);
        detector.detect(imgQuery, queryKeypoints);

        // OpenCVTestRunner.Log("Keypoints found: " + trainKeypoints.size() +
        // ":" + queryKeypoints.size());

        Mat trainDescriptors = new Mat();
        Mat queryDescriptors = new Mat();

        extractor.compute(imgTrain, trainKeypoints, trainDescriptors);
        extractor.compute(imgQuery, queryKeypoints, queryDescriptors);

        MatOfDMatch matches = new MatOfDMatch();

        matcher.add(Arrays.asList(trainDescriptors));
        matcher.match(queryDescriptors, matches);

        // OpenCVTestRunner.Log("Matches found: " + matches.size());

        DMatch adm[] = matches.toArray();
        List<Point> lp1 = new ArrayList<Point>(adm.length);
        List<Point> lp2 = new ArrayList<Point>(adm.length);
        KeyPoint tkp[] = trainKeypoints.toArray();
        KeyPoint qkp[] = queryKeypoints.toArray();
        for (int i = 0; i < adm.length; i++) {
            DMatch dm = adm[i];
            lp1.add(tkp[dm.trainIdx].pt);
            lp2.add(qkp[dm.queryIdx].pt);
        }

        MatOfPoint2f points1 = new MatOfPoint2f(lp1.toArray(new Point[0]));
        MatOfPoint2f points2 = new MatOfPoint2f(lp2.toArray(new Point[0]));

        Mat hmg = Calib3d.findHomography(points1, points2, Calib3d.RANSAC, 3);

        assertMatEqual(Mat.eye(3, 3, CvType.CV_64F), hmg, EPS);

        Mat outimg = new Mat();
        Features2d.drawMatches(imgQuery, queryKeypoints, imgTrain, trainKeypoints, matches, outimg);
        String outputPath = OpenCVTestRunner.getOutputFileName("PTODresult.png");
        Imgcodecs.imwrite(outputPath, outimg);
        // OpenCVTestRunner.Log("Output image is saved to: " + outputPath);
    }
}
