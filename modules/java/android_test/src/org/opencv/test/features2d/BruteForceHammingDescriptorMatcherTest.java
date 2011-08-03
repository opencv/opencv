package org.opencv.test.features2d;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BruteForceHammingDescriptorMatcherTest extends OpenCVTestCase {

    DescriptorMatcher matcher;
    int matSize;
    DMatch[] truth;

    protected void setUp() throws Exception {
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        matSize = 100;

        truth = new DMatch[] { 
                new DMatch (0, 0, 0, 51),
                new DMatch (1, 2, 0, 42),
                new DMatch (2, 1, 0, 40),
                new DMatch (3, 3, 0, 53) };
        super.setUp();
    }

    private Mat getTrainImg() {
        Mat img = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(img, new Point(40, 40), new Point(matSize - 40, matSize - 40), new Scalar(0), 8);
        return img;
    }
    
    private Mat getQueryImg() {
        Mat img = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(img, new Point(40, matSize - 40), new Point(matSize - 50, 50), new Scalar(0), 8);
        return img;
    }
    
    private Mat getTestDescriptors(Mat img) {
        List<KeyPoint> keypoints = new ArrayList<KeyPoint>();
        Mat descriptors = new Mat();

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.FAST);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.BRIEF);

        detector.detect(img, keypoints);
        extractor.compute(img, keypoints, descriptors);
        
        return descriptors;
    }

    private Mat getQueryDescriptors() {
        return getTestDescriptors(getQueryImg());
    }
    
    private Mat getTrainDescriptors() {
        return getTestDescriptors(getTrainImg());
    }

    private Mat getMaskImg() {
        return new Mat(4, 4, CvType.CV_8U, new Scalar(0)) {
            {
                put(0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
            }
        };
    }

    public void testAdd() {
        matcher.add(Arrays.asList(new Mat()));
        assertFalse(matcher.empty());
    }

    public void testClear() {
        matcher.add(Arrays.asList(new Mat()));

        matcher.clear();

        assertTrue(matcher.empty());
    }

    public void testCloneBoolean() {
        matcher.add(Arrays.asList(new Mat()));

        DescriptorMatcher cloned = matcher.clone(true);

        assertNotNull(cloned);
        assertTrue(cloned.empty());
    }

    public void testClone() {
        Mat train = new Mat(1, 1, CvType.CV_8U, new Scalar(123));
        Mat truth = train.clone();
        matcher.add(Arrays.asList(train));

        DescriptorMatcher cloned = matcher.clone();

        assertNotNull(cloned);

        List<Mat> descriptors = cloned.getTrainDescriptors();
        assertEquals(1, descriptors.size());
        assertMatEqual(truth, descriptors.get(0));
    }

    public void testCreate() {
        assertNotNull(matcher);
    }

    public void testEmpty() {
        assertTrue(matcher.empty());
    }

    public void testGetTrainDescriptors() {
        Mat train = new Mat(1, 1, CvType.CV_8U, new Scalar(123));
        Mat truth = train.clone();
        matcher.add(Arrays.asList(train));

        List<Mat> descriptors = matcher.getTrainDescriptors();

        assertEquals(1, descriptors.size());
        assertMatEqual(truth, descriptors.get(0));
    }

    public void testIsMaskSupported() {
        assertTrue(matcher.isMaskSupported());
    }

    public void testMatchMatMatListOfDMatchMat() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        Mat mask = getMaskImg();
        List<DMatch> matches = new ArrayList<DMatch>();
        
        matcher.match(query, train, matches, mask);

        assertListDMatchEquals(Arrays.asList(truth[0], truth[1]), matches, EPS);
    }

    public void testMatchMatMatListOfDMatch() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        List<DMatch> matches = new ArrayList<DMatch>();
        
        matcher.match(query, train, matches);
        
        assertListDMatchEquals(Arrays.asList(truth), matches, EPS);
    }

    public void testMatchMatListOfDMatchListOfMat() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        Mat mask = getMaskImg();
        List<DMatch> matches = new ArrayList<DMatch>();
        matcher.add(Arrays.asList(train));

        matcher.match(query, matches, Arrays.asList(mask));

        assertListDMatchEquals(Arrays.asList(truth[0], truth[1]), matches, EPS);
    }

    public void testMatchMatListOfDMatch() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        List<DMatch> matches = new ArrayList<DMatch>();
        matcher.add(Arrays.asList(train));
        
        matcher.match(query, matches);
        
        assertListDMatchEquals(Arrays.asList(truth), matches, EPS);
    }

    public void testRead() {
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n");

        matcher.read(filename);
        assertTrue(true);// BruteforceMatcher has no settings
    }

    public void testTrain() {
        matcher.train();// BruteforceMatcher does not need to train
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        matcher.write(filename);

        String truth = "%YAML:1.0\n";
        assertEquals(truth, readFile(filename));
    }

}
