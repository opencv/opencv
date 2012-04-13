package org.opencv.test.features2d;

import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class FlannBasedDescriptorMatcherTest extends OpenCVTestCase {

    static final String xmlParamsDefault = "<?xml version=\"1.0\"?>\n"
            + "<opencv_storage>\n"
            + "<indexParams>\n"
            + "  <_>\n"
            + "    <name>algorithm</name>\n"
            + "    <type>23</type>\n"
            + "    <value>1</value></_>\n"
            + "  <_>\n"
            + "    <name>trees</name>\n"
            + "    <type>4</type>\n"
            + "    <value>4</value></_></indexParams>\n"
            + "<searchParams>\n"
            + "  <_>\n"
            + "    <name>checks</name>\n"
            + "    <type>4</type>\n"
            + "    <value>32</value></_>\n"
            + "  <_>\n"
            + "    <name>eps</name>\n"
            + "    <type>5</type>\n"
            + "    <value>0.</value></_>\n"
            + "  <_>\n"
            + "    <name>sorted</name>\n"
            + "    <type>15</type>\n"
            + "    <value>1</value></_></searchParams>\n"
            + "</opencv_storage>\n";
    static final String ymlParamsDefault = "%YAML:1.0\n"
            + "indexParams:\n"
            + "   -\n"
            + "      name: algorithm\n"
            + "      type: 23\n"
            + "      value: 1\n"
            + "   -\n"
            + "      name: trees\n"
            + "      type: 4\n"
            + "      value: 4\n"
            + "searchParams:\n"
            + "   -\n"
            + "      name: checks\n"
            + "      type: 4\n"
            + "      value: 32\n"
            + "   -\n"
            + "      name: eps\n"
            + "      type: 5\n"
            + "      value: 0.\n"
            + "   -\n"
            + "      name: sorted\n"
            + "      type: 15\n"
            + "      value: 1\n";
    static final String ymlParamsModified = "%YAML:1.0\n"
            + "indexParams:\n"
            + "   -\n"
            + "      name: algorithm\n"
            + "      type: 23\n"
            + "      value: 6\n"// this line is changed!
            + "   -\n"
            + "      name: trees\n"
            + "      type: 4\n"
            + "      value: 4\n"
            + "searchParams:\n"
            + "   -\n"
            + "      name: checks\n"
            + "      type: 4\n"
            + "      value: 32\n"
            + "   -\n"
            + "      name: eps\n"
            + "      type: 5\n"
            + "      value: 4.\n"// this line is changed!
            + "   -\n"
            + "      name: sorted\n"
            + "      type: 15\n"
            + "      value: 1\n";

    DescriptorMatcher matcher;

    int matSize;

    DMatch[] truth;

    private Mat getMaskImg() {
        return new Mat(5, 2, CvType.CV_8U, new Scalar(0)) {
            {
                put(0, 0, 1, 1, 1, 1);
            }
        };
    }

    private Mat getQueryDescriptors() {
        Mat img = getQueryImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nhessianThreshold: 8000.\noctaves: 3\noctaveLayers: 4\nupright: 0\n");
        detector.read(filename);

        detector.detect(img, keypoints);
        extractor.compute(img, keypoints, descriptors);

        return descriptors;
    }

    private Mat getQueryImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(30, matSize / 2), new Point(matSize - 31, matSize / 2), new Scalar(100), 3);
        Core.line(cross, new Point(matSize / 2, 30), new Point(matSize / 2, matSize - 31), new Scalar(100), 3);

        return cross;
    }

    private Mat getTrainDescriptors() {
        Mat img = getTrainImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint(new KeyPoint(50, 50, 16, 0, 20000, 1, -1), new KeyPoint(42, 42, 16, 160, 10000, 1, -1));
        Mat descriptors = new Mat();

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

        extractor.compute(img, keypoints, descriptors);

        return descriptors;
    }

    private Mat getTrainImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Core.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    protected void setUp() throws Exception {
        matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        matSize = 100;

        truth = new DMatch[] {
        		/*
                new DMatch(0, 0, 0, 0.643284f),
                new DMatch(1, 1, 0, 0.92945856f),
                new DMatch(2, 1, 0, 0.2841479f),
                new DMatch(3, 1, 0, 0.9194034f),
                new DMatch(4, 1, 0, 0.3006621f)
                */
        		new DMatch(0, 0, 0, 1.049694f), 
        		new DMatch(1, 0, 0, 1.083795f), 
        		new DMatch(2, 1, 0, 0.484352f), 
        		new DMatch(3, 0, 0, 1.098605f), 
        		new DMatch(4, 1, 0, 0.494587f)
                };

        super.setUp();
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

    public void testClone() {
        Mat train = new Mat(1, 1, CvType.CV_8U, new Scalar(123));
        matcher.add(Arrays.asList(train));

        try {
            matcher.clone();
            fail("Expected CvException (CV_StsNotImplemented)");
        } catch (CvException cverr) {
            // expected
        }
    }

    public void testCloneBoolean() {
        matcher.add(Arrays.asList(new Mat()));

        DescriptorMatcher cloned = matcher.clone(true);

        assertNotNull(cloned);
        assertTrue(cloned.empty());
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
        assertFalse(matcher.isMaskSupported());
    }

    public void testKnnMatchMatListOfListOfDMatchInt() {
        fail("Not yet implemented");
    }

    public void testKnnMatchMatListOfListOfDMatchIntListOfMat() {
        fail("Not yet implemented");
    }

    public void testKnnMatchMatListOfListOfDMatchIntListOfMatBoolean() {
        fail("Not yet implemented");
    }

    public void testKnnMatchMatMatListOfListOfDMatchInt() {
        fail("Not yet implemented");
    }

    public void testKnnMatchMatMatListOfListOfDMatchIntMat() {
        fail("Not yet implemented");
    }

    public void testKnnMatchMatMatListOfListOfDMatchIntMatBoolean() {
        fail("Not yet implemented");
    }

    public void testMatchMatListOfDMatch() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        MatOfDMatch matches = new MatOfDMatch();
        matcher.add(Arrays.asList(train));
        matcher.train();

        matcher.match(query, matches);

        assertArrayDMatchEquals(truth, matches.toArray(), EPS);
    }

    public void testMatchMatListOfDMatchListOfMat() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        Mat mask = getMaskImg();
        MatOfDMatch matches = new MatOfDMatch();
        matcher.add(Arrays.asList(train));
        matcher.train();

        matcher.match(query, matches, Arrays.asList(mask));

        assertArrayDMatchEquals(truth, matches.toArray(), EPS);
    }

    public void testMatchMatMatListOfDMatch() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        MatOfDMatch matches = new MatOfDMatch();

        matcher.match(query, train, matches);

        assertArrayDMatchEquals(truth, matches.toArray(), EPS);

        // OpenCVTestRunner.Log("matches found: " + matches.size());
        // for (DMatch m : matches)
        // OpenCVTestRunner.Log(m.toString());
    }

    public void testMatchMatMatListOfDMatchMat() {
        Mat train = getTrainDescriptors();
        Mat query = getQueryDescriptors();
        Mat mask = getMaskImg();
        MatOfDMatch matches = new MatOfDMatch();

        matcher.match(query, train, matches, mask);

        assertListDMatchEquals(Arrays.asList(truth), matches.toList(), EPS);
    }

    public void testRadiusMatchMatListOfListOfDMatchFloat() {
        fail("Not yet implemented");
    }

    public void testRadiusMatchMatListOfListOfDMatchFloatListOfMat() {
        fail("Not yet implemented");
    }

    public void testRadiusMatchMatListOfListOfDMatchFloatListOfMatBoolean() {
        fail("Not yet implemented");
    }

    public void testRadiusMatchMatMatListOfListOfDMatchFloat() {
        fail("Not yet implemented");
    }

    public void testRadiusMatchMatMatListOfListOfDMatchFloatMat() {
        fail("Not yet implemented");
    }

    public void testRadiusMatchMatMatListOfListOfDMatchFloatMatBoolean() {
        fail("Not yet implemented");
    }

    public void testRead() {
        String filenameR = OpenCVTestRunner.getTempFileName("yml");
        String filenameW = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filenameR, ymlParamsModified);

        matcher.read(filenameR);
        matcher.write(filenameW);
        
        assertEquals(ymlParamsModified, readFile(filenameW));
    }

    public void testTrain() {
        Mat train = getTrainDescriptors();
        matcher.add(Arrays.asList(train));
        matcher.train();
    }

    public void testTrainNoData() {
        try {
            matcher.train();
            fail("Expected CvException - FlannBasedMatcher::train should fail on empty train set");
        } catch (CvException cverr) {
            // expected
        }
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        matcher.write(filename);

        assertEquals(xmlParamsDefault, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        matcher.write(filename);

        assertEquals(ymlParamsDefault, readFile(filename));
    }

}
