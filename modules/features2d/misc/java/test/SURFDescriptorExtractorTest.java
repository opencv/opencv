package org.opencv.test.features2d;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.Feature2D;

public class SURFDescriptorExtractorTest extends OpenCVTestCase {

    Feature2D extractor;
    int matSize;

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Imgproc.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Imgproc.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        Class[] cParams = {double.class, int.class, int.class, boolean.class, boolean.class};
        Object[] oValues = {100, 2, 4, true, false};
        extractor = createClassInstance(XFEATURES2D+"SURF", DEFAULT_FACTORY, cParams, oValues);

        matSize = 100;
    }

    public void testComputeListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testComputeMatListOfKeyPointMat() {
        KeyPoint point = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        MatOfKeyPoint keypoints = new MatOfKeyPoint(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 128, CvType.CV_32FC1) {
            {
                put(0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0.058821894, 0.058821894, -0.045962855, 0.046261817, 0.0085156476,
                          0.0085754395, -0.0064509804, 0.0064509804, 0.00044069235, 0.00044069235, 0, 0, 0.00025723741,
                          0.00025723741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00025723741, 0.00025723741, -0.00044069235,
                          0.00044069235, 0, 0, 0.36278215, 0.36278215, -0.24688604, 0.26173124, 0.052068226, 0.052662034,
                          -0.032815345, 0.032815345, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0064523756,
                          0.0064523756, 0.0082002236, 0.0088908644, -0.059001274, 0.059001274, 0.045789491, 0.04648013,
                          0.11961588, 0.22789426, -0.01322381, 0.18291828, -0.14042182, 0.23973691, 0.073782086, 0.23769434,
                          -0.027880307, 0.027880307, 0.049587864, 0.049587864, -0.33991757, 0.33991757, 0.21437603, 0.21437603,
                          -0.0020763327, 0.0020763327, 0.006245892, 0.006245892, -0.04067041, 0.04067041, 0.019361559,
                          0.019361559, 0, 0, -0.0035977389, 0.0035977389, 0, 0, -0.00099993451, 0.00099993451, 0.040670406,
                          0.040670406, -0.019361559, 0.019361559, 0.006245892, 0.006245892, -0.0020763327, 0.0020763327,
                          -0.00034532088, 0.00034532088, 0, 0, 0, 0, 0.00034532088, 0.00034532088, -0.00099993451,
                          0.00099993451, 0, 0, 0, 0, 0.0035977389, 0.0035977389
                    );
            }
        };

        assertMatEqual(truth, descriptors, EPS);
    }

    public void testCreate() {
        assertNotNull(extractor);
    }

    public void testDescriptorSize() {
        assertEquals(128, extractor.descriptorSize());
    }

    public void testDescriptorType() {
        assertEquals(CvType.CV_32F, extractor.descriptorType());
    }

    public void testEmpty() {
//        assertFalse(extractor.empty());
        fail("Not yet implemented");
    }

    public void testRead() {
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n---\nnOctaves: 4\nnOctaveLayers: 2\nextended: 1\nupright: 0\n");

        extractor.read(filename);

        assertEquals(128, extractor.descriptorSize());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

//        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.SURF</name>\n<extended>1</extended>\n<hessianThreshold>100.</hessianThreshold>\n<nOctaveLayers>2</nOctaveLayers>\n<nOctaves>4</nOctaves>\n<upright>0</upright>\n</opencv_storage>\n";
        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

//        String truth = "%YAML:1.0\n---\nname: \"Feature2D.SURF\"\nextended: 1\nhessianThreshold: 100.\nnOctaveLayers: 2\nnOctaves: 4\nupright: 0\n";
        String truth = "%YAML:1.0\n---\n";
        assertEquals(truth, readFile(filename));
    }

}
