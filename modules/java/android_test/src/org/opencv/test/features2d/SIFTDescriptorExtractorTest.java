package org.opencv.test.features2d;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

import java.util.Arrays;
import java.util.List;

public class SIFTDescriptorExtractorTest extends OpenCVTestCase {

    DescriptorExtractor extractor;
    KeyPoint keypoint;
    Mat truth;
    int matSize;

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Core.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    @Override
    protected void setUp() throws Exception {
        extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        keypoint = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        matSize = 100;
        truth = new Mat(1, 128, CvType.CV_32FC1) {
            {
                put(0,0, 123, 0, 0, 1, 123, 0, 0, 1, 123, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 123, 0, 0, 2, 123, 0, 0, 2, 123, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 30,
                        7, 31, 123, 0, 0, 0, 123, 52, 88, 0, 0, 0, 0, 0, 0, 2, 123, 0, 0, 0, 0, 0, 0, 1, 110, 0, 0, 0, 0, 0, 18, 37, 18, 34, 16,
                        21, 12, 23, 12, 50, 123, 0, 0, 0, 90, 26, 0, 3, 123, 0, 0, 1, 122, 0, 0, 2, 123, 0, 0, 1, 93, 0);
            }
        };

        super.setUp();
    }

    public void testCompute() {
        List<KeyPoint> keypoints = Arrays.asList(keypoint);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        extractor.compute(img, keypoints, descriptors);

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
        assertFalse(extractor.empty());
    }

    public void testRead() {
        List<KeyPoint> keypoints = Arrays.asList(keypoint);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nmagnification: 3.\nisNormalize: 1\nrecalculateAngles: 1\nnOctaves: 6\nnOctaveLayers: 4\nfirstOctave: -1\nangleMode: 0\n");

        extractor.read(filename);

        extractor.compute(img, keypoints, descriptors);
        assertMatNotEqual(truth, descriptors, EPS);
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<magnification>3.</magnification>\n<isNormalize>1</isNormalize>\n<recalculateAngles>1</recalculateAngles>\n<nOctaves>4</nOctaves>\n<nOctaveLayers>3</nOctaveLayers>\n<firstOctave>-1</firstOctave>\n<angleMode>0</angleMode>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\nmagnification: 3.\nisNormalize: 1\nrecalculateAngles: 1\nnOctaves: 4\nnOctaveLayers: 3\nfirstOctave: -1\nangleMode: 0\n";
        assertEquals(truth, readFile(filename));
    }

}
