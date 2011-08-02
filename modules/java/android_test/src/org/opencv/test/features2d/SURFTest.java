package org.opencv.test.features2d;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.KeyPoint;
import org.opencv.features2d.SURF;
import org.opencv.test.OpenCVTestCase;

public class SURFTest extends OpenCVTestCase {

    int matSize;
    KeyPoint[] truth;

    public void test_1() {
        super.test_1("features2d.SURF");
    }

    @Override
    protected void setUp() throws Exception {
        matSize = 100;

        truth = new KeyPoint[] { new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1),
                new KeyPoint(44.224422454833984f, 44.224422454833984f, 16, 99.75463f, 8617.863f, 1, -1),
                new KeyPoint(44.224422454833984f, 55.775577545166016f, 16, 189.7546f, 8617.863f, 1, -1),
                new KeyPoint(55.775577545166016f, 55.775577545166016f, 16, 279.75464f, 8617.863f, 1, -1) };

        super.setUp();
    }

    private Mat getCross() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Core.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    private void order(List<KeyPoint> points) {
        Collections.sort(points, new Comparator<KeyPoint>() {
            public int compare(KeyPoint p1, KeyPoint p2) {
                if (p1.angle < p2.angle)
                    return -1;
                if (p1.angle > p2.angle)
                    return 1;
                return 0;
            }
        });
    }

    public void testDescriptorSize() {
        SURF surf = new SURF(500.0, 4, 2, false);
        assertEquals(64, surf.descriptorSize());

        surf = new SURF(500.0, 4, 2, true);
        assertEquals(128, surf.descriptorSize());
    }

    public void testDetectMatMatListOfKeyPoint_noPointsDetected() {
        SURF surf = new SURF(8000);
        List<KeyPoint> keypoints = new LinkedList<KeyPoint>();
        Mat gray0 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));

        surf.detect(gray0, new Mat(), keypoints);

        assertEquals(0, keypoints.size());
    }

    public void testDetectMatMatListOfKeyPoint() {
        SURF surf = new SURF(8000);
        List<KeyPoint> keypoints = new LinkedList<KeyPoint>();
        Mat cross = getCross();

        surf.detect(cross, new Mat(), keypoints);

        assertEquals(truth.length, keypoints.size());
        order(keypoints);
        for (int i = 0; i < truth.length; i++)
            assertKeyPointEqual(truth[i], keypoints.get(i), EPS);

        // for(KeyPoint kp : keypoints)
        // OpenCVTestRunner.Log(kp.toString());
    }

    public void testDetectMatMatListOfKeyPointListOfFloat() {
        SURF surf = new SURF(8000);
        List<KeyPoint> keypoints = new LinkedList<KeyPoint>();
        List<Float> descriptors = new LinkedList<Float>();
        Mat cross = getCross();

        surf.detect(cross, new Mat(), keypoints, descriptors);

        assertEquals(truth.length, keypoints.size());
        assertEquals(truth.length * surf.descriptorSize(), descriptors.size());
        order(keypoints);
        for (int i = 0; i < truth.length; i++)
            assertKeyPointEqual(truth[i], (KeyPoint) keypoints.get(i), EPS);
    }

    public void testDetectMatMatListOfKeyPointListOfFloatBoolean() {
        SURF surf = new SURF(8000);
        List<KeyPoint> original_keypoints = Arrays.asList(truth);
        List<KeyPoint> keypoints = new LinkedList<KeyPoint>(original_keypoints);
        List<Float> descriptors = new LinkedList<Float>();
        Mat gray255 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));

        surf.detect(gray255, new Mat(), keypoints, descriptors, true);

        // unmodified keypoints
        assertEquals(original_keypoints.size(), keypoints.size());
        for (int i = 0; i < keypoints.size(); i++)
            assertKeyPointEqual(original_keypoints.get(i), keypoints.get(i), EPS);

        // zero descriptors
        assertEquals(surf.descriptorSize() * original_keypoints.size(), descriptors.size());
        for (float d : descriptors)
            assertTrue(Math.abs(d) < EPS);
    }

    public void testSURF() {
        SURF surf = new SURF();
        assertNotNull(surf);
    }

    public void testSURFDouble() {
        SURF surf = new SURF(500.0);
        assertNotNull(surf);
    }

    public void testSURFDoubleInt() {
        SURF surf = new SURF(500.0, 4);
        assertNotNull(surf);
    }

    public void testSURFDoubleIntInt() {
        SURF surf = new SURF(500.0, 4, 2);
        assertNotNull(surf);
    }

    public void testSURFDoubleIntIntBoolean() {
        SURF surf = new SURF(500.0, 4, 2, false);
        assertNotNull(surf);
    }

    public void testSURFDoubleIntIntBooleanBoolean() {
        SURF surf = new SURF(500.0, 4, 2, false, false);
        assertNotNull(surf);
    }

}
