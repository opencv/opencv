package org.opencv.test.features2d;

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
import org.opencv.test.OpenCVTestRunner;

public class SURFTest extends OpenCVTestCase {

    int matSize;
    KeyPoint[] truth;

    public void test_1() {
        super.test_1("features2d.SURF");
    }

    @Override
    protected void setUp() throws Exception {
        matSize = 100;

        truth = new KeyPoint[] {
                new KeyPoint(55.775577545166016f, 44.224422454833984f, 16,
                        9.754629f, 8617.863f, 1, -1),
                new KeyPoint(44.224422454833984f, 44.224422454833984f, 16,
                        99.75463f, 8617.863f, 1, -1),
                new KeyPoint(44.224422454833984f, 55.775577545166016f, 16,
                        189.7546f, 8617.863f, 1, -1),
                new KeyPoint(55.775577545166016f, 55.775577545166016f, 16,
                        279.75464f, 8617.863f, 1, -1) };

        super.setUp();
    }

    private Mat getCross() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(20, matSize / 2), new Point(matSize - 21,
                matSize / 2), new Scalar(100), 2);
        Core.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2,
                matSize - 21), new Scalar(100), 2);

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

    public void testDetectMatMatListOfKeyPoint() {
        SURF surf = new SURF(8000);
        List<KeyPoint> keypoints = new LinkedList<KeyPoint>();

        Mat gray0 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        surf.detect(gray0, new Mat(), keypoints);
        assertEquals(0, keypoints.size());

        Mat cross = getCross();
        surf.detect(cross, new Mat(), keypoints);
        assertEquals(truth.length, keypoints.size());

        order(keypoints);

        for (int i = 0; i < truth.length; i++)
            assertKeyPointEqual(truth[i], (KeyPoint) keypoints.get(i), EPS);

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

        float[] _truth = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.09548335f, 0.01814415f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.101962745f, 0.009914574f,
                0.57075155f, 0.047922116f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0029440068f, -0.011540107f, 0.01814415f,
                0.09548335f, 0.08538555f, -0.054076977f, 0.34105155f,
                0.47911066f, 0.02339545f, -0.11012388f, 0.08819653f,
                0.50863767f, 0.003179069f, -0.019882837f, 0.008947697f,
                0.054817006f, -0.0033560959f, -0.0011770058f, 0.0033560959f,
                0.0011770058f, 0.019882834f, 0.0031790687f, 0.054817006f,
                0.008947698f, 0.0f, 0.0f, 0.0f, 0.0f, -0.0011770058f,
                0.0033560959f, 0.0011770058f, 0.0033560959f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.011540107f, 0.0029440077f, 0.09548335f, 0.01814415f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.101962745f, 0.009914574f, 0.57075155f,
                0.047922116f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0029440068f, -0.011540107f, 0.01814415f, 0.09548335f,
                0.08538555f, -0.054076977f, 0.34105155f, 0.47911066f,
                0.02339545f, -0.11012388f, 0.08819653f, 0.50863767f,
                0.003179069f, -0.019882837f, 0.008947697f, 0.054817006f,
                -0.0033560959f, -0.0011770058f, 0.0033560959f, 0.0011770058f,
                0.019882834f, 0.0031790687f, 0.054817006f, 0.008947698f, 0.0f,
                0.0f, 0.0f, 0.0f, -0.0011770058f, 0.0033560959f, 0.0011770058f,
                0.0033560959f, 0.0f, 0.0f, 0.0f, 0.0f, 0.011540107f,
                0.0029440077f, 0.09548335f, 0.01814415f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.101962745f, 0.009914574f, 0.57075155f, 0.047922116f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0029440068f,
                -0.011540107f, 0.01814415f, 0.09548335f, 0.08538555f,
                -0.054076977f, 0.34105155f, 0.47911066f, 0.02339545f,
                -0.11012388f, 0.08819653f, 0.50863767f, 0.003179069f,
                -0.019882837f, 0.008947697f, 0.054817006f, -0.0033560959f,
                -0.0011770058f, 0.0033560959f, 0.0011770058f, 0.019882834f,
                0.0031790687f, 0.054817006f, 0.008947698f, 0.0f, 0.0f, 0.0f,
                0.0f, -0.0011770058f, 0.0033560959f, 0.0011770058f,
                0.0033560959f, 0.0f, 0.0f, 0.0f, 0.0f, 0.011540107f,
                0.0029440077f, 0.09548335f, 0.01814415f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.101962745f, 0.009914574f, 0.57075155f, 0.047922116f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0029440068f,
                -0.011540107f, 0.01814415f, 0.09548335f, 0.08538555f,
                -0.054076977f, 0.34105155f, 0.47911066f, 0.02339545f,
                -0.11012388f, 0.08819653f, 0.50863767f, 0.003179069f,
                -0.019882837f, 0.008947697f, 0.054817006f, -0.0033560959f,
                -0.0011770058f, 0.0033560959f, 0.0011770058f, 0.019882834f,
                0.0031790687f, 0.054817006f, 0.008947698f, 0.0f, 0.0f, 0.0f,
                0.0f, -0.0011770058f, 0.0033560959f, 0.0011770058f,
                0.0033560959f };
        
        for (int i = 0; i < descriptors.size(); i++)
            assertTrue(Math.abs(_truth[i] - (float)descriptors.get(i)) < EPS);

        for (KeyPoint kp : keypoints)
            OpenCVTestRunner.Log(kp.toString());

        OpenCVTestRunner.Log("desc - " + descriptors.size());

        String q = "";
        for (Float d : descriptors) {
            if (Math.abs(d) < EPS)
                d = 0f;
            q += ", " + d + "f";
        }
        q = q.substring(2);

        OpenCVTestRunner.Log("[" + q + "]");
    }

    public void testDetectMatMatListOfKeyPointListOfFloatBoolean() {
        fail("Not yet implemented");
        SURF surf = new SURF(8000);
        List<KeyPoint> keypoints = new LinkedList<KeyPoint>();
        List<Float> descriptors = new LinkedList<Float>();
        
        keypoints.add(truth[0]);
        //keypoints.add(truth[1]);
        //keypoints.add(truth[2]);
        //keypoints.add(truth[3]);
        //assertEquals(1, keypoints.size());

        Mat cross = getCross();
        surf.detect(cross, new Mat(), keypoints, descriptors, true);
        assertEquals(1, keypoints.size());
        assertEquals(surf.descriptorSize(), descriptors.size());
        
        
        for (KeyPoint kp : keypoints)
            OpenCVTestRunner.Log(kp.toString());

        OpenCVTestRunner.Log("desc - " + descriptors.size());

        String q = "";
        for (Float d : descriptors) {
            if (Math.abs(d) < EPS)
                d = 0f;
            q += ", " + d + "f";
        }
        q = q.substring(2);

        OpenCVTestRunner.Log("[" + q + "]");
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
