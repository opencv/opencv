package org.opencv.test.ml;

import org.opencv.ml.CvKNearest;
import org.opencv.test.OpenCVTestCase;

public class CvKNearestTest extends OpenCVTestCase {

    public void testCvKNearest() {
        new CvKNearest();
    }

    public void testCvKNearestMatMat() {
        fail("Not yet implemented");
    }

    public void testCvKNearestMatMatMat() {
        fail("Not yet implemented");
    }

    public void testCvKNearestMatMatMatBoolean() {
        fail("Not yet implemented");
    }

    public void testCvKNearestMatMatMatBooleanInt() {
        fail("Not yet implemented");
    }

    public void testFind_nearest() {
        fail("Not yet implemented");
    }

    public void testTrainMatMat() {
        fail("Not yet implemented");
    }

    public void testTrainMatMatMat() {
        fail("Not yet implemented");
    }

    public void testTrainMatMatMatBoolean() {
        fail("Not yet implemented");
    }

    public void testTrainMatMatMatBooleanInt() {
        fail("Not yet implemented");
    }

    public void testTrainMatMatMatBooleanIntBoolean() {
        fail("Not yet implemented");
    }

    public void testKmeansMatIntMatTermCriteriaIntInt() {
        Mat data = new Mat(4, 5, CvType.CV_32FC1) {
            {
                put(0, 0, 1, 2, 3, 4, 5);
                put(1, 0, 2, 3, 4, 5, 6);
                put(2, 0, 5, 4, 3, 2, 1);
                put(3, 0, 6, 5, 4, 3, 2);
            }
        };
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS, 0, EPS);
        Mat labels = new Mat();

        ml.kmeans(data, 2, labels, criteria, 1, ml.KMEANS_PP_CENTERS);

        int[] first_center = new int[1];
        labels.get(0, 0, first_center);
        final int c1 = first_center[0];
        Mat expected_labels = new Mat(4, 1, CvType.CV_32S) {
            {
                put(0, 0, c1, c1, 1 - c1, 1 - c1);
            }
        };
        assertMatEqual(expected_labels, labels);
    }

    public void testKmeansMatIntMatTermCriteriaIntIntMat() {
        Mat data = new Mat(4, 5, CvType.CV_32FC1) {
            {
                put(0, 0, 1, 2, 3, 4, 5);
                put(1, 0, 2, 3, 4, 5, 6);
                put(2, 0, 5, 4, 3, 2, 1);
                put(3, 0, 6, 5, 4, 3, 2);
            }
        };
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS, 0, EPS);
        Mat labels = new Mat();
        Mat centers = new Mat();

        ml.kmeans(data, 2, labels, criteria, 6, ml.KMEANS_RANDOM_CENTERS, centers);

        int[] first_center = new int[1];
        labels.get(0, 0, first_center);
        final int c1 = first_center[0];
        Mat expected_labels = new Mat(4, 1, CvType.CV_32S) {
            {
                put(0, 0, c1, c1, 1 - c1, 1 - c1);
            }
        };
        Mat expected_centers = new Mat(2, 5, CvType.CV_32FC1) {
            {
                put(c1, 0, 1.5, 2.5, 3.5, 4.5, 5.5);
                put(1 - c1, 0, 5.5, 4.5, 3.5, 2.5, 1.5);
            }
        };
        assertMatEqual(expected_labels, labels);
        assertMatEqual(expected_centers, centers, EPS);
    }

}
