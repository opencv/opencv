package org.opencv.test.ml;

import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.CvType;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class MLTest extends OpenCVTestCase {

    public void testSaveLoad() {
        Mat samples = new MatOfFloat(new float[] {
            5.1f, 3.5f, 1.4f, 0.2f,
            4.9f, 3.0f, 1.4f, 0.2f,
            4.7f, 3.2f, 1.3f, 0.2f,
            4.6f, 3.1f, 1.5f, 0.2f,
            5.0f, 3.6f, 1.4f, 0.2f,
            7.0f, 3.2f, 4.7f, 1.4f,
            6.4f, 3.2f, 4.5f, 1.5f,
            6.9f, 3.1f, 4.9f, 1.5f,
            5.5f, 2.3f, 4.0f, 1.3f,
            6.5f, 2.8f, 4.6f, 1.5f
        }).reshape(1, 10);
        Mat responses = new MatOfInt(new int[] {
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1
        }).reshape(1, 10);
        SVM saved = SVM.create();
        assertFalse(saved.isTrained());

        saved.train(samples, Ml.ROW_SAMPLE, responses);
        assertTrue(saved.isTrained());

        String filename = OpenCVTestRunner.getTempFileName("yml");
        saved.save(filename);
        SVM loaded = SVM.load(filename);
        assertTrue(saved.isTrained());
    }

}
