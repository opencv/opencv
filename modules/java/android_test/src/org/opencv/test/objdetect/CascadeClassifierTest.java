package org.opencv.test.objdetect;

import org.opencv.core.CvVectorRect;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class CascadeClassifierTest extends OpenCVTestCase {

    private CascadeClassifier cc;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        cc = null;
    }

    public void testCascadeClassifier() {
        cc = new CascadeClassifier();
        assertTrue(null != cc);
    }

    public void testCascadeClassifierString() {
        cc = new CascadeClassifier(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        assertTrue(null != cc);
    }

    public void testDetectMultiScaleMatListOfRect() {
        CascadeClassifier cc = new CascadeClassifier(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        CvVectorRect faces = new CvVectorRect();

        Mat greyLena = new Mat();
        Imgproc.cvtColor(rgbLena, greyLena, Imgproc.COLOR_RGB2GRAY);

        // TODO: doesn't detect with 1.1 scale
        cc.detectMultiScale(greyLena, faces, 1.09, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
        assertEquals(1, faces.total());
    }

    public void testDetectMultiScaleMatListOfRectDouble() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectDoubleInt() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectDoubleIntInt() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectDoubleIntIntSize() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectDoubleIntIntSizeSize() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDouble() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleInt() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntInt() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntIntSize() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntIntSizeSize() {
        fail("Not yet implemented");
    }

    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntIntSizeSizeBoolean() {
        fail("Not yet implemented");
    }

    public void testEmpty() {
        cc = new CascadeClassifier();
        assertTrue(cc.empty());
    }

    public void testLoad() {
        cc = new CascadeClassifier();
        cc.load(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        assertTrue(!cc.empty());
    }

}
