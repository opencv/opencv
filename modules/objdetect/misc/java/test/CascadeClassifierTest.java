package org.opencv.test.objdetect;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.test.NotYetImplemented;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class CascadeClassifierTest extends OpenCVTestCase {

    private CascadeClassifier cc;

    @Test
    public void testCascadeClassifier() {
        cc = new CascadeClassifier();
        assertNotNull(cc);
    }

    @Test
    public void testCascadeClassifierString() {
        cc = new CascadeClassifier(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        assertNotNull(cc);
    }

    @Test
    public void testDetectMultiScaleMatListOfRect() {
        CascadeClassifier cc = new CascadeClassifier(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        MatOfRect faces = new MatOfRect();

        Mat greyLena = new Mat();
        Imgproc.cvtColor(rgbLena, greyLena, Imgproc.COLOR_RGB2GRAY);

        // TODO: doesn't detect with 1.1 scale
        cc.detectMultiScale(greyLena, faces, 1.09, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
        assertEquals(1, faces.total());
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectDouble() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectDoubleInt() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectDoubleIntInt() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectDoubleIntIntSize() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectDoubleIntIntSizeSize() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDouble() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDouble() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleInt() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntInt() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntIntSize() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntIntSizeSize() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testDetectMultiScaleMatListOfRectListOfIntegerListOfDoubleDoubleIntIntSizeSizeBoolean() {
        fail("Not yet implemented");
    }

    @Test
    public void testEmpty() {
        cc = new CascadeClassifier();
        assertTrue(cc.empty());
    }

    @Test
    public void testLoad() {
        cc = new CascadeClassifier();
        cc.load(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        assertFalse(cc.empty());
    }

}
