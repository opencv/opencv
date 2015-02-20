package org.opencv.test.highgui;

import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class HighguiTest extends OpenCVTestCase {

    public void testImdecode() {
        fail("Not yet implemented");
    }

    public void testImencodeStringMatListOfByte() {
        MatOfByte buff = new MatOfByte();
        assertEquals(0, buff.total());
        assertTrue( Imgcodecs.imencode(".jpg", gray127, buff) );
        assertFalse(0 == buff.total());
    }

    public void testImencodeStringMatListOfByteListOfInteger() {
        MatOfInt  params40 = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 40);
        MatOfInt  params90 = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        /* or
        MatOfInt  params = new MatOfInt();
        params.fromArray(Imgcodecs.IMWRITE_JPEG_QUALITY, 40);
        */
        MatOfByte buff40 = new MatOfByte();
        MatOfByte buff90 = new MatOfByte();

        assertTrue( Imgcodecs.imencode(".jpg", rgbLena, buff40, params40) );
        assertTrue( Imgcodecs.imencode(".jpg", rgbLena, buff90, params90) );

        assertTrue(buff40.total() > 0);
        assertTrue(buff40.total() < buff90.total());
    }

    public void testImreadString() {
        dst = Imgcodecs.imread(OpenCVTestRunner.LENA_PATH);
        assertTrue(!dst.empty());
        assertEquals(3, dst.channels());
        assertTrue(512 == dst.cols());
        assertTrue(512 == dst.rows());
    }

    public void testImreadStringInt() {
        dst = Imgcodecs.imread(OpenCVTestRunner.LENA_PATH, 0);
        assertTrue(!dst.empty());
        assertEquals(1, dst.channels());
        assertTrue(512 == dst.cols());
        assertTrue(512 == dst.rows());
    }

    public void testImwriteStringMat() {
        fail("Not yet implemented");
    }

    public void testImwriteStringMatListOfInteger() {
        fail("Not yet implemented");
    }

}
