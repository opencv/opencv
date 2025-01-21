package org.opencv.test.imgcodecs;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgcodecs.Animation;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class ImgcodecsTest extends OpenCVTestCase {

    public void testAnimation() {
        Mat src = Imgcodecs.imread(OpenCVTestRunner.LENA_PATH, Imgcodecs.IMREAD_REDUCED_COLOR_4);
        assertFalse(src.empty());

        Mat rgb = new Mat();
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_BGR2RGB);

        Animation animation = new Animation();
        List<Mat> frames = new ArrayList<>();
        MatOfInt durations = new MatOfInt();

        frames.add(src);
        durations.add(100);
        frames.add(rgb);
        durations.add(100);

        animation.set_frames(frames);
        animation.set_durations(durations);

        assertTrue(Imgcodecs.imwriteanimation("animationtest.png", animation));
        assertTrue(Imgcodecs.imreadanimation("animationtest.png", animation));
    }

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
        assertFalse(dst.empty());
        assertEquals(3, dst.channels());
        assertTrue(512 == dst.cols());
        assertTrue(512 == dst.rows());
    }

    public void testImreadStringInt() {
        dst = Imgcodecs.imread(OpenCVTestRunner.LENA_PATH, Imgcodecs.IMREAD_GRAYSCALE);
        assertFalse(dst.empty());
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
