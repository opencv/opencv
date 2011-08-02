package org.opencv.test.video;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.video.Video;
import org.opencv.test.OpenCVTestCase;

public class videoTest extends OpenCVTestCase {

    private int shift1;
    private int shift2;
    private int w;
    private int h;

    private Mat subLena1 = null;
    private Mat subLena2 = null;

    private Mat nextPts = null;
    private Mat status = null;
    private Mat err = null;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        shift1 = 10;
        shift2 = 17;
        w = rgbLena.cols() / 2;
        h = rgbLena.rows() / 2;

        subLena1 = rgbLena.submat(shift1, h + shift1, shift1, w + shift1);
        subLena2 = rgbLena.submat(shift2, h + shift2, shift2, w + shift2);

        nextPts = new Mat();
        status = new Mat();
        err = new Mat();
    }

    public void testCalcGlobalOrientation() {
        fail("Not yet implemented");
    }

    public void testCalcMotionGradientMatMatMatDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testCalcMotionGradientMatMatMatDoubleDoubleInt() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowFarneback() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowPyrLKMatMatMatMatMatMat() {
        Mat prevPts = new Mat(1, 3, CvType.CV_32FC2);
        prevPts.put(0, 0, 1.0, 1.0, 5.0, 5.0, 10.0, 10.0);

        Video.calcOpticalFlowPyrLK(subLena1, subLena2, prevPts, nextPts, status, err);
        assertEquals(3, Core.countNonZero(status));
    }

    public void testCalcOpticalFlowPyrLKMatMatMatMatMatMatSize() {
        Mat prevPts = new Mat(1, 3, CvType.CV_32FC2);
        prevPts.put(0, 0, 1.0, 1.0, 5.0, 5.0, 10.0, 10.0);

        Size sz = new Size(5, 5);
        Video.calcOpticalFlowPyrLK(subLena1, subLena2, prevPts, nextPts, status, err, sz);
        assertEquals(0, Core.countNonZero(status));
    }

    public void testCalcOpticalFlowPyrLKMatMatMatMatMatMatSizeInt() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowPyrLKMatMatMatMatMatMatSizeIntTermCriteria() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowPyrLKMatMatMatMatMatMatSizeIntTermCriteriaDouble() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowPyrLKMatMatMatMatMatMatSizeIntTermCriteriaDoubleInt() {
        fail("Not yet implemented");
    }

    public void testCamShift() {
        fail("Not yet implemented");
    }

    public void testEstimateRigidTransform() {
        fail("Not yet implemented");
    }

    public void testMeanShift() {
        fail("Not yet implemented");
    }

    public void testSegmentMotion() {
        fail("Not yet implemented");
    }

    public void testUpdateMotionHistory() {
        fail("Not yet implemented");
    }

}
