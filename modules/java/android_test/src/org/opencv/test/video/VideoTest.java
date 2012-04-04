package org.opencv.test.video;

import org.opencv.core.Core;
import org.opencv.core.CvVectorByte;
import org.opencv.core.CvVectorFloat;
import org.opencv.core.CvVectorPoint2f;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;
import org.opencv.video.Video;

public class VideoTest extends OpenCVTestCase {

    private CvVectorFloat err = null;
    private int h;
    private CvVectorPoint2f nextPts = null;
    private CvVectorPoint2f prevPts = null;

    private int shift1;
    private int shift2;

    private CvVectorByte status = null;
    private Mat subLena1 = null;
    private Mat subLena2 = null;
    private int w;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        shift1 = 10;
        shift2 = 17;
        w = (int)(rgbLena.cols() / 2);
        h = (int)(rgbLena.rows() / 2);

        subLena1 = rgbLena.submat(shift1, h + shift1, shift1, w + shift1);
        subLena2 = rgbLena.submat(shift2, h + shift2, shift2, w + shift2);

        prevPts = new CvVectorPoint2f(new Point(11d, 8d), new Point(5d, 5d), new Point(10d, 10d));

        nextPts = new CvVectorPoint2f();
        status = new CvVectorByte();
        err = new CvVectorFloat();
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

    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloat() {
        Video.calcOpticalFlowPyrLK(subLena1, subLena2, prevPts, nextPts, status, err);
        assertEquals(3, Core.countNonZero(status));
    }

    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloatSize() {
        Size sz = new Size(3, 3);
        Video.calcOpticalFlowPyrLK(subLena1, subLena2, prevPts, nextPts, status, err, sz, 3);
        assertEquals(0, Core.countNonZero(status));
    }


    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloatSizeIntTermCriteriaDoubleIntDouble() {
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
