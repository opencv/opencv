package org.opencv.test.video;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;
import org.opencv.utils.Converters;
import org.opencv.video.Video;

public class VideoTest extends OpenCVTestCase {

    private List<Float> err = null;
    private int h;
    private List<Point> nextPts = null;
    List<Point> prevPts = null;

    private int shift1;
    private int shift2;

    private List<Byte> status = null;
    private Mat subLena1 = null;
    private Mat subLena2 = null;
    private int w;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        shift1 = 10;
        shift2 = 17;
        w = rgbLena.cols() / 2;
        h = rgbLena.rows() / 2;

        subLena1 = rgbLena.submat(shift1, h + shift1, shift1, w + shift1);
        subLena2 = rgbLena.submat(shift2, h + shift2, shift2, w + shift2);

        prevPts = new ArrayList<Point>();
        prevPts.add(new Point(1.0, 1.0));
        prevPts.add(new Point(5.0, 5.0));
        prevPts.add(new Point(10.0, 10.0));

        nextPts = new ArrayList<Point>();
        status = new ArrayList<Byte>();
        err = new ArrayList<Float>();
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
        assertEquals(3, Core.countNonZero(Converters.vector_uchar_to_Mat(status)));
    }

    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloatSize() {
        Size sz = new Size(3, 3);
        Video.calcOpticalFlowPyrLK(subLena1, subLena2, prevPts, nextPts, status, err, sz);
        assertEquals(0, Core.countNonZero(Converters.vector_uchar_to_Mat(status)));
    }

    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloatSizeInt() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloatSizeIntTermCriteria() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloatSizeIntTermCriteriaDouble() {
        fail("Not yet implemented");
    }

    public void testCalcOpticalFlowPyrLKMatMatListOfPointListOfPointListOfByteListOfFloatSizeIntTermCriteriaDoubleInt() {
        fail("Not yet implemented");
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
