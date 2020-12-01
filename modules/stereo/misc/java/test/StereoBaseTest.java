package org.opencv.test.stereoBase;

import java.util.ArrayList;

import org.opencv.cv3d.Cv3d;
import org.opencv.stereo.Stereo;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;
import org.opencv.imgproc.Imgproc;

public class StereoBaseTest extends OpenCVTestCase {

    Size size;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        size = new Size(3, 3);
    }

    public void testFilterSpecklesMatDoubleIntDouble() {
        gray_16s_1024.copyTo(dst);
        Point center = new Point(gray_16s_1024.rows() / 2., gray_16s_1024.cols() / 2.);
        Imgproc.circle(dst, center, 1, Scalar.all(4096));

        assertMatNotEqual(gray_16s_1024, dst);
        Stereo.filterSpeckles(dst, 1024.0, 100, 0.);
        assertMatEqual(gray_16s_1024, dst);
    }

    public void testFilterSpecklesMatDoubleIntDoubleMat() {
        fail("Not yet implemented");
    }

    public void testGetValidDisparityROI() {
        fail("Not yet implemented");
    }

    public void testRectify3Collinear() {
        fail("Not yet implemented");
    }

    public void testReprojectImageTo3DMatMatMat() {
        Mat transformMatrix = new Mat(4, 4, CvType.CV_64F);
        transformMatrix.put(0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

        Mat disparity = new Mat(matSize, matSize, CvType.CV_32F);

        float[] disp = new float[matSize * matSize];
        for (int i = 0; i < matSize; i++)
            for (int j = 0; j < matSize; j++)
                disp[i * matSize + j] = i - j;
        disparity.put(0, 0, disp);

        Mat _3dPoints = new Mat();

        Stereo.reprojectImageTo3D(disparity, _3dPoints, transformMatrix);

        assertEquals(CvType.CV_32FC3, _3dPoints.type());
        assertEquals(matSize, _3dPoints.rows());
        assertEquals(matSize, _3dPoints.cols());

        truth = new Mat(matSize, matSize, CvType.CV_32FC3);

        float[] _truth = new float[matSize * matSize * 3];
        for (int i = 0; i < matSize; i++)
            for (int j = 0; j < matSize; j++) {
                _truth[(i * matSize + j) * 3 + 0] = i;
                _truth[(i * matSize + j) * 3 + 1] = j;
                _truth[(i * matSize + j) * 3 + 2] = i - j;
            }
        truth.put(0, 0, _truth);

        assertMatEqual(truth, _3dPoints, EPS);
    }

    public void testReprojectImageTo3DMatMatMatBoolean() {
        Mat transformMatrix = new Mat(4, 4, CvType.CV_64F);
        transformMatrix.put(0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

        Mat disparity = new Mat(matSize, matSize, CvType.CV_32F);

        float[] disp = new float[matSize * matSize];
        for (int i = 0; i < matSize; i++)
            for (int j = 0; j < matSize; j++)
                disp[i * matSize + j] = i - j;
        disp[0] = -Float.MAX_VALUE;
        disparity.put(0, 0, disp);

        Mat _3dPoints = new Mat();

        Stereo.reprojectImageTo3D(disparity, _3dPoints, transformMatrix, true);

        assertEquals(CvType.CV_32FC3, _3dPoints.type());
        assertEquals(matSize, _3dPoints.rows());
        assertEquals(matSize, _3dPoints.cols());

        truth = new Mat(matSize, matSize, CvType.CV_32FC3);

        float[] _truth = new float[matSize * matSize * 3];
        for (int i = 0; i < matSize; i++)
            for (int j = 0; j < matSize; j++) {
                _truth[(i * matSize + j) * 3 + 0] = i;
                _truth[(i * matSize + j) * 3 + 1] = j;
                _truth[(i * matSize + j) * 3 + 2] = i - j;
            }
        _truth[2] = 10000;
        truth.put(0, 0, _truth);

        assertMatEqual(truth, _3dPoints, EPS);
    }

    public void testReprojectImageTo3DMatMatMatBooleanInt() {
        Mat transformMatrix = new Mat(4, 4, CvType.CV_64F);
        transformMatrix.put(0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

        Mat disparity = new Mat(matSize, matSize, CvType.CV_32F);

        float[] disp = new float[matSize * matSize];
        for (int i = 0; i < matSize; i++)
            for (int j = 0; j < matSize; j++)
                disp[i * matSize + j] = i - j;
        disparity.put(0, 0, disp);

        Mat _3dPoints = new Mat();

        Stereo.reprojectImageTo3D(disparity, _3dPoints, transformMatrix, false, CvType.CV_16S);

        assertEquals(CvType.CV_16SC3, _3dPoints.type());
        assertEquals(matSize, _3dPoints.rows());
        assertEquals(matSize, _3dPoints.cols());

        truth = new Mat(matSize, matSize, CvType.CV_16SC3);

        short[] _truth = new short[matSize * matSize * 3];
        for (short i = 0; i < matSize; i++)
            for (short j = 0; j < matSize; j++) {
                _truth[(i * matSize + j) * 3 + 0] = i;
                _truth[(i * matSize + j) * 3 + 1] = j;
                _truth[(i * matSize + j) * 3 + 2] = (short) (i - j);
            }
        truth.put(0, 0, _truth);

        assertMatEqual(truth, _3dPoints, EPS);
    }

    public void testStereoCalibrateListOfMatListOfMatListOfMatMatMatMatMatSizeMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testStereoCalibrateListOfMatListOfMatListOfMatMatMatMatMatSizeMatMatMatMatTermCriteria() {
        fail("Not yet implemented");
    }

    public void testStereoCalibrateListOfMatListOfMatListOfMatMatMatMatMatSizeMatMatMatMatTermCriteriaInt() {
        fail("Not yet implemented");
    }

    public void testStereoRectifyUncalibratedMatMatMatSizeMatMat() {
        fail("Not yet implemented");
    }

    public void testStereoRectifyUncalibratedMatMatMatSizeMatMatDouble() {
        fail("Not yet implemented");
    }

    public void testValidateDisparityMatMatIntInt() {
        fail("Not yet implemented");
    }

    public void testValidateDisparityMatMatIntIntInt() {
        fail("Not yet implemented");
    }
}
