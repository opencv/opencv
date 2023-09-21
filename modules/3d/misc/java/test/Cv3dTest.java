package org.opencv.test.cv3d;

import java.util.ArrayList;

import org.opencv.cv3d.Cv3d;
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

public class Cv3dTest extends OpenCVTestCase {

    Size size;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        size = new Size(3, 3);
    }

    public void testComposeRTMatMatMatMatMatMat() {
        Mat rvec1 = new Mat(3, 1, CvType.CV_32F);
        rvec1.put(0, 0, 0.5302828, 0.19925919, 0.40105945);
        Mat tvec1 = new Mat(3, 1, CvType.CV_32F);
        tvec1.put(0, 0, 0.81438506, 0.43713298, 0.2487897);
        Mat rvec2 = new Mat(3, 1, CvType.CV_32F);
        rvec2.put(0, 0, 0.77310503, 0.76209372, 0.30779448);
        Mat tvec2 = new Mat(3, 1, CvType.CV_32F);
        tvec2.put(0, 0, 0.70243168, 0.4784472, 0.79219002);

        Mat rvec3 = new Mat();
        Mat tvec3 = new Mat();

        Mat outRvec = new Mat(3, 1, CvType.CV_32F);
        outRvec.put(0, 0, 1.418641, 0.88665926, 0.56020796);
        Mat outTvec = new Mat(3, 1, CvType.CV_32F);
        outTvec.put(0, 0, 1.4560841, 1.0680628, 0.81598103);

        Cv3d.composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3);

        assertMatEqual(outRvec, rvec3, EPS);
        assertMatEqual(outTvec, tvec3, EPS);
    }

    public void testComposeRTMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testComposeRTMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testComposeRTMatMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testComposeRTMatMatMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testComposeRTMatMatMatMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testComposeRTMatMatMatMatMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testComposeRTMatMatMatMatMatMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testComposeRTMatMatMatMatMatMatMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
        // Mat dr3dr1;
        // Mat dr3dt1;
        // Mat dr3dr2;
        // Mat dr3dt2;
        // Mat dt3dr1;
        // Mat dt3dt1;
        // Mat dt3dr2;
        // Mat dt3dt2;
        // , dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2);
        // [0.97031879, -0.091774099, 0.38594806;
        // 0.15181915, 0.98091727, -0.44186208;
        // -0.39509675, 0.43839464, 0.93872648]
        // [0, 0, 0;
        // 0, 0, 0;
        // 0, 0, 0]
        // [1.0117353, 0.16348237, -0.083180845;
        // -0.1980398, 1.006078, 0.30299222;
        // 0.075766489, -0.32784501, 1.0163091]
        // [0, 0, 0;
        // 0, 0, 0;
        // 0, 0, 0]
        // [0, 0, 0;
        // 0, 0, 0;
        // 0, 0, 0]
        // [0.69658804, 0.018115902, 0.7172426;
        // 0.51114357, 0.68899536, -0.51382649;
        // -0.50348526, 0.72453934, 0.47068608]
        // [0.18536358, -0.20515044, -0.48834875;
        // -0.25120571, 0.29043972, 0.60573936;
        // 0.35370794, -0.69923931, 0.45781645]
        // [1, 0, 0;
        // 0, 1, 0;
        // 0, 0, 1]
    }

    public void testConvertPointsFromHomogeneous() {
        fail("Not yet implemented");
    }

    public void testConvertPointsToHomogeneous() {
        fail("Not yet implemented");
    }

    public void testDecomposeProjectionMatrixMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testDecomposeProjectionMatrixMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testDecomposeProjectionMatrixMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testDecomposeProjectionMatrixMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testDecomposeProjectionMatrixMatMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testEstimateAffine3DMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testEstimateAffine3DMatMatMatMatDouble() {
        fail("Not yet implemented");
    }

    public void testEstimateAffine3DMatMatMatMatDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testFindFundamentalMatListOfPointListOfPointInt() {
        fail("Not yet implemented");
    }

    public void testFindFundamentalMatListOfPointListOfPointIntDouble() {
        fail("Not yet implemented");
    }

    public void testFindFundamentalMatListOfPointListOfPointIntDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testFindFundamentalMatListOfPointListOfPointIntDoubleDoubleMat() {
        fail("Not yet implemented");
    }

    public void testFindHomographyListOfPointListOfPoint() {
        final int NUM = 20;

        MatOfPoint2f originalPoints = new MatOfPoint2f();
        originalPoints.alloc(NUM);
        MatOfPoint2f transformedPoints = new MatOfPoint2f();
        transformedPoints.alloc(NUM);

        for (int i = 0; i < NUM; i++) {
            double x = Math.random() * 100 - 50;
            double y = Math.random() * 100 - 50;
            originalPoints.put(i, 0, x, y);
            transformedPoints.put(i, 0, y, x);
        }

        Mat hmg = Cv3d.findHomography(originalPoints, transformedPoints);

        truth = new Mat(3, 3, CvType.CV_64F);
        truth.put(0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1);

        assertMatEqual(truth, hmg, EPS);
    }

    public void testFindHomographyListOfPointListOfPointInt() {
        fail("Not yet implemented");
    }

    public void testFindHomographyListOfPointListOfPointIntDouble() {
        fail("Not yet implemented");
    }

    public void testFindHomographyListOfPointListOfPointIntDoubleMat() {
        fail("Not yet implemented");
    }

    public void testGetOptimalNewCameraMatrixMatMatSizeDouble() {
        fail("Not yet implemented");
    }

    public void testGetOptimalNewCameraMatrixMatMatSizeDoubleSize() {
        fail("Not yet implemented");
    }

    public void testGetOptimalNewCameraMatrixMatMatSizeDoubleSizeRect() {
        fail("Not yet implemented");
    }

    public void testGetOptimalNewCameraMatrixMatMatSizeDoubleSizeRectBoolean() {
        fail("Not yet implemented");
    }

    public void testGetValidDisparityROI() {
        fail("Not yet implemented");
    }

    public void testMatMulDeriv() {
        fail("Not yet implemented");
    }

    public void testProjectPointsMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testProjectPointsMatMatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testProjectPointsMatMatMatMatMatMatMatDouble() {
        fail("Not yet implemented");
    }

    public void testRectify3Collinear() {
        fail("Not yet implemented");
    }

    public void testRodriguesMatMat() {
        Mat r = new Mat(3, 1, CvType.CV_32F);
        Mat R = new Mat(3, 3, CvType.CV_32F);

        r.put(0, 0, Math.PI, 0, 0);

        Cv3d.Rodrigues(r, R);

        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1);
        assertMatEqual(truth, R, EPS);

        Mat r2 = new Mat();
        Cv3d.Rodrigues(R, r2);

        assertMatEqual(r, r2, EPS);
    }

    public void testRodriguesMatMatMat() {
        fail("Not yet implemented");
    }

    public void testRQDecomp3x3MatMatMat() {
        fail("Not yet implemented");
    }

    public void testRQDecomp3x3MatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testRQDecomp3x3MatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testRQDecomp3x3MatMatMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testSolvePnPListOfPoint3ListOfPointMatMatMatMat() {
        Mat intrinsics = Mat.eye(3, 3, CvType.CV_64F);
        intrinsics.put(0, 0, 400);
        intrinsics.put(1, 1, 400);
        intrinsics.put(0, 2, 640 / 2);
        intrinsics.put(1, 2, 480 / 2);

        final int minPnpPointsNum = 4;

        MatOfPoint3f points3d = new MatOfPoint3f();
        points3d.alloc(minPnpPointsNum);
        MatOfPoint2f points2d = new MatOfPoint2f();
        points2d.alloc(minPnpPointsNum);

        for (int i = 0; i < minPnpPointsNum; i++) {
            double x = Math.random() * 100 - 50;
            double y = Math.random() * 100 - 50;
            points2d.put(i, 0, x, y); //add(new Point(x, y));
            points3d.put(i, 0, 0, y, x); // add(new Point3(0, y, x));
        }

        Mat rvec = new Mat();
        Mat tvec = new Mat();
        Cv3d.solvePnP(points3d, points2d, intrinsics, new MatOfDouble(), rvec, tvec);

        Mat truth_rvec = new Mat(3, 1, CvType.CV_64F);
        truth_rvec.put(0, 0, 0, Math.PI / 2, 0);

        Mat truth_tvec = new Mat(3, 1, CvType.CV_64F);
        truth_tvec.put(0, 0, -320, -240, 400);

        assertMatEqual(truth_rvec, rvec, EPS*2);
        assertMatEqual(truth_tvec, tvec, EPS*2);
    }

    public void testSolvePnPListOfPoint3ListOfPointMatMatMatMatBoolean() {
        fail("Not yet implemented");
    }

    public void testSolvePnPRansacListOfPoint3ListOfPointMatMatMatMat() {
        fail("Not yet implemented");
    }

    public void testSolvePnPRansacListOfPoint3ListOfPointMatMatMatMatBoolean() {
        fail("Not yet implemented");
    }

    public void testSolvePnPRansacListOfPoint3ListOfPointMatMatMatMatBooleanInt() {
        fail("Not yet implemented");
    }

    public void testSolvePnPRansacListOfPoint3ListOfPointMatMatMatMatBooleanIntFloat() {
        fail("Not yet implemented");
    }

    public void testSolvePnPRansacListOfPoint3ListOfPointMatMatMatMatBooleanIntFloatInt() {
        fail("Not yet implemented");
    }

    public void testSolvePnPRansacListOfPoint3ListOfPointMatMatMatMatBooleanIntFloatIntMat() {
        fail("Not yet implemented");
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

    public void testComputeCorrespondEpilines()
    {
        Mat fundamental = new Mat(3, 3, CvType.CV_64F);
        fundamental.put(0, 0, 0, -0.577, 0.288, 0.577, 0, 0.288, -0.288, -0.288, 0);
        MatOfPoint2f left = new MatOfPoint2f();
        left.alloc(1);
        left.put(0, 0, 2, 3); //add(new Point(x, y));
        Mat lines = new Mat();
        Mat truth = new Mat(1, 1, CvType.CV_32FC3);
        truth.put(0, 0, -0.70735186, 0.70686162, -0.70588124);
        Cv3d.computeCorrespondEpilines(left, 1, fundamental, lines);
        assertMatEqual(truth, lines, EPS);
    }

    public void testSolvePnPGeneric_regression_16040() {
        Mat intrinsics = Mat.eye(3, 3, CvType.CV_64F);
        intrinsics.put(0, 0, 400);
        intrinsics.put(1, 1, 400);
        intrinsics.put(0, 2, 640 / 2);
        intrinsics.put(1, 2, 480 / 2);

        final int minPnpPointsNum = 4;

        MatOfPoint3f points3d = new MatOfPoint3f();
        points3d.alloc(minPnpPointsNum);
        MatOfPoint2f points2d = new MatOfPoint2f();
        points2d.alloc(minPnpPointsNum);

        for (int i = 0; i < minPnpPointsNum; i++) {
            double x = Math.random() * 100 - 50;
            double y = Math.random() * 100 - 50;
            points2d.put(i, 0, x, y); //add(new Point(x, y));
            points3d.put(i, 0, 0, y, x); // add(new Point3(0, y, x));
        }

        ArrayList<Mat> rvecs = new ArrayList<Mat>();
        ArrayList<Mat> tvecs = new ArrayList<Mat>();

        Mat rvec = new Mat();
        Mat tvec = new Mat();

        Mat reprojectionError = new Mat(2, 1, CvType.CV_64FC1);

        Cv3d.solvePnPGeneric(points3d, points2d, intrinsics, new MatOfDouble(), rvecs, tvecs, false, Cv3d.SOLVEPNP_IPPE, rvec, tvec, reprojectionError);

        Mat truth_rvec = new Mat(3, 1, CvType.CV_64F);
        truth_rvec.put(0, 0, 0, Math.PI / 2, 0);

        Mat truth_tvec = new Mat(3, 1, CvType.CV_64F);
        truth_tvec.put(0, 0, -320, -240, 400);

        assertMatEqual(truth_rvec, rvecs.get(0), 10 * EPS);
        assertMatEqual(truth_tvec, tvecs.get(0), 1000 * EPS);
    }

    public void testGetDefaultNewCameraMatrixMat() {
        Mat mtx = Cv3d.getDefaultNewCameraMatrix(gray0);

        assertFalse(mtx.empty());
        assertEquals(0, Core.countNonZero(mtx));
    }

    public void testGetDefaultNewCameraMatrixMatSizeBoolean() {
        Mat mtx = Cv3d.getDefaultNewCameraMatrix(gray0, size, true);

        assertFalse(mtx.empty());
        assertFalse(0 == Core.countNonZero(mtx));
        // TODO_: write better test
    }

    public void testInitUndistortRectifyMap() {
        fail("Not yet implemented");
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 1);
        cameraMatrix.put(2, 0, 0, 0, 1);

        Mat R = new Mat(3, 3, CvType.CV_32F, new Scalar(2));
        Mat newCameraMatrix = new Mat(3, 3, CvType.CV_32F, new Scalar(3));

        Mat distCoeffs = new Mat();
        Mat map1 = new Mat();
        Mat map2 = new Mat();

        // TODO: complete this test
        Cv3d.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, CvType.CV_32F, map1, map2);
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMat() {
        fail("Not yet implemented");
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F);
        // Size imageSize = new Size(2, 2);

        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 2);
        cameraMatrix.put(2, 0, 0, 0, 1);

        distCoeffs.put(0, 0, 1, 3, 2, 4);
        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 0, 0);
        truth.put(2, 0, 0, 3, 0);
        // TODO: No documentation for this function
        // Cv3d.initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize,
        // 5, m1type, truthput1, truthput2);
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMatInt() {
        fail("Not yet implemented");
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMatIntDouble() {
        fail("Not yet implemented");
    }

    public void testUndistortMatMatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 0, 1);
                put(1, 0, 0, 1, 2);
                put(2, 0, 0, 0, 1);
            }
        };
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 3, 2, 4);
            }
        };

        Cv3d.undistort(src, dst, cameraMatrix, distCoeffs);

        truth = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 0, 0, 0);
                put(1, 0, 0, 0, 0);
                put(2, 0, 0, 3, 0);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testUndistortMatMatMatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 0, 1);
                put(1, 0, 0, 1, 2);
                put(2, 0, 0, 0, 1);
            }
        };
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 1, 4, 5);
            }
        };
        Mat newCameraMatrix = new Mat(3, 3, CvType.CV_32F, new Scalar(1));

        Cv3d.undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix);

        truth = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
        assertMatEqual(truth, dst, EPS);
    }

    //undistortPoints(List<Point> src, List<Point> dst, Mat cameraMatrix, Mat distCoeffs)
    public void testUndistortPointsListOfPointListOfPointMatMat() {
        MatOfPoint2f src = new MatOfPoint2f(new Point(1, 2), new Point(3, 4), new Point(-1, -1));
        MatOfPoint2f dst = new MatOfPoint2f();
        Mat cameraMatrix = Mat.eye(3, 3, CvType.CV_64FC1);
        Mat distCoeffs = new Mat(8, 1, CvType.CV_64FC1, new Scalar(0));

        Cv3d.undistortPoints(src, dst, cameraMatrix, distCoeffs);

        assertEquals(src.cols(), dst.rows());
        assertEquals(src.rows(), dst.cols());
        for(int i=0; i<src.toList().size(); i++) {
            //Log.d("UndistortPoints", "s="+src.get(i)+", d="+dst.get(i));
            assertTrue(src.toList().get(i).equals(dst.toList().get(i)));
        }
    }

    public void testEstimateNewCameraMatrixForUndistortRectify() {
        Mat K = new Mat().eye(3, 3, CvType.CV_64FC1);
        Mat K_new = new Mat().eye(3, 3, CvType.CV_64FC1);
        Mat K_new_truth = new Mat().eye(3, 3, CvType.CV_64FC1);
        Mat D = new Mat().zeros(4, 1, CvType.CV_64FC1);

        K.put(0,0,600.4447738238429);
        K.put(1,1,578.9929805505851);
        K.put(0,2,992.0642578801213);
        K.put(1,2,549.2682624212172);

        D.put(0,0,-0.05090103223466704);
        D.put(1,0,0.030944413642173308);
        D.put(2,0,-0.021509225493198905);
        D.put(3,0,0.0043378096628297145);

        K_new_truth.put(0,0, 387.5118215642316);
        K_new_truth.put(0,2, 1033.936556777084);
        K_new_truth.put(1,1, 373.6673784974842);
        K_new_truth.put(1,2, 538.794152656429);

        Cv3d.fisheye_estimateNewCameraMatrixForUndistortRectify(K,D,new Size(1920,1080),
                    new Mat().eye(3, 3, CvType.CV_64F), K_new, 0.0, new Size(1920,1080));

        assertMatEqual(K_new, K_new_truth, EPS);
    }
}
