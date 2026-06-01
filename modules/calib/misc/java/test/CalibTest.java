package org.opencv.test.calib;

import java.util.ArrayList;

import org.opencv.calib.Calib;
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

public class CalibTest extends OpenCVTestCase {

    public void testConstants()
    {
        // calib3d.hpp: some constants have conflict with constants from 'fisheye' namespace
        assertEquals(1, Calib.CALIB_USE_INTRINSIC_GUESS);
        assertEquals(2, Calib.CALIB_FIX_ASPECT_RATIO);
        assertEquals(4, Calib.CALIB_FIX_PRINCIPAL_POINT);
        assertEquals(8, Calib.CALIB_ZERO_TANGENT_DIST);
        assertEquals(16, Calib.CALIB_FIX_FOCAL_LENGTH);
        assertEquals(32, Calib.CALIB_FIX_K1);
        assertEquals(64, Calib.CALIB_FIX_K2);
        assertEquals(128, Calib.CALIB_FIX_K3);
        assertEquals(0x0800, Calib.CALIB_FIX_K4);
        assertEquals(0x1000, Calib.CALIB_FIX_K5);
        assertEquals(0x2000, Calib.CALIB_FIX_K6);
        assertEquals(0x4000, Calib.CALIB_RATIONAL_MODEL);
        assertEquals(0x8000, Calib.CALIB_THIN_PRISM_MODEL);
        assertEquals(0x10000, Calib.CALIB_FIX_S1_S2_S3_S4);
        assertEquals(0x40000, Calib.CALIB_TILTED_MODEL);
        assertEquals(0x80000, Calib.CALIB_FIX_TAUX_TAUY);
        assertEquals(0x100000, Calib.CALIB_USE_QR);
        assertEquals(0x200000, Calib.CALIB_FIX_TANGENT_DIST);
        assertEquals(0x100, Calib.CALIB_FIX_INTRINSIC);
        assertEquals(0x200, Calib.CALIB_SAME_FOCAL_LENGTH);
        assertEquals(0x400, Calib.CALIB_ZERO_DISPARITY);
        assertEquals((1 << 17), Calib.CALIB_USE_LU);
        assertEquals((1 << 22), Calib.CALIB_USE_EXTRINSIC_GUESS);
    }
}
