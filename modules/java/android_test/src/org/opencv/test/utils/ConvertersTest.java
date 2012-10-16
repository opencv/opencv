package org.opencv.test.utils;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

public class ConvertersTest extends OpenCVTestCase {

    public void testMat_to_vector_char() {
        Mat src = new Mat(3, 1, CvType.CV_8SC1);
        src.put(0, 0, 2, 4, 3);
        List<Byte> bs = new ArrayList<Byte>();

        Converters.Mat_to_vector_char(src, bs);

        List<Byte> truth = new ArrayList<Byte>();
        byte value1 = 2;
        byte value2 = 4;
        byte value3 = 3;
        truth.add(new Byte(value1));
        truth.add(new Byte(value2));
        truth.add(new Byte(value3));
        assertEquals(truth, bs);
    }

    public void testMat_to_vector_DMatch() {
        Mat src = new Mat(4, 1, CvType.CV_64FC4);
        src.put(0, 0, 1, 4, 4, 10, 2, 3, 5, 6, 3, 1, 8, 12, 4, 9, 5, 15);
        List<DMatch> matches = new ArrayList<DMatch>();

        Converters.Mat_to_vector_DMatch(src, matches);

        List<DMatch> truth = new ArrayList<DMatch>();
        truth.add(new DMatch(1, 4, 4, 10));
        truth.add(new DMatch(2, 3, 5, 6));
        truth.add(new DMatch(3, 1, 8, 12));
        truth.add(new DMatch(4, 9, 5, 15));
        //assertListDMatchEquals(truth, matches, EPS);
        fail("Not yet implemented");
    }

    public void testMat_to_vector_float() {
        Mat src = new Mat(4, 1, CvType.CV_32FC1);
        src.put(0, 0, 2, 4, 3, 9);
        List<Float> fs = new ArrayList<Float>();

        Converters.Mat_to_vector_float(src, fs);

        List<Float> truth = new ArrayList<Float>();
        truth.add(2.0f);
        truth.add(4.0f);
        truth.add(3.0f);
        truth.add(9.0f);
        assertListEquals(truth, fs, EPS);
    }

    public void testMat_to_vector_int() {
        Mat src = new Mat(4, 1, CvType.CV_32SC1);
        src.put(0, 0, 2, 4, 3, 9);
        List<Integer> fs = new ArrayList<Integer>();

        Converters.Mat_to_vector_int(src, fs);
        List<Integer> truth = new ArrayList<Integer>();
        truth.add(2);
        truth.add(4);
        truth.add(3);
        truth.add(9);
        assertListEquals(truth, fs);
    }

    public void testMat_to_vector_KeyPoint() {
        Mat src = new Mat(1, 1, CvType.CV_64FC(7));
        src.put(0, 0, 2, 4, 3, 9, 10, 12, 7);
        List<KeyPoint> kps = new ArrayList<KeyPoint>();

        Converters.Mat_to_vector_KeyPoint(src, kps);

        List<KeyPoint> truth = new ArrayList<KeyPoint>();
        truth.add(new KeyPoint(2, 4, 3, 9, 10, 12, 7));
        assertListKeyPointEquals(truth, kps, EPS);
    }

    public void testMat_to_vector_Mat() {
        // Mat src = new Mat(4, 1, CvType.CV_32SC2);
        // src.put(0, 0, 2, 2, 3, 3, 4, 4, 5, 5);
        //
        // List<Mat> mats = new ArrayList<Mat>();
        // Converters.Mat_to_vector_Mat(src, mats);
        //
        // List<Mat> truth = new ArrayList<Mat>();
        // truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        // truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(3.0)));
        // truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(4.0)));
        // truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(5.0)));
        // assertListEqualMat(truth, mats, EPS);
        fail("Not yet implemented");
    }

    public void testMat_to_vector_Point() {
        Mat src = new Mat(4, 1, CvType.CV_32SC2);
        src.put(0, 0, 2, 4, 3, 9, 10, 4, 35, 54);
        List<Point> points = new ArrayList<Point>();

        Converters.Mat_to_vector_Point(src, points);

        List<Point> truth = new ArrayList<Point>();
        truth.add(new Point(2, 4));
        truth.add(new Point(3, 9));
        truth.add(new Point(10, 4));
        truth.add(new Point(35, 54));
        assertListPointEquals(truth, points, EPS);
    }

    public void testMat_to_vector_Point2d() {
        Mat src = new Mat(4, 1, CvType.CV_64FC2);
        src.put(0, 0, 12.0, 4.0, 3.0, 29.0, 10.0, 24.0, 35.0, 54.0);
        List<Point> points = new ArrayList<Point>();

        Converters.Mat_to_vector_Point2d(src, points);

        List<Point> truth = new ArrayList<Point>();
        truth.add(new Point(12.0, 4.0));
        truth.add(new Point(3.0, 29.0));
        truth.add(new Point(10.0, 24.0));
        truth.add(new Point(35.0, 54.0));
        assertListPointEquals(truth, points, EPS);
    }

    public void testMat_to_vector_Point2f() {
        Mat src = new Mat(4, 1, CvType.CV_32FC2);
        src.put(0, 0, 2, 14, 31, 19, 10, 44, 5, 41);
        List<Point> points = new ArrayList<Point>();

        Converters.Mat_to_vector_Point(src, points);

        List<Point> truth = new ArrayList<Point>();
        truth.add(new Point(2, 14));
        truth.add(new Point(31, 19));
        truth.add(new Point(10, 44));
        truth.add(new Point(5, 41));
        assertListPointEquals(truth, points, EPS);
    }

    public void testMat_to_vector_Point3() {
        Mat src = new Mat(4, 1, CvType.CV_32SC3);
        src.put(0, 0, 2, 14, 12, 31, 19, 22, 10, 44, 45, 5, 41, 31);
        List<Point3> points = new ArrayList<Point3>();

        Converters.Mat_to_vector_Point3(src, points);

        List<Point3> truth = new ArrayList<Point3>();
        truth.add(new Point3(2, 14, 12));
        truth.add(new Point3(31, 19, 22));
        truth.add(new Point3(10, 44, 45));
        truth.add(new Point3(5, 41, 31));
        assertListPoint3Equals(truth, points, EPS);
    }

    public void testMat_to_vector_Point3d() {
        Mat src = new Mat(4, 1, CvType.CV_64FC3);
        src.put(0, 0, 2.0, 4.0, 3.0, 5.0, 9.0, 12.0, 10.0, 14.0, 15.0, 5.0, 11.0, 31.0);
        List<Point3> points = new ArrayList<Point3>();

        Converters.Mat_to_vector_Point3(src, points);

        List<Point3> truth = new ArrayList<Point3>();
        truth.add(new Point3(2.0, 4.0, 3.0));
        truth.add(new Point3(5.0, 9.0, 12.0));
        truth.add(new Point3(10.0, 14.0, 15.0));
        truth.add(new Point3(5.0, 11.0, 31.0));
        assertListPoint3Equals(truth, points, EPS);
    }

    public void testMat_to_vector_Point3f() {
        Mat src = new Mat(4, 1, CvType.CV_32FC3);
        src.put(0, 0, 2.0, 4.0, 3.0, 5.0, 9.0, 12.0, 10.0, 14.0, 15.0, 5.0, 11.0, 31.0);
        List<Point3> points = new ArrayList<Point3>();

        Converters.Mat_to_vector_Point3(src, points);

        List<Point3> truth = new ArrayList<Point3>();
        truth.add(new Point3(2.0, 4.0, 3.0));
        truth.add(new Point3(5.0, 9.0, 12.0));
        truth.add(new Point3(10.0, 14.0, 15.0));
        truth.add(new Point3(5.0, 11.0, 31.0));
        assertListPoint3Equals(truth, points, EPS);
    }

    public void testMat_to_vector_Point3i() {
        Mat src = new Mat(4, 1, CvType.CV_32SC3);
        src.put(0, 0, 2, 14, 12, 31, 19, 22, 10, 44, 45, 5, 41, 31);
        List<Point3> points = new ArrayList<Point3>();

        Converters.Mat_to_vector_Point3(src, points);

        List<Point3> truth = new ArrayList<Point3>();
        truth.add(new Point3(2, 14, 12));
        truth.add(new Point3(31, 19, 22));
        truth.add(new Point3(10, 44, 45));
        truth.add(new Point3(5, 41, 31));
        assertListPoint3Equals(truth, points, EPS);
    }

    public void testMat_to_vector_Rect() {
        Mat src = new Mat(2, 1, CvType.CV_32SC4);
        src.put(0, 0, 2, 2, 5, 2, 0, 0, 6, 4);
        List<Rect> rectangles = new ArrayList<Rect>();

        Converters.Mat_to_vector_Rect(src, rectangles);
        List<Rect> truth = new ArrayList<Rect>();
        truth.add(new Rect(2, 2, 5, 2));
        truth.add(new Rect(0, 0, 6, 4));
        assertListRectEquals(truth, rectangles);
    }

    public void testMat_to_vector_uchar() {
        Mat src = new Mat(3, 1, CvType.CV_8UC1);
        src.put(0, 0, 2, 4, 3);
        List<Byte> bs = new ArrayList<Byte>();

        Converters.Mat_to_vector_uchar(src, bs);

        List<Byte> truth = new ArrayList<Byte>();
        byte value1 = 2;
        byte value2 = 4;
        byte value3 = 3;
        truth.add(new Byte(value1));
        truth.add(new Byte(value2));
        truth.add(new Byte(value3));
        assertEquals(truth, bs);
    }

    public void testMat_to_vector_vector_char() {
        fail("Not yet implemented");
    }

    public void testMat_to_vector_vector_DMatch() {
        fail("Not yet implemented");
    }

    public void testMat_to_vector_vector_KeyPoint() {
        fail("Not yet implemented");
    }

    public void testMat_to_vector_vector_Point2f() {
        fail("Not yet implemented");
    }

    public void testVector_char_to_Mat() {
        List<Byte> bytes = new ArrayList<Byte>();
        byte value1 = 1;
        byte value2 = 2;
        byte value3 = 3;
        byte value4 = 4;
        bytes.add(new Byte(value1));
        bytes.add(new Byte(value2));
        bytes.add(new Byte(value3));
        bytes.add(new Byte(value4));

        dst = Converters.vector_char_to_Mat(bytes);
        truth = new Mat(4, 1, CvType.CV_8SC1);
        truth.put(0, 0, 1, 2, 3, 4);
        assertMatEqual(truth, dst);

    }

    public void testVector_DMatch_to_Mat() {
        List<DMatch> matches = new ArrayList<DMatch>();
        matches.add(new DMatch(1, 4, 4, 10));
        matches.add(new DMatch(2, 3, 5, 6));
        matches.add(new DMatch(3, 1, 8, 12));
        matches.add(new DMatch(4, 9, 5, 15));

        dst = Converters.vector_DMatch_to_Mat(matches);

        Mat truth = new Mat(4, 1, CvType.CV_64FC4);
        truth.put(0, 0, 1, 4, 4, 10, 2, 3, 5, 6, 3, 1, 8, 12, 4, 9, 5, 15);
        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_double_to_Mat() {
        List<Double> inputVector = new ArrayList<Double>();
        inputVector.add(2.0);
        inputVector.add(4.0);
        inputVector.add(3.0);
        inputVector.add(9.0);

        dst = Converters.vector_double_to_Mat(inputVector);
        truth = new Mat(4, 1, CvType.CV_64FC1);
        truth.put(0, 0, 2, 4, 3, 9);
        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_float_to_Mat() {
        List<Float> inputVector = new ArrayList<Float>();
        inputVector.add(2.0f);
        inputVector.add(4.0f);
        inputVector.add(3.0f);
        inputVector.add(9.0f);

        dst = Converters.vector_float_to_Mat(inputVector);
        truth = new Mat(4, 1, CvType.CV_32FC1);
        truth.put(0, 0, 2, 4, 3, 9);
        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_int_to_Mat() {
        List<Integer> inputVector = new ArrayList<Integer>();
        inputVector.add(2);
        inputVector.add(4);
        inputVector.add(3);
        inputVector.add(9);

        dst = Converters.vector_int_to_Mat(inputVector);
        truth = new Mat(4, 1, CvType.CV_32SC1);
        truth.put(0, 0, 2, 4, 3, 9);
        assertMatEqual(truth, dst);
    }

    public void testVector_KeyPoint_to_Mat() {
        List<KeyPoint> kps = new ArrayList<KeyPoint>();
        kps.add(new KeyPoint(2, 4, 3, 9, 10, 12, 7));

        dst = Converters.vector_KeyPoint_to_Mat(kps);

        Mat truth = new Mat(1, 1, CvType.CV_64FC(7));
        truth.put(0, 0, 2, 4, 3, 9, 10, 12, 7);

        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_Mat_to_Mat() {
        // List<Mat> mats = new ArrayList<Mat>();
        // mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        // mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        // mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        // mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        // mats.add(gray0);
        // mats.add(gray255);
        //
        // dst = Converters.vector_Mat_to_Mat(mats);
        fail("Not yet implemented");
    }

    public void testVector_Point_to_Mat() {
        List<Point> points = new ArrayList<Point>();
        points.add(new Point(2, 4));
        points.add(new Point(3, 9));
        points.add(new Point(10, 4));
        points.add(new Point(35, 54));

        dst = Converters.vector_Point_to_Mat(points);

        truth = new Mat(4, 1, CvType.CV_32SC2);
        truth.put(0, 0, 2, 4, 3, 9, 10, 4, 35, 54);
        assertMatEqual(truth, dst);
    }

    public void testVector_Point_to_MatListOfPoint() {
        fail("Not yet implemented");
    }

    public void testVector_Point_to_MatListOfPointInt() {
        fail("Not yet implemented");
    }

    public void testVector_Point2d_to_Mat() {
        List<Point> points = new ArrayList<Point>();
        points.add(new Point(12.0, 4.0));
        points.add(new Point(3.0, 9.0));
        points.add(new Point(1.0, 2.0));

        dst = Converters.vector_Point2d_to_Mat(points);

        truth = new Mat(3, 1, CvType.CV_64FC2);
        truth.put(0, 0, 12.0, 4.0, 3.0, 9.0, 1.0, 2.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_Point2f_to_Mat() {
        List<Point> points = new ArrayList<Point>();
        points.add(new Point(2.0, 3.0));
        points.add(new Point(1.0, 2.0));
        points.add(new Point(1.0, 4.0));

        dst = Converters.vector_Point2f_to_Mat(points);

        truth = new Mat(3, 1, CvType.CV_32FC2);
        truth.put(0, 0, 2.0, 3.0, 1.0, 2.0, 1.0, 4.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_Point3_to_Mat() {
        List<Point3> points = new ArrayList<Point3>();
        points.add(new Point3(2, 4, 3));
        points.add(new Point3(5, 9, 12));
        points.add(new Point3(10, 14, 15));
        points.add(new Point3(5, 11, 31));

        dst = Converters.vector_Point3_to_Mat(points, CvType.CV_32S);
        truth = new Mat(4, 1, CvType.CV_32SC3);
        truth.put(0, 0, 2.0, 4.0, 3.0, 5.0, 9.0, 12.0, 10.0, 14.0, 15.0, 5.0, 11.0, 31.0);

        assertMatEqual(truth, dst);
    }

    public void testVector_Point3d_to_Mat() {
        List<Point3> points = new ArrayList<Point3>();
        points.add(new Point3(2.0, 4.0, 3.0));
        points.add(new Point3(5.0, 9.0, 12.0));
        points.add(new Point3(10.0, 14.0, 15.0));
        points.add(new Point3(5.0, 11.0, 31.0));

        dst = Converters.vector_Point3d_to_Mat(points);
        truth = new Mat(4, 1, CvType.CV_64FC3);
        truth.put(0, 0, 2.0, 4.0, 3.0, 5.0, 9.0, 12.0, 10.0, 14.0, 15.0, 5.0, 11.0, 31.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_Point3f_to_Mat() {
        List<Point3> points = new ArrayList<Point3>();
        points.add(new Point3(2.0, 4.0, 3.0));
        points.add(new Point3(5.0, 9.0, 12.0));
        points.add(new Point3(10.0, 14.0, 15.0));
        points.add(new Point3(5.0, 11.0, 31.0));

        dst = Converters.vector_Point3f_to_Mat(points);
        truth = new Mat(4, 1, CvType.CV_32FC3);
        truth.put(0, 0, 2.0, 4.0, 3.0, 5.0, 9.0, 12.0, 10.0, 14.0, 15.0, 5.0, 11.0, 31.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testVector_Point3i_to_Mat() {
        List<Point3> points = new ArrayList<Point3>();
        points.add(new Point3(2, 4, 3));
        points.add(new Point3(5, 6, 2));
        points.add(new Point3(0, 4, 5));
        points.add(new Point3(5, 1, 3));

        dst = Converters.vector_Point3i_to_Mat(points);

        truth = new Mat(4, 1, CvType.CV_32SC3);
        truth.put(0, 0, 2, 4, 3, 5, 6, 2, 0, 4, 5, 5, 1, 3);
        assertMatEqual(truth, dst);
    }

    public void testVector_Rect_to_Mat() {
        List<Rect> rectangles = new ArrayList<Rect>();
        rectangles.add(new Rect(2, 2, 5, 2));
        rectangles.add(new Rect(0, 0, 6, 4));

        dst = Converters.vector_Rect_to_Mat(rectangles);

        truth = new Mat(2, 1, CvType.CV_32SC4);
        truth.put(0, 0, 2, 2, 5, 2, 0, 0, 6, 4);
        assertMatEqual(truth, dst);
    }

    public void testVector_uchar_to_Mat() {
        List<Byte> bytes = new ArrayList<Byte>();
        byte value1 = 1;
        byte value2 = 2;
        byte value3 = 3;
        byte value4 = 4;
        bytes.add(new Byte(value1));
        bytes.add(new Byte(value2));
        bytes.add(new Byte(value3));
        bytes.add(new Byte(value4));

        dst = Converters.vector_uchar_to_Mat(bytes);
        truth = new Mat(4, 1, CvType.CV_8UC1);
        truth.put(0, 0, 1, 2, 3, 4);
        assertMatEqual(truth, dst);
    }

    public void testVector_vector_char_to_Mat() {
        fail("Not yet implemented");
    }

    public void testVector_vector_DMatch_to_Mat() {
        fail("Not yet implemented");
    }

    public void testVector_vector_KeyPoint_to_Mat() {
        fail("Not yet implemented");
    }

    public void testVector_vector_Point_to_Mat() {
        fail("Not yet implemented");

    }

}
