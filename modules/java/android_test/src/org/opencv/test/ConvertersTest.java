package org.opencv.test;

import java.util.ArrayList;
import java.util.List;

import org.opencv.utils.Converters;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.features2d.KeyPoint;

public class ConvertersTest extends OpenCVTestCase {

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
        assertListFloatEquals(truth, fs, EPS);
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
        assertListIntegerEquals(truth, fs);
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
        //Mat src = new Mat(4, 1, CvType.CV_32SC2);
        //src.put(0, 0, 2, 2, 3, 3, 4, 4, 5, 5);
        //
        //List<Mat> mats = new ArrayList<Mat>();
        //Converters.Mat_to_vector_Mat(src, mats);
        //
        //List<Mat> truth = new ArrayList<Mat>();
        //truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        //truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(3.0)));
        //truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(4.0)));
        //truth.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(5.0)));
        //assertListEqualMat(truth, mats, EPS);
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

    public void testVector_Mat_to_Mat() {
        //List<Mat> mats = new ArrayList<Mat>();
        //mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        //mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        //mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        //mats.add(new Mat(2, 1, CvType.CV_32SC1, Scalar.all(2.0)));
        //mats.add(gray0);
        //mats.add(gray255);
        //
        //dst = Converters.vector_Mat_to_Mat(mats);
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

}
