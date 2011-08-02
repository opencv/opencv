package org.opencv;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;

public class Converters {

    public static Mat vector_Point_to_Mat(List<Point> pts) {
        return vector_Point_to_Mat(pts, CvType.CV_32S);
    }

    public static Mat vector_Point2f_to_Mat(List<Point> pts) {
        return vector_Point_to_Mat(pts, CvType.CV_32F);
    }

    public static Mat vector_Point2d_to_Mat(List<Point> pts) {
        return vector_Point_to_Mat(pts, CvType.CV_64F);
    }

    public static Mat vector_Point_to_Mat(List<Point> pts, int typeDepth) {
        Mat res;
        int count = (pts!=null) ? pts.size() : 0;
        if(count>0){
            switch (typeDepth) {
                case CvType.CV_32S:
                {
                    res = new Mat(count, 1, CvType.CV_32SC2);
                    int[] buff = new int[count*2];
                    for(int i=0; i<count; i++) {
                        Point p = pts.get(i);
                        buff[i*2]   = (int)p.x;
                        buff[i*2+1] = (int)p.y;
                    }
                    res.put(0, 0, buff);
                }
                break;

                case CvType.CV_32F:
                {
                    res = new Mat(count, 1, CvType.CV_32FC2);
                    float[] buff = new float[count*2];
                    for(int i=0; i<count; i++) {
                        Point p = pts.get(i);
                        buff[i*2]   = (float)p.x;
                        buff[i*2+1] = (float)p.y;
                    }
                    res.put(0, 0, buff);
                }
                break;

                case CvType.CV_64F:
                {
                    res = new Mat(count, 1, CvType.CV_64FC2);
                    double[] buff = new double[count*2];
                    for(int i=0; i<count; i++) {
                        Point p = pts.get(i);
                        buff[i*2]   = p.x;
                        buff[i*2+1] = p.y;
                    }
                    res.put(0, 0, buff);
                }
                break;

                default:
                    throw new IllegalArgumentException("'typeDepth' can be CV_32S, CV_32F or CV_64F");
            }
        } else {
            res = new Mat();
        }
        return res;
    }


    public static Mat vector_Point3i_to_Mat(List<Point3> pts) {
        return vector_Point3_to_Mat(pts, CvType.CV_32S);
    }

    public static Mat vector_Point3f_to_Mat(List<Point3> pts) {
        return vector_Point3_to_Mat(pts, CvType.CV_32F);
    }

    public static Mat vector_Point3d_to_Mat(List<Point3> pts) {
        return vector_Point3_to_Mat(pts, CvType.CV_64F);
    }

    public static Mat vector_Point3_to_Mat(List<Point3> pts, int typeDepth) {
        Mat res;
        int count = (pts!=null) ? pts.size() : 0;
        if(count>0){
            switch (typeDepth){
                case CvType.CV_32S:
                {
                    res = new Mat(count, 1, CvType.CV_32SC3);
                    int[] buff = new int[count*3];
                    for(int i=0; i<count; i++) {
                        Point3 p = pts.get(i);
                        buff[i*3]   = (int)p.x;
                        buff[i*3+1] = (int)p.y;
                        buff[i*3+2] = (int)p.z;
                    }
                    res.put(0, 0, buff);
                }
                break;

                case CvType.CV_32F:
                {
                    res = new Mat(count, 1, CvType.CV_64FC3);
                    float[] buff = new float[count*3];
                    for(int i=0; i<count; i++) {
                        Point3 p = pts.get(i);
                        buff[i*3]   = (float)p.x;
                        buff[i*3+1] = (float)p.y;
                        buff[i*3+2] = (float)p.z;
                    }
                    res.put(0, 0, buff);
                }
                break;

                case CvType.CV_64F:
                {
                    res = new Mat(count, 1, CvType.CV_64FC3);
                    double[] buff = new double[count*3];
                    for(int i=0; i<count; i++) {
                        Point3 p = pts.get(i);
                        buff[i*3]   = p.x;
                        buff[i*3+1] = p.y;
                        buff[i*3+2] = p.z;
                    }
                    res.put(0, 0, buff);
                }
                break;

                default:
                    throw new IllegalArgumentException("'typeDepth' can be CV_32S, CV_32F or CV_64F");
            }
        } else {
            res = new Mat();
        }
        return res;
    }

    public static void Mat_to_vector_Point2f(Mat m, List<Point> pts) {
        Mat_to_vector_Point(m, pts);
    }

    public static void Mat_to_vector_Point2d(Mat m, List<Point> pts) {
        Mat_to_vector_Point(m, pts);
    }
    public static void Mat_to_vector_Point(Mat m, List<Point> pts) {
        if(pts == null)
            throw new java.lang.IllegalArgumentException("Output List can't be null");
        int count = m.rows();
        int type = m.type();
        if(m.cols() != 1)
            throw new java.lang.IllegalArgumentException( "Input Mat should have one column\n" + m );

        pts.clear();
        if(type == CvType.CV_32SC2) {
            int[] buff = new int[2*count];
            m.get(0, 0, buff);
            for(int i=0; i<count; i++) {
                pts.add( new Point(buff[i*2], buff[i*2+1]) );
            }
        } else if(type == CvType.CV_32FC2){
            float[] buff = new float[2*count];
            m.get(0, 0, buff);
            for(int i=0; i<count; i++) {
                pts.add( new Point(buff[i*2], buff[i*2+1]) );
            }
        } else if(type == CvType.CV_64FC2){
            double[] buff = new double[2*count];
            m.get(0, 0, buff);
            for(int i=0; i<count; i++) {
                pts.add( new Point(buff[i*2], buff[i*2+1]) );
            }
        } else {
            throw new java.lang.IllegalArgumentException(
                    "Input Mat should be of CV_32SC2, CV_32FC2 or CV_64FC2 type\n" + m );
        }
    }

    public static void Mat_to_vector_Point3i(Mat m, List<Point3> pts) {
        Mat_to_vector_Point3(m, pts);
    }
    public static void Mat_to_vector_Point3f(Mat m, List<Point3> pts) {
        Mat_to_vector_Point3(m, pts);
    }

    public static void Mat_to_vector_Point3d(Mat m, List<Point3> pts) {
        Mat_to_vector_Point3(m, pts);
    }
    public static void Mat_to_vector_Point3(Mat m, List<Point3> pts) {
        if(pts == null)
            throw new java.lang.IllegalArgumentException("Output List can't be null");
        int count = m.rows();
        int type = m.type();
        if(m.cols() != 1)
            throw new java.lang.IllegalArgumentException( "Input Mat should have one column\n" + m );

        pts.clear();
        if(type == CvType.CV_32SC3) {
            int[] buff = new int[3*count];
            m.get(0, 0, buff);
            for(int i=0; i<count; i++) {
                pts.add( new Point3(buff[i*3], buff[i*3+1], buff[i*3+2]) );
            }
        } else if(type == CvType.CV_32FC3){
            float[] buff = new float[3*count];
            m.get(0, 0, buff);
            for(int i=0; i<count; i++) {
                pts.add( new Point3(buff[i*3], buff[i*3+1], buff[i*3+2]) );
            }
        } else if(type == CvType.CV_64FC3){
            double[] buff = new double[3*count];
            m.get(0, 0, buff);
            for(int i=0; i<count; i++) {
                pts.add( new Point3(buff[i*3], buff[i*3+1], buff[i*3+2]) );
            }
        } else {
            throw new java.lang.IllegalArgumentException(
                    "Input Mat should be of CV_32SC3, CV_32FC3 or CV_64FC3 type\n" + m );
        }
    }

    public static Mat vector_Mat_to_Mat(List<Mat> mats) {
        Mat res;
        int count = (mats!=null) ? mats.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_32SC2);
            int[] buff = new int[count*2];
            for(int i=0; i<count; i++) {
                long addr = mats.get(i).nativeObj;
                buff[i*2]   = (int)(addr >> 32);
                buff[i*2+1] = (int)(addr & 0xffffffff);
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

    public static void Mat_to_vector_Mat(Mat m, List<Mat> mats) {
        if(mats == null)
            throw new java.lang.IllegalArgumentException("mats == null");
        int count = m.rows();
        if( CvType.CV_32SC2 != m.type() ||  m.cols()!=1 )
            throw new java.lang.IllegalArgumentException(
                    "CvType.CV_32SC2 != m.type() ||  m.cols()!=1\n" + m);

        mats.clear();
        int[] buff = new int[count*2];
        m.get(0, 0, buff);
        for(int i=0; i<count; i++) {
            long addr = (((long)buff[i*2])<<32) | ((long)buff[i*2+1]);
            mats.add( new Mat(addr) );
        }
    }

    public static Mat vector_float_to_Mat(List<Float> fs) {
        Mat res;
        int count = (fs!=null) ? fs.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_32FC1);
            float[] buff = new float[count];
            for(int i=0; i<count; i++) {
                float f = fs.get(i);
                buff[i]   = f;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

    public static void Mat_to_vector_float(Mat m, List<Float> fs) {
        if(fs == null)
            throw new java.lang.IllegalArgumentException("fs == null");
        int count = m.rows();
        if( CvType.CV_32FC1 != m.type() ||  m.cols()!=1 )
            throw new java.lang.IllegalArgumentException(
                    "CvType.CV_32FC1 != m.type() ||  m.cols()!=1\n" + m);

        fs.clear();
        float[] buff = new float[count];
        m.get(0, 0, buff);
        for(int i=0; i<count; i++) {
            fs.add( new Float(buff[i]) );
        }
    }

    public static Mat vector_uchar_to_Mat(List<Byte> bs) {
        Mat res;
        int count = (bs!=null) ? bs.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_8UC1);
            byte[] buff = new byte[count];
            for(int i=0; i<count; i++) {
                byte b = bs.get(i);
                buff[i]   = b;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

    public static Mat vector_char_to_Mat(List<Byte> bs) {
        Mat res;
        int count = (bs!=null) ? bs.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_8SC1);
            byte[] buff = new byte[count];
            for(int i=0; i<count; i++) {
                byte b = bs.get(i);
                buff[i]   = b;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

    public static Mat vector_int_to_Mat(List<Integer> is) {
        Mat res;
        int count = (is!=null) ? is.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_32SC1);
            int[] buff = new int[count];
            for(int i=0; i<count; i++) {
                int v = is.get(i);
                buff[i]   = v;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

    public static void Mat_to_vector_int(Mat m, List<Integer> is) {
        if(is == null)
            throw new java.lang.IllegalArgumentException("is == null");
        int count = m.rows();
        if( CvType.CV_32SC1 != m.type() ||  m.cols()!=1 )
            throw new java.lang.IllegalArgumentException(
                    "CvType.CV_32SC1 != m.type() ||  m.cols()!=1\n" + m);

        is.clear();
        int[] buff = new int[count];
        m.get(0, 0, buff);
        for(int i=0; i<count; i++) {
            is.add( new Integer(buff[i]) );
        }
    }

    public static void Mat_to_vector_char(Mat m, List<Byte> bs) {
        if(bs == null)
            throw new java.lang.IllegalArgumentException("Output List can't be null");
        int count = m.rows();
        if( CvType.CV_8SC1 != m.type() ||  m.cols()!=1 )
            throw new java.lang.IllegalArgumentException(
                    "CvType.CV_8SC1 != m.type() ||  m.cols()!=1\n" + m);

        bs.clear();
        byte[] buff = new byte[count];
        m.get(0, 0, buff);
        for(int i=0; i<count; i++) {
            bs.add( new Byte(buff[i]) );
        }
    }

    public static Mat vector_Rect_to_Mat(List<Rect> rs) {
        Mat res;
        int count = (rs!=null) ? rs.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_32SC4);
            int[] buff = new int[4*count];
            for(int i=0; i<count; i++) {
                Rect r = rs.get(i);
                buff[4*i  ]   = r.x;
                buff[4*i+1]   = r.y;
                buff[4*i+2]   = r.width;
                buff[4*i+3]   = r.height;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

    public static void Mat_to_vector_Rect(Mat m, List<Rect> rs) {
        if(rs == null)
            throw new java.lang.IllegalArgumentException("rs == null");
        int count = m.rows();
        if(CvType.CV_32SC4 != m.type() ||  m.cols()!=1 )
            throw new java.lang.IllegalArgumentException(
                    "CvType.CV_32SC4 != m.type() ||  m.rows()!=1\n" + m);

        rs.clear();
        int[] buff = new int[4*count];
        m.get(0, 0, buff);
        for(int i=0; i<count; i++) {
            rs.add( new Rect(buff[4*i], buff[4*i+1], buff[4*i+2], buff[4*i+3]) );
        }
    }


    public static Mat vector_KeyPoint_to_Mat(List<KeyPoint> kps) {
        Mat res;
        int count = (kps!=null) ? kps.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_64FC(7));
            double[] buff = new double[count * 7];
            for(int i=0; i<count; i++) {
                KeyPoint kp = kps.get(i);
                buff[7*i  ]   = kp.pt.x;
                buff[7*i+1]   = kp.pt.y;
                buff[7*i+2]   = kp.size;
                buff[7*i+3]   = kp.angle;
                buff[7*i+4]   = kp.response;
                buff[7*i+5]   = kp.octave;
                buff[7*i+6]   = kp.class_id;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

    public static void Mat_to_vector_KeyPoint(Mat m, List<KeyPoint> kps) {
        if(kps == null)
            throw new java.lang.IllegalArgumentException("Output List can't be null");
        int count = m.rows();
        if( CvType.CV_64FC(7) != m.type() ||  m.cols()!=1 )
            throw new java.lang.IllegalArgumentException(
                    "CvType.CV_64FC(7) != m.type() ||  m.cols()!=1\n" + m);

        kps.clear();
        double[] buff = new double[7*count];
        m.get(0, 0, buff);
        for(int i=0; i<count; i++) {
            kps.add( new KeyPoint( (float)buff[7*i], (float)buff[7*i+1], (float)buff[7*i+2], (float)buff[7*i+3],
                                   (float)buff[7*i+4], (int)buff[7*i+5], (int)buff[7*i+6] ) );
        }
    }


    public static Mat vector_double_to_Mat(List<Double> ds) {
        Mat res;
        int count = (ds!=null) ? ds.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_64FC1);
            double[] buff = new double[count];
            for(int i=0; i<count; i++) {
                double v = ds.get(i);
                buff[i]   = v;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
    }

	public static Mat vector_DMatch_to_Mat(List<DMatch> matches) {
        Mat res;
        int count = (matches!=null) ? matches.size() : 0;
        if(count>0){
            res = new Mat(count, 1, CvType.CV_64FC4);
            double[] buff = new double[count * 4];
            for(int i=0; i<count; i++) {
                DMatch m = matches.get(i);
                buff[4*i  ]   = m.queryIdx;
                buff[4*i+1]   = m.trainIdx;
                buff[4*i+2]   = m.imgIdx;
                buff[4*i+3]   = m.distance;
            }
            res.put(0, 0, buff);
        } else {
            res = new Mat();
        }
        return res;
	}
	
    public static void Mat_to_vector_DMatch(Mat m, List<DMatch> matches) {
        if(matches == null)
            throw new java.lang.IllegalArgumentException("Output List can't be null");
        int count = m.rows();
        if( CvType.CV_64FC4 != m.type() ||  m.cols()!=1 )
            throw new java.lang.IllegalArgumentException(
                    "CvType.CV_64FC4 != m.type() ||  m.cols()!=1\n" + m);

        matches.clear();
        double[] buff = new double[4*count];
        m.get(0, 0, buff);
        for(int i=0; i<count; i++) {
        	matches.add( new DMatch( (int)buff[4*i], (int)buff[4*i+1], (int)buff[4*i+2], (float)buff[4*i+3] ) );
        }
    }

}
