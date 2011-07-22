package org.opencv;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.features2d.KeyPoint;

public class Converters {
	
	public static Mat vector_Point_to_Mat(List<Point> pts) {
		Mat res;
		int count = (pts!=null) ? pts.size() : 0;
		if(count>0){
			res = new Mat(1, count, CvType.CV_32SC2);
			int[] buff = new int[count*2];
			for(int i=0; i<count; i++) {
				Point p = pts.get(i);
				buff[i*2]   = (int)p.x;
				buff[i*2+1] = (int)p.y;
			}
			res.put(0, 0, buff);
		} else {
			res = new Mat();
		}
		return res;
	}

	public static void Mat_to_vector_Point(Mat m, List<Point> pts) {
		if(pts == null)
			throw new java.lang.IllegalArgumentException();
		int cols = m.cols();
		if(CvType.CV_32SC2 != m.type() ||  m.rows()!=1 )
			throw new java.lang.IllegalArgumentException();
		
		pts.clear();
		int[] buff = new int[2*cols];
		m.get(0, 0, buff);
		for(int i=0; i<cols; i++) {
			pts.add( new Point(buff[i*2], buff[i*2+1]) );
		}
	}

	public static Mat vector_Mat_to_Mat(List<Mat> mats) {
		Mat res;
		int count = (mats!=null) ? mats.size() : 0;
		if(count>0){
			res = new Mat(1, count, CvType.CV_32SC2);
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
			throw new java.lang.IllegalArgumentException();
		int cols = m.cols();
		if(CvType.CV_32SC2 != m.type() ||  m.rows()!=1 )
			throw new java.lang.IllegalArgumentException();
		
		mats.clear();
		int[] buff = new int[cols*2];
		m.get(0, 0, buff);
		for(int i=0; i<cols; i++) {
			long addr = (((long)buff[i*2])<<32) | ((long)buff[i*2+1]);
			mats.add( new Mat(addr) );
		}
	}

	public static void Mat_to_vector_KeyPoint(Mat kp_mat, List<KeyPoint> kps) {
		// TODO Auto-generated method stub
	}

	public static Mat vector_float_to_Mat(List<Float> fs) {
		Mat res;
		int count = (fs!=null) ? fs.size() : 0;
		if(count>0){
			res = new Mat(1, count, CvType.CV_32FC1); //Point can be saved into double[2]
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
			throw new java.lang.IllegalArgumentException();
		int cols = m.cols();
		if(CvType.CV_32FC1 != m.type() ||  m.rows()!=1 )
			throw new java.lang.IllegalArgumentException();
		
		fs.clear();
		float[] buff = new float[cols];
		m.get(0, 0, buff);
		for(int i=0; i<cols; i++) {
			fs.add( new Float(buff[i]) );
		}
	}

	public static Mat vector_uchar_to_Mat(List<Byte> bs) {
		Mat res;
		int count = (bs!=null) ? bs.size() : 0;
		if(count>0){
			res = new Mat(1, count, CvType.CV_8UC1); //Point can be saved into double[2]
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
			res = new Mat(1, count, CvType.CV_32SC1); //Point can be saved into double[2]
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
			throw new java.lang.IllegalArgumentException();
		int cols = m.cols();
		if(CvType.CV_32SC1 != m.type() ||  m.rows()!=1 )
			throw new java.lang.IllegalArgumentException();
		
		is.clear();
		int[] buff = new int[cols];
		m.get(0, 0, buff);
		for(int i=0; i<cols; i++) {
			is.add( new Integer(buff[i]) );
		}
	}

	public static Mat vector_Rect_to_Mat(List<Rect> rs) {
		Mat res;
		int count = (rs!=null) ? rs.size() : 0;
		if(count>0){
			res = new Mat(1, count, CvType.CV_32SC4); //Point can be saved into double[2]
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
			throw new java.lang.IllegalArgumentException();
		int cols = m.cols();
		if(CvType.CV_32SC4 != m.type() ||  m.rows()!=1 )
			throw new java.lang.IllegalArgumentException();
		
		rs.clear();
		int[] buff = new int[4*cols];
		m.get(0, 0, buff);
		for(int i=0; i<cols; i++) {
			rs.add( new Rect(buff[4*i], buff[4*i+1], buff[4*i+2], buff[4*i+3]) );
		}
	}

	public static Mat vector_double_to_Mat(List<Double> ds) {
		Mat res;
		int count = (ds!=null) ? ds.size() : 0;
		if(count>0){
			res = new Mat(1, count, CvType.CV_64FC1); //Point can be saved into double[2]
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

}
