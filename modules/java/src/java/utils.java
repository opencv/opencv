package org.opencv;

import java.util.List;


public class utils {
	
	public static Mat vector_Point_to_Mat(List<Point> pts) {
		Mat res;
		int count = (pts!=null) ? pts.size() : 0;
		if(count>0){
			res = new Mat(1, count, CvType.CV_64FC2); //Point can be saved into double[2]
			double[] buff = new double[count*2];
			for(int i=0; i<count; i++) {
				Point p = pts.get(i);
				buff[i*2]   = p.x;
				buff[i*2+1] = p.y;
			}
			res.put(0, 0, buff);
		} else {
			res = new Mat();
		}
		return res;
	}

	public static void Mat_to_vector_Point(Mat m, List<Point> pts) {
		if(pts == null)
			return;
		int cols = m.cols();
		if(!CvType.CV_64FC2.equals(m.type()) ||  m.rows()!=1 || cols%2!=0)
			return;
		
		pts.clear();
		double[] buff = new double[cols];
		m.get(0, 0, buff);
		for(int i=0; i<cols/2; i++) {
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
			return;
		int cols = m.cols();
		if(!CvType.CV_32SC2.equals(m.type()) ||  m.rows()!=1 || cols%2!=0)
			return;
		
		mats.clear();
		int[] buff = new int[cols];
		m.get(0, 0, buff);
		for(int i=0; i<cols/2; i++) {
			long addr = (((long)buff[i*2])<<32) | ((long)buff[i*2+1]);
			mats.add( new Mat(addr) );
		}
	}

	public static void Mat_to_vector_KeyPoint(Mat kp_mat, List<features2d.KeyPoint> kps) {
		// TODO Auto-generated method stub
	}

	public static Mat vector_float_to_Mat(List<Float> fs) {
		// TODO Auto-generated method stub
		return null;
	}

	public static void Mat_to_vector_float(Mat m, List<Float> fs) {
		// TODO Auto-generated method stub
	}

	public static Mat vector_uchar_to_Mat(List<Byte> bs) {
		// TODO Auto-generated method stub
		return null;
	}

	public static Mat vector_int_to_Mat(List<Integer> is) {
		// TODO Auto-generated method stub
		return null;
	}

	public static void Mat_to_vector_int(Mat m, List<Integer> is) {
		// TODO Auto-generated method stub
		
	}

	public static Mat vector_Rect_to_Mat(List<Rect> rs) {
		// TODO Auto-generated method stub
		return null;
	}

	public static void Mat_to_vector_Rect(Mat m, List<Rect> rs) {
		// TODO Auto-generated method stub
		
	}

	public static Mat vector_double_to_Mat(List<Double> ds) {
		// TODO Auto-generated method stub
		return null;
	}

}
