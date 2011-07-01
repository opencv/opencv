package org.opencv;

public class Mat {

	public static class CvType {
		
		// predefined type constants
		public static final CvType
		CV_8UC1  = CV_8UC(1),  CV_8UC2  = CV_8UC(2),  CV_8UC3  = CV_8UC(3),  CV_8UC4  = CV_8UC(4),
		CV_8SC1  = CV_8SC(1),  CV_8SC2  = CV_8SC(2),  CV_8SC3  = CV_8SC(3),  CV_8SC4  = CV_8SC(4),
		CV_16UC1 = CV_16UC(1), CV_16UC2 = CV_16UC(2), CV_16UC3 = CV_16UC(3), CV_16UC4 = CV_16UC(4),
		CV_16SC1 = CV_16SC(1), CV_16SC2 = CV_16SC(2), CV_16SC3 = CV_16SC(3), CV_16SC4 = CV_16SC(4),
		CV_32SC1 = CV_32SC(1), CV_32SC2 = CV_32SC(2), CV_32SC3 = CV_32SC(3), CV_32SC4 = CV_32SC(4),
		CV_32FC1 = CV_32FC(1), CV_32FC2 = CV_32FC(2), CV_32FC3 = CV_32FC(3), CV_32FC4 = CV_32FC(4),
		CV_64FC1 = CV_64FC(1), CV_64FC2 = CV_64FC(2), CV_64FC3 = CV_64FC(3), CV_64FC4 = CV_64FC(4);
		
		// type depth constants
		public static final int CV_8U  = 0, 
								CV_8S  = 1, 
								CV_16U = 2, 
								CV_16S = 3, 
								CV_32S = 4, 
								CV_32F = 5, 
								CV_64F = 6, 
								CV_USRTYPE1=7;
		
		private static final int CV_CN_MAX = 512, CV_CN_SHIFT = 3, CV_DEPTH_MAX = (1 << CV_CN_SHIFT);

		private final int value;
		
		protected CvType(int depth, int channels) {
			if(channels<=0 || channels>=CV_CN_MAX) {
				throw new java.lang.UnsupportedOperationException(
						"Channels count should be 1.." + (CV_CN_MAX-1) );
			}
			if(depth<0 || depth>=CV_DEPTH_MAX) {
				throw new java.lang.UnsupportedOperationException(
						"Data type depth should be 0.." + (CV_DEPTH_MAX-1) );
			}
			value  = (depth & (CV_DEPTH_MAX-1)) + ((channels-1) << CV_CN_SHIFT); 
		}

		protected CvType(int val) { value = val; }

		public static final CvType CV_8UC(int ch)  { return new CvType(CV_8U, ch); }
		
		public static final CvType CV_8SC(int ch)  { return new CvType(CV_8S, ch); }
		
		public static final CvType CV_16UC(int ch) { return new CvType(CV_16U, ch); }
		
		public static final CvType CV_16SC(int ch) { return new CvType(CV_16S, ch); }
		
		public static final CvType CV_32SC(int ch) { return new CvType(CV_32S, ch); }
		
		public static final CvType CV_32FC(int ch) { return new CvType(CV_32F, ch); }
		
		public static final CvType CV_64FC(int ch) { return new CvType(CV_64F, ch); }
		
		public final int toInt() { return value; }
		
		public final int channels() { return (value >> CV_CN_SHIFT) + 1; }
		
		public final int depth() { return value & (CV_DEPTH_MAX-1); }
		
		public final boolean isInteger() { return depth() < CV_32F; }

		public final int CV_ELEM_SIZE() { 
			int depth = value & (CV_DEPTH_MAX-1);
			switch (depth) {
				case CV_8U:
				case CV_8S:
					return channels();
				case CV_16U:
				case CV_16S:
					return 2 * channels();
				case CV_32S:
				case CV_32F:
					return 4 * channels();
				case CV_64F:
					return 8 * channels();
				default:
					throw new java.lang.UnsupportedOperationException(
							"Unsupported CvType value: " + value );
			}
		}

		@Override
		public final String toString() {
			String s;
			switch (depth()) {
				case CV_8U:
					s = "CV_8U";
					break;
				case CV_8S:
					s = "CV_8S";
					break;
				case CV_16U:
					s = "CV_16U";
					break;
				case CV_16S:
					s = "CV_16S";
					break;
				case CV_32S:
					s = "CV_32S";
					break;
				case CV_32F:
					s = "CV_32F";
					break;
				case CV_64F:
					s = "CV_64F";
					break;
				default:
					s = "CV_USRTYPE1";
			}
			
			return s + "(" + channels() + ")";
		}
		
		// hashCode() has to be overridden if equals() is
		@Override
		public final int hashCode() { return value; }

		@Override
		public final boolean equals(Object obj) {
			if (this == obj) return true;
			if ( !(obj instanceof CvType) ) return false;
			CvType other = (CvType) obj;
			return value == other.value;
		}
	}

	protected Mat(long nativeMat) {
		/*if(nativeMat == 0) 
			throw new java.lang.UnsupportedOperationException("Native object address is NULL");*/
		this.nativeObj = nativeMat;
	}
	
	public Mat(int rows, int cols, CvType type) {
		this( nCreateMat(rows, cols, type.toInt()) );
	}

	public Mat(int rows, int cols, CvType type, double v0, double v1, double v2, double v3) {
		this( nCreateMat(rows, cols, type.toInt(), v0, v1, v2, v3) );
	}

	public Mat(int rows, int cols, CvType type, double v0, double v1, double v2) {
		this( nCreateMat(rows, cols, type.toInt(), v0, v1, v2, 0) );
	}

	public Mat(int rows, int cols, CvType type, double v0, double v1) {
		this( nCreateMat(rows, cols, type.toInt(), v0, v1, 0, 0) );
	}

	public Mat(int rows, int cols, CvType type, double v0) {
		this( nCreateMat(rows, cols, type.toInt(), v0, 0, 0, 0) );
	}

	public void dispose() {
		if(nativeObj != 0)
			nDispose(nativeObj);
		nativeObj = 0;
	}
	
	@Override
	protected void finalize() throws Throwable {
		dispose();
		super.finalize();
	}

	@Override
	public String toString() {
		if(nativeObj == 0) return  "Mat [ nativeObj=NULL ]";
		return  "Mat [ " +
				rows() + "*" + cols() + "*" + type() + 
				", isCont=" + isContinuous() + ", isSubmat=" + isSubmatrix() +
				", nativeObj=0x" + Long.toHexString(nativeObj) + 
				", dataAddr=0x" + Long.toHexString(dataAddr()) + 
				" ]";
	}
	
	public boolean empty() {
		if(nativeObj == 0) return true;
		return nIsEmpty(nativeObj); 
}

	private void checkNull() {
		if(nativeObj == 0) 
			throw new java.lang.UnsupportedOperationException("Native object address is NULL");
	}
	
	public CvType type() {
		checkNull();
		return new CvType( nType(nativeObj) );
	}
	public int depth() { return type().depth(); }
	public int channels() { return type().channels(); }
	public int elemSize() { return type().CV_ELEM_SIZE(); }

	public int rows() {
		if(nativeObj == 0) 
			return 0;
		return nRows(nativeObj);
	}
	public int height() { return rows(); }
	public int cols() {
		if(nativeObj == 0) 
			return 0;
		return nCols(nativeObj);
	}
	public int width() { return cols(); }
	public int total() { return rows() * cols(); }

	public long dataAddr() {
		if(nativeObj == 0) 
			return 0;
		return nData(nativeObj);
	}

	public boolean isContinuous() {
		if(nativeObj == 0) 
			return false; // maybe throw an exception instead?
		return nIsCont(nativeObj);
	}

	public boolean isSubmatrix() {
		if(nativeObj == 0) 
			return false; // maybe throw an exception instead?
		return nIsSubmat(nativeObj);
	}

	public Mat submat(int rowStart, int rowEnd, int colStart, int colEnd) {
		checkNull();
		return new Mat( nSubmat(nativeObj, rowStart, rowEnd, colStart, colEnd) );
	}
	public Mat rowRange(int start, int end) { return submat(start, end, 0, -1); }
	public Mat row(int num) { return submat(num, num+1, 0, -1); }
	public Mat colRange(int start, int end) { return submat(0, -1, start, end); }
	public Mat col(int num) { return submat(0, -1, num, num+1); }
	
	public Mat clone() {
		checkNull();
		return new Mat( nClone(nativeObj) );
	}
	
	public int put(int row, int col, double...data) {
		checkNull();
		if(data != null)
			return nPutD(nativeObj, row, col, data.length, data);
		else
			return 0;
	}
	
	public int put(int row, int col, float[] data) {
		checkNull();
		if(data != null) {
			CvType t = type(); 
			if(t.depth() == CvType.CV_32F) {
				return nPutF(nativeObj, row, col, data.length, data);
			}
			throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
		} else return 0;
	}
	
	public int put(int row, int col, int[] data) {
		checkNull();
		if(data != null) {
			CvType t = type(); 
			if(t.depth() == CvType.CV_32S) {
				return nPutI(nativeObj, row, col, data.length, data);
			}
			throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
		} else return 0;
	}
	
	public int put(int row, int col, short[] data) {
		checkNull();
		if(data != null) {
			CvType t = type(); 
			if(t.depth() == CvType.CV_16U || t.depth() == CvType.CV_16S) {
				return nPutS(nativeObj, row, col, data.length, data);
			}
			throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
		} else return 0;
	}
	
	public int put(int row, int col, byte[] data) {
		checkNull();
		if(data != null) {
			CvType t = type(); 
			if(t.depth() == CvType.CV_8U || t.depth() == CvType.CV_8S) {
				return nPutB(nativeObj, row, col, data.length, data);
			}
			throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
		} else return 0;
	}
	
	public int get(int row, int col, byte[] data) {
		checkNull();
		CvType t = type(); 
		if(t.depth() == CvType.CV_8U || t.depth() == CvType.CV_8S) {
			return nGetB(nativeObj, row, col, data.length, data);
		}
		throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
	}

	public int get(int row, int col, short[] data) {
		checkNull();
		CvType t = type(); 
		if(t.depth() == CvType.CV_16U || t.depth() == CvType.CV_16S) {
			return nGetS(nativeObj, row, col, data.length, data);
		}
		throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
	}

	public int get(int row, int col, int[] data) {
		checkNull();
		CvType t = type(); 
		if(t.depth() == CvType.CV_32S) {
			return nGetI(nativeObj, row, col, data.length, data);
		}
		throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
	}

	public int get(int row, int col, float[] data) {
		checkNull();
		CvType t = type(); 
		if(t.depth() == CvType.CV_32F) {
			return nGetF(nativeObj, row, col, data.length, data);
		}
		throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
	}

	public int get(int row, int col, double[] data) {
		checkNull();
		CvType t = type(); 
		if(t.depth() == CvType.CV_64F) {
			return nGetD(nativeObj, row, col, data.length, data);
		}
		throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
	}

	public double[] get(int row, int col) {
		checkNull();
		//CvType t = type();
		//if(t.depth() == CvType.CV_64F) {
			return nGet(nativeObj, row, col);
		//}
		//throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
	}


	public void setTo(double v0, double v1, double v2, double v3) {
		checkNull();
		nSetTo(nativeObj, v0, v1, v2, v3);
	}
	public void setTo(double v0, double v1, double v2) { setTo(v0, v1, v2, 0); }
	public void setTo(double v0, double v1) { setTo(v0, v1, 0, 0); }
	public void setTo(double v0) { setTo(v0, 0, 0, 0); }
	
	public void copyTo(Mat m) {
		checkNull();
		if(m.nativeObj == 0) 
			throw new java.lang.UnsupportedOperationException("Destination native object address is NULL");
		nCopyTo(nativeObj, m.nativeObj);
	}
	
	public double dot(Mat m) {
		checkNull();
		return nDot(nativeObj, m.nativeObj);
	}
	
	public Mat cross(Mat m) {
		checkNull();
		return new Mat( nCross(nativeObj, m.nativeObj) );
	}

	public Mat inv() {
		checkNull();
		return new Mat( nInv(nativeObj) );
	}
	
	public long getNativeObjAddr() {
		return nativeObj;
	}
	
    static public Mat eye(int rows, int cols, CvType type) {
        return new Mat( nEye(rows, cols, type.toInt()) );
    }
	
	// native stuff
	static { System.loadLibrary("opencv_java"); }
	protected long nativeObj;
	private static native long nCreateMat(int rows, int cols, int type);
	private static native long nCreateMat(int rows, int cols, int type, double v0, double v1, double v2, double v3);
	private static native void nDispose(long self);
	private static native int nType(long self);
	private static native int nRows(long self);
	private static native int nCols(long self);
	private static native long nData(long self);
	private static native boolean nIsEmpty(long self);
	private static native boolean nIsCont(long self);
	private static native boolean nIsSubmat(long self);
	private static native long nSubmat(long self, int rowStart, int rowEnd, int colStart, int colEnd);
	private static native long nClone(long self);
	private static native int nPutD(long self, int row, int col, int count, double[] data);
	private static native int nPutF(long self, int row, int col, int count, float[] data);
	private static native int nPutI(long self, int row, int col, int count, int[] data);
	private static native int nPutS(long self, int row, int col, int count, short[] data);
	private static native int nPutB(long self, int row, int col, int count, byte[] data);
	private static native int nGetB(long self, int row, int col, int count, byte[] vals);
	private static native int nGetS(long self, int row, int col, int count, short[] vals);
	private static native int nGetI(long self, int row, int col, int count, int[] vals);
	private static native int nGetF(long self, int row, int col, int count, float[] vals);
	private static native int nGetD(long self, int row, int col, int count, double[] vals);
	private static native double[] nGet(long self, int row, int col);
	private static native void nSetTo(long self, double v0, double v1, double v2, double v3);
	private static native void nCopyTo(long self, long mat);
	private static native double nDot(long self, long mat);
	private static native long nCross(long self, long mat);
	private static native long nInv(long self);
    private static native long nEye(int rows, int cols, int type);

}
