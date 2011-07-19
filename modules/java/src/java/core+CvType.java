package org.opencv.core;


public final class CvType {
	
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
		switch (depth()) {
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
		
		int ch = channels();
		if(ch<=4) return s + "C" + ch;
		else return s + "C(" + ch + ")";
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
