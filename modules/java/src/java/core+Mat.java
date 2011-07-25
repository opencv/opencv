package org.opencv.core;

//javadoc:Mat
public class Mat {


    public Mat(long nativeMat) {
        if(nativeMat == 0) 
            throw new java.lang.UnsupportedOperationException("Native object address is NULL");
        this.nativeObj = nativeMat;
    }
    
    //javadoc:Mat::Mat()
    public Mat() {
        this( nCreateMat() );
    }

    //javadoc:Mat::Mat(rows,cols,type)
    public Mat(int rows, int cols, int type) {
        this( nCreateMat(rows, cols, type) );
    }

    //javadoc:Mat::Mat(rows,cols,type,s)
    public Mat(int rows, int cols, int type, Scalar s) {
        this( nCreateMat(rows, cols, type, s.val[0], s.val[1], s.val[2], s.val[3]) );
    }

    //javadoc:Mat::eye(rows,cols,type)
    public static Mat eye(int rows, int cols, int type) {
        return new Mat( nEye(rows, cols, type) );
    }
    
    //javadoc:Mat::release()
    public void release() {
        nRelease(nativeObj);
    }
    
    //javadoc:Mat::finalize()
    @Override
    protected void finalize() throws Throwable {
        nDelete(nativeObj);
        super.finalize();
    }

    //javadoc:Mat::toString()
    @Override
    public String toString() {
        return  "Mat [ " +
                rows() + "*" + cols() + "*" + CvType.typeToString(type()) +
                ", isCont=" + isContinuous() + ", isSubmat=" + isSubmatrix() +
                ", nativeObj=0x" + Long.toHexString(nativeObj) + 
                ", dataAddr=0x" + Long.toHexString(dataAddr()) +  
                " ]";
    }

    //javadoc:Mat::dump()
    public String dump() {
        return nDump(nativeObj);
    }
    
    //javadoc:Mat::empty()
    public boolean empty() {
        if(nativeObj == 0) return true;
        return nIsEmpty(nativeObj); 
    }

    //javadoc:Mat::size()
    public Size size() {
        return new Size(nSize(nativeObj));
    }

    //javadoc:Mat::type()
    public int type() {
        return nType(nativeObj);
    }

    //javadoc:Mat::depth()
    public int depth() { 
    	return CvType.depth(type()); 
    }

    //javadoc:Mat::channels()
    public int channels() { 
    	return CvType.channels(type());
    }

    //javadoc:Mat::elemSize()
    public int elemSize() { 
    	return CvType.ELEM_SIZE(type()); 
    }

    //javadoc:Mat::rows()
    public int rows() {
        return nRows(nativeObj);
    }

    //javadoc:Mat::height()
    public int height() { 
    	return rows(); 
    }

    //javadoc:Mat::cols()
    public int cols() {
        return nCols(nativeObj);
    }

    //javadoc:Mat::width()
    public int width() { 
    	return cols(); 
    }

    //javadoc:Mat::total()
    public int total() { 
    	return rows() * cols(); 
    }

    //javadoc:Mat::dataAddr()
    public long dataAddr() {
        return nData(nativeObj);
    }

    //javadoc:Mat::isContinuous()
    public boolean isContinuous() {
        return nIsCont(nativeObj);
    }

    //javadoc:Mat::isSubmatrix()
    public boolean isSubmatrix() {
        return nIsSubmat(nativeObj);
    }

    //javadoc:Mat::submat(rowStart,rowEnd,colStart,colEnd)
    public Mat submat(int rowStart, int rowEnd, int colStart, int colEnd) {
        return new Mat( nSubmat(nativeObj, rowStart, rowEnd, colStart, colEnd) );
    }

    //javadoc:Mat::rowRange(startrow,endrow)
    public Mat rowRange(int startrow, int endrow) { 
    	return submat(startrow, endrow, 0, -1); 
    }

    //javadoc:Mat::row(i)
    public Mat row(int i) { 
    	return submat(i, i+1, 0, -1); 
    }

    //javadoc:Mat::colRange(startcol,endcol)
    public Mat colRange(int startcol, int endcol) { 
    	return submat(0, -1, startcol, endcol); 
    }

    //javadoc:Mat::col(j)
    public Mat col(int j) { 
    	return submat(0, -1, j, j+1); 
    }
    
    //javadoc:Mat::clone()
    public Mat clone() {
        return new Mat( nClone(nativeObj) );
    }

    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, double...data) {
         return nPutD(nativeObj, row, col, data.length, data);
    }

    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, float[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_32F) {
            return nPutF(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::put(row,col,data) 
    public int put(int row, int col, int[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_32S) {
            return nPutI(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }
    
    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, short[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_16U || CvType.depth(t) == CvType.CV_16S) {
            return nPutS(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }
    
    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, byte[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_8U || CvType.depth(t) == CvType.CV_8S) {
            return nPutB(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }
    
    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, byte[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_8U || CvType.depth(t) == CvType.CV_8S) {
            return nGetB(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, short[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_16U || CvType.depth(t) == CvType.CV_16S) {
            return nGetS(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, int[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_32S) {
            return nGetI(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, float[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_32F) {
            return nGetF(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, double[] data) {
        int t = type();
        if(CvType.depth(t) == CvType.CV_64F) {
            return nGetD(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col)
    public double[] get(int row, int col) {
        return nGet(nativeObj, row, col);
    }


    //javadoc:Mat::setTo(s)
    public void setTo(Scalar s) {
        nSetTo(nativeObj, s.val[0], s.val[1], s.val[2], s.val[3]);
    }
    
    //javadoc:Mat::copyTo(m)
    public void copyTo(Mat m) {
        nCopyTo(nativeObj, m.nativeObj);
    }
    
    //javadoc:Mat::dot(m)
    public double dot(Mat m) {
        return nDot(nativeObj, m.nativeObj);
    }
    
    //javadoc:Mat::cross(m)
    public Mat cross(Mat m) {
        return new Mat( nCross(nativeObj, m.nativeObj) );
    }

    //javadoc:Mat::inv()
    public Mat inv() {
        return new Mat( nInv(nativeObj) );
    }
    
    //javadoc:Mat::getNativeObjAddr()
    public long getNativeObjAddr() {
        return nativeObj;
    }
    
    // native stuff
    static { System.loadLibrary("opencv_java"); }
    public final long nativeObj;
    private static native long nCreateMat();
    private static native long nCreateMat(int rows, int cols, int type);
    private static native long nCreateMat(int rows, int cols, int type, double v0, double v1, double v2, double v3);
    private static native void nRelease(long self);
    private static native void nDelete(long self);
    private static native int nType(long self);
    private static native int nRows(long self);
    private static native int nCols(long self);
    private static native long nData(long self);
    private static native boolean nIsEmpty(long self);
    private static native boolean nIsCont(long self);
    private static native boolean nIsSubmat(long self);
    private static native double[] nSize(long self);
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
    private static native String nDump(long self);

}
