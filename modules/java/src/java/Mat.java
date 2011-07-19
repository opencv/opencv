package org.opencv;

//javadoc:Mat
public class Mat {


    protected Mat(long nativeMat) {
        /*if(nativeMat == 0) 
            throw new java.lang.UnsupportedOperationException("Native object address is NULL");*/
        this.nativeObj = nativeMat;
    }
    
    //javadoc:Mat::Mat()
    public Mat() {
        this( nCreateMat() );
    }

    //javadoc:Mat::Mat(rows,cols,type)
    public Mat(int rows, int cols, CvType type) {
        this( nCreateMat(rows, cols, type.toInt()) );
    }

    //javadoc:Mat::Mat(rows,cols,depth)
    public Mat(int rows, int cols, int depth) {
        this( rows, cols, new CvType(depth, 1) );
    }

    //javadoc:Mat::Mat(rows,cols,type,s)
    public Mat(int rows, int cols, CvType type, Scalar s) {
        this( nCreateMat(rows, cols, type.toInt(), s.val[0], s.val[1], s.val[2], s.val[3]) );
    }

    //javadoc:Mat::Mat(rows,cols,depth,s)
    public Mat(int rows, int cols, int depth, Scalar s) {
        this( rows, cols, new CvType(depth, 1), s );
    }

    //javadoc:Mat::dispose()
    public void dispose() {
        nRelease(nativeObj);
    }
    
    //javadoc:Mat::finalize()
    @Override
    protected void finalize() throws Throwable {
        nDelete(nativeObj);
        nativeObj = 0;
        super.finalize();
    }

    //javadoc:Mat::toString()
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
        if(nativeObj == 0) return new Size();
        return new Size(nSize(nativeObj));
    }

    private void checkNull() {
        if(nativeObj == 0) 
            throw new java.lang.UnsupportedOperationException("Native object address is NULL");
    }

    //javadoc:Mat::type()
    public CvType type() {
        checkNull();
        return new CvType( nType(nativeObj) );
    }

    //javadoc:Mat::depth()
    public int depth() { return type().depth(); }

    //javadoc:Mat::channels()
    public int channels() { return type().channels(); }

    //javadoc:Mat::elemSize()
    public int elemSize() { return type().CV_ELEM_SIZE(); }

    //javadoc:Mat::rows()
    public int rows() {
        if(nativeObj == 0) 
            return 0;
        return nRows(nativeObj);
    }

    //javadoc:Mat::height()
    public int height() { return rows(); }

    //javadoc:Mat::cols()
    public int cols() {
        if(nativeObj == 0) 
            return 0;
        return nCols(nativeObj);
    }

    //javadoc:Mat::width()
    public int width() { return cols(); }

    //javadoc:Mat::total()
    public int total() { return rows() * cols(); }

    //javadoc:Mat::dataAddr()
    public long dataAddr() {
        if(nativeObj == 0) 
            return 0;
        return nData(nativeObj);
    }

    //javadoc:Mat::isContinuous()
    public boolean isContinuous() {
        if(nativeObj == 0) 
            return false; // maybe throw an exception instead?
        return nIsCont(nativeObj);
    }

    //javadoc:Mat::isSubmatrix()
    public boolean isSubmatrix() {
        if(nativeObj == 0) 
            return false; // maybe throw an exception instead?
        return nIsSubmat(nativeObj);
    }

    //javadoc:Mat::submat(rowStart,rowEnd,colStart,colEnd)
    public Mat submat(int rowStart, int rowEnd, int colStart, int colEnd) {
        checkNull();
        return new Mat( nSubmat(nativeObj, rowStart, rowEnd, colStart, colEnd) );
    }

    //javadoc:Mat::rowRange(startrow,endrow)
    public Mat rowRange(int startrow, int endrow) { return submat(startrow, endrow, 0, -1); }

    //javadoc:Mat::row(i)
    public Mat row(int i) { return submat(i, i+1, 0, -1); }

    //javadoc:Mat::colRange(startcol,endcol)
    public Mat colRange(int startcol, int endcol) { return submat(0, -1, startcol, endcol); }

    //javadoc:Mat::col(j)
    public Mat col(int j) { return submat(0, -1, j, j+1); }
    
    //javadoc:Mat::clone()
    public Mat clone() {
        checkNull();
        return new Mat( nClone(nativeObj) );
    }

    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, double...data) {
        checkNull();
        if(data != null)
            return nPutD(nativeObj, row, col, data.length, data);
        else
            return 0;
    }

    //javadoc:Mat::put(row,col,data)
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

    //javadoc:Mat::put(row,col,data) 
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
    
    //javadoc:Mat::put(row,col,data)
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
    
    //javadoc:Mat::put(row,col,data)
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
    
    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, byte[] data) {
        checkNull();
        CvType t = type(); 
        if(t.depth() == CvType.CV_8U || t.depth() == CvType.CV_8S) {
            return nGetB(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, short[] data) {
        checkNull();
        CvType t = type(); 
        if(t.depth() == CvType.CV_16U || t.depth() == CvType.CV_16S) {
            return nGetS(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, int[] data) {
        checkNull();
        CvType t = type(); 
        if(t.depth() == CvType.CV_32S) {
            return nGetI(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, float[] data) {
        checkNull();
        CvType t = type(); 
        if(t.depth() == CvType.CV_32F) {
            return nGetF(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, double[] data) {
        checkNull();
        CvType t = type(); 
        if(t.depth() == CvType.CV_64F) {
            return nGetD(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col)
    public double[] get(int row, int col) {
        checkNull();
        //CvType t = type();
        //if(t.depth() == CvType.CV_64F) {
            return nGet(nativeObj, row, col);
        //}
        //throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }


    //javadoc:Mat::setTo(s)
    public void setTo(Scalar s) {
        checkNull();
        nSetTo(nativeObj, s.val[0], s.val[1], s.val[2], s.val[3]);
    }
    
    //javadoc:Mat::copyTo(m)
    public void copyTo(Mat m) {
        checkNull();
        if(m.nativeObj == 0) 
            throw new java.lang.UnsupportedOperationException("Destination native object address is NULL");
        nCopyTo(nativeObj, m.nativeObj);
    }
    
    //javadoc:Mat::dot(m)
    public double dot(Mat m) {
        checkNull();
        return nDot(nativeObj, m.nativeObj);
    }
    
    //javadoc:Mat::cross(m)
    public Mat cross(Mat m) {
        checkNull();
        return new Mat( nCross(nativeObj, m.nativeObj) );
    }

    //javadoc:Mat::inv()
    public Mat inv() {
        checkNull();
        return new Mat( nInv(nativeObj) );
    }
    
    //javadoc:Mat::getNativeObjAddr()
    public long getNativeObjAddr() {
        return nativeObj;
    }
    
    //javadoc:Mat::eye(rows,cols,type)
    static public Mat eye(int rows, int cols, CvType type) {
        return new Mat( nEye(rows, cols, type.toInt()) );
    }
    
    // native stuff
    static { System.loadLibrary("opencv_java"); }
    protected long nativeObj;
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
