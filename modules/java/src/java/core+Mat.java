package org.opencv.core;

// C++: class Mat
//javadoc: Mat
public class Mat {

    public final long nativeObj;
	public Mat(long addr)
	{
        if(addr == 0) 
            throw new java.lang.UnsupportedOperationException("Native object address is NULL");
        nativeObj = addr;
    }


    //
    // C++:   Mat::Mat()
    //

    //javadoc: Mat::Mat()
    public   Mat()
    {
        
        nativeObj = n_Mat();
        
        return;
    }


    //
    // C++:   Mat::Mat(int rows, int cols, int type)
    //

    //javadoc: Mat::Mat(rows, cols, type)
    public   Mat(int rows, int cols, int type)
    {
        
        nativeObj = n_Mat(rows, cols, type);
        
        return;
    }


    //
    // C++:   Mat::Mat(Size size, int type)
    //

    //javadoc: Mat::Mat(size, type)
    public   Mat(Size size, int type)
    {
        
        nativeObj = n_Mat(size.width, size.height, type);
        
        return;
    }


    //
    // C++:   Mat::Mat(int rows, int cols, int type, Scalar s)
    //

    //javadoc: Mat::Mat(rows, cols, type, s)
    public   Mat(int rows, int cols, int type, Scalar s)
    {
        
        nativeObj = n_Mat(rows, cols, type, s.val[0], s.val[1], s.val[2], s.val[3]);
        
        return;
    }


    //
    // C++:   Mat::Mat(Size size, int type, Scalar s)
    //

    //javadoc: Mat::Mat(size, type, s)
    public   Mat(Size size, int type, Scalar s)
    {
        
        nativeObj = n_Mat(size.width, size.height, type, s.val[0], s.val[1], s.val[2], s.val[3]);
        
        return;
    }


    //
    // C++:   Mat::Mat(Mat m, Range rowRange, Range colRange = Range::all())
    //

    //javadoc: Mat::Mat(m, rowRange, colRange)
    public   Mat(Mat m, Range rowRange, Range colRange)
    {
        
        nativeObj = n_Mat(m.nativeObj, rowRange.start, rowRange.end, colRange.start, colRange.end);
        
        return;
    }

    //javadoc: Mat::Mat(m, rowRange)
    public   Mat(Mat m, Range rowRange)
    {
        
        nativeObj = n_Mat(m.nativeObj, rowRange.start, rowRange.end);
        
        return;
    }


    //
    // C++:   Mat::Mat(Mat m, Rect roi)
    //

    //javadoc: Mat::Mat(m, roi)
    public   Mat(Mat m, Rect roi)
    {
        
        nativeObj = n_Mat(m.nativeObj, roi.x, roi.y, roi.width, roi.height);
        
        return;
    }


    //
    // C++:  Mat Mat::adjustROI(int dtop, int dbottom, int dleft, int dright)
    //

    //javadoc: Mat::adjustROI(dtop, dbottom, dleft, dright)
    public  Mat adjustROI(int dtop, int dbottom, int dleft, int dright)
    {
        
        Mat retVal = new Mat(n_adjustROI(nativeObj, dtop, dbottom, dleft, dright));
        
        return retVal;
    }


    //
    // C++:  void Mat::assignTo(Mat m, int type = -1)
    //

    //javadoc: Mat::assignTo(m, type)
    public  void assignTo(Mat m, int type)
    {
        
        n_assignTo(nativeObj, m.nativeObj, type);
        
        return;
    }

    //javadoc: Mat::assignTo(m)
    public  void assignTo(Mat m)
    {
        
        n_assignTo(nativeObj, m.nativeObj);
        
        return;
    }


    //
    // C++:  int Mat::channels()
    //

    //javadoc: Mat::channels()
    public  int channels()
    {
        
        int retVal = n_channels(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int Mat::checkVector(int elemChannels, int depth = -1, bool requireContinuous = true)
    //

    //javadoc: Mat::checkVector(elemChannels, depth, requireContinuous)
    public  int checkVector(int elemChannels, int depth, boolean requireContinuous)
    {
        
        int retVal = n_checkVector(nativeObj, elemChannels, depth, requireContinuous);
        
        return retVal;
    }

    //javadoc: Mat::checkVector(elemChannels, depth)
    public  int checkVector(int elemChannels, int depth)
    {
        
        int retVal = n_checkVector(nativeObj, elemChannels, depth);
        
        return retVal;
    }

    //javadoc: Mat::checkVector(elemChannels)
    public  int checkVector(int elemChannels)
    {
        
        int retVal = n_checkVector(nativeObj, elemChannels);
        
        return retVal;
    }


    //
    // C++:  Mat Mat::clone()
    //

    //javadoc: Mat::clone()
    public  Mat clone()
    {
        
        Mat retVal = new Mat(n_clone(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::col(int x)
    //

    //javadoc: Mat::col(x)
    public  Mat col(int x)
    {
        
        Mat retVal = new Mat(n_col(nativeObj, x));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::colRange(int startcol, int endcol)
    //

    //javadoc: Mat::colRange(startcol, endcol)
    public  Mat colRange(int startcol, int endcol)
    {
        
        Mat retVal = new Mat(n_colRange(nativeObj, startcol, endcol));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::colRange(Range r)
    //

    //javadoc: Mat::colRange(r)
    public  Mat colRange(Range r)
    {
        
        Mat retVal = new Mat(n_colRange(nativeObj, r.start, r.end));
        
        return retVal;
    }


    //
    // C++:  int Mat::cols()
    //

    //javadoc: Mat::cols()
    public  int cols()
    {
        
        int retVal = n_cols(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void Mat::convertTo(Mat& m, int rtype, double alpha = 1, double beta = 0)
    //

    //javadoc: Mat::convertTo(m, rtype, alpha, beta)
    public  void convertTo(Mat m, int rtype, double alpha, double beta)
    {
        
        n_convertTo(nativeObj, m.nativeObj, rtype, alpha, beta);
        
        return;
    }

    //javadoc: Mat::convertTo(m, rtype, alpha)
    public  void convertTo(Mat m, int rtype, double alpha)
    {
        
        n_convertTo(nativeObj, m.nativeObj, rtype, alpha);
        
        return;
    }

    //javadoc: Mat::convertTo(m, rtype)
    public  void convertTo(Mat m, int rtype)
    {
        
        n_convertTo(nativeObj, m.nativeObj, rtype);
        
        return;
    }


    //
    // C++:  void Mat::copyTo(Mat& m)
    //

    //javadoc: Mat::copyTo(m)
    public  void copyTo(Mat m)
    {
        
        n_copyTo(nativeObj, m.nativeObj);
        
        return;
    }


    //
    // C++:  void Mat::copyTo(Mat& m, Mat mask)
    //

    //javadoc: Mat::copyTo(m, mask)
    public  void copyTo(Mat m, Mat mask)
    {
        
        n_copyTo(nativeObj, m.nativeObj, mask.nativeObj);
        
        return;
    }


    //
    // C++:  void Mat::create(int rows, int cols, int type)
    //

    //javadoc: Mat::create(rows, cols, type)
    public  void create(int rows, int cols, int type)
    {
        
        n_create(nativeObj, rows, cols, type);
        
        return;
    }


    //
    // C++:  void Mat::create(Size size, int type)
    //

    //javadoc: Mat::create(size, type)
    public  void create(Size size, int type)
    {
        
        n_create(nativeObj, size.width, size.height, type);
        
        return;
    }


    //
    // C++:  Mat Mat::cross(Mat m)
    //

    //javadoc: Mat::cross(m)
    public  Mat cross(Mat m)
    {
        
        Mat retVal = new Mat(n_cross(nativeObj, m.nativeObj));
        
        return retVal;
    }


    //
    // C++:  long Mat::dataAddr()
    //

    //javadoc: Mat::dataAddr()
    public  long dataAddr()
    {
        
        long retVal = n_dataAddr(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int Mat::depth()
    //

    //javadoc: Mat::depth()
    public  int depth()
    {
        
        int retVal = n_depth(nativeObj);
        
        return retVal;
    }


    //
    // C++:  Mat Mat::diag(int d = 0)
    //

    //javadoc: Mat::diag(d)
    public  Mat diag(int d)
    {
        
        Mat retVal = new Mat(n_diag(nativeObj, d));
        
        return retVal;
    }

    //javadoc: Mat::diag()
    public  Mat diag()
    {
        
        Mat retVal = new Mat(n_diag(nativeObj, 0));
        
        return retVal;
    }


    //
    // C++: static Mat Mat::diag(Mat d)
    //

    //javadoc: Mat::diag(d)
    public static Mat diag(Mat d)
    {
        
        Mat retVal = new Mat(n_diag(d.nativeObj));
        
        return retVal;
    }


    //
    // C++:  double Mat::dot(Mat m)
    //

    //javadoc: Mat::dot(m)
    public  double dot(Mat m)
    {
        
        double retVal = n_dot(nativeObj, m.nativeObj);
        
        return retVal;
    }


    //
    // C++:  size_t Mat::elemSize()
    //

    //javadoc: Mat::elemSize()
    public  long elemSize()
    {
        
        long retVal = n_elemSize(nativeObj);
        
        return retVal;
    }


    //
    // C++:  size_t Mat::elemSize1()
    //

    //javadoc: Mat::elemSize1()
    public  long elemSize1()
    {
        
        long retVal = n_elemSize1(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool Mat::empty()
    //

    //javadoc: Mat::empty()
    public  boolean empty()
    {
        
        boolean retVal = n_empty(nativeObj);
        
        return retVal;
    }


    //
    // C++: static Mat Mat::eye(int rows, int cols, int type)
    //

    //javadoc: Mat::eye(rows, cols, type)
    public static Mat eye(int rows, int cols, int type)
    {
        
        Mat retVal = new Mat(n_eye(rows, cols, type));
        
        return retVal;
    }


    //
    // C++: static Mat Mat::eye(Size size, int type)
    //

    //javadoc: Mat::eye(size, type)
    public static Mat eye(Size size, int type)
    {
        
        Mat retVal = new Mat(n_eye(size.width, size.height, type));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::inv(int method = DECOMP_LU)
    //

    //javadoc: Mat::inv(method)
    public  Mat inv(int method)
    {
        
        Mat retVal = new Mat(n_inv(nativeObj, method));
        
        return retVal;
    }

    //javadoc: Mat::inv()
    public  Mat inv()
    {
        
        Mat retVal = new Mat(n_inv(nativeObj));
        
        return retVal;
    }


    //
    // C++:  bool Mat::isContinuous()
    //

    //javadoc: Mat::isContinuous()
    public  boolean isContinuous()
    {
        
        boolean retVal = n_isContinuous(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool Mat::isSubmatrix()
    //

    //javadoc: Mat::isSubmatrix()
    public  boolean isSubmatrix()
    {
        
        boolean retVal = n_isSubmatrix(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void Mat::locateROI(Size wholeSize, Point ofs)
    //

    //javadoc: Mat::locateROI(wholeSize, ofs)
    public  void locateROI(Size wholeSize, Point ofs)
    {
        
        n_locateROI(nativeObj, wholeSize.width, wholeSize.height, ofs.x, ofs.y);
        
        return;
    }


    //
    // C++:  Mat Mat::mul(Mat m, double scale = 1)
    //

    //javadoc: Mat::mul(m, scale)
    public  Mat mul(Mat m, double scale)
    {
        
        Mat retVal = new Mat(n_mul(nativeObj, m.nativeObj, scale));
        
        return retVal;
    }

    //javadoc: Mat::mul(m)
    public  Mat mul(Mat m)
    {
        
        Mat retVal = new Mat(n_mul(nativeObj, m.nativeObj));
        
        return retVal;
    }


    //
    // C++: static Mat Mat::ones(int rows, int cols, int type)
    //

    //javadoc: Mat::ones(rows, cols, type)
    public static Mat ones(int rows, int cols, int type)
    {
        
        Mat retVal = new Mat(n_ones(rows, cols, type));
        
        return retVal;
    }


    //
    // C++: static Mat Mat::ones(Size size, int type)
    //

    //javadoc: Mat::ones(size, type)
    public static Mat ones(Size size, int type)
    {
        
        Mat retVal = new Mat(n_ones(size.width, size.height, type));
        
        return retVal;
    }


    //
    // C++:  void Mat::push_back(Mat m)
    //

    //javadoc: Mat::push_back(m)
    public  void push_back(Mat m)
    {
        
        n_push_back(nativeObj, m.nativeObj);
        
        return;
    }


    //
    // C++:  void Mat::release()
    //

    //javadoc: Mat::release()
    public  void release()
    {
        
        n_release(nativeObj);
        
        return;
    }


    //
    // C++:  Mat Mat::reshape(int cn, int rows = 0)
    //

    //javadoc: Mat::reshape(cn, rows)
    public  Mat reshape(int cn, int rows)
    {
        
        Mat retVal = new Mat(n_reshape(nativeObj, cn, rows));
        
        return retVal;
    }

    //javadoc: Mat::reshape(cn)
    public  Mat reshape(int cn)
    {
        
        Mat retVal = new Mat(n_reshape(nativeObj, cn));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::row(int y)
    //

    //javadoc: Mat::row(y)
    public  Mat row(int y)
    {
        
        Mat retVal = new Mat(n_row(nativeObj, y));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::rowRange(int startrow, int endrow)
    //

    //javadoc: Mat::rowRange(startrow, endrow)
    public  Mat rowRange(int startrow, int endrow)
    {
        
        Mat retVal = new Mat(n_rowRange(nativeObj, startrow, endrow));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::rowRange(Range r)
    //

    //javadoc: Mat::rowRange(r)
    public  Mat rowRange(Range r)
    {
        
        Mat retVal = new Mat(n_rowRange(nativeObj, r.start, r.end));
        
        return retVal;
    }


    //
    // C++:  int Mat::rows()
    //

    //javadoc: Mat::rows()
    public  int rows()
    {
        
        int retVal = n_rows(nativeObj);
        
        return retVal;
    }


    //
    // C++:  Mat Mat::operator =(Scalar s)
    //

    //javadoc: Mat::operator =(s)
    public  Mat setTo(Scalar s)
    {
        
        Mat retVal = new Mat(n_setTo(nativeObj, s.val[0], s.val[1], s.val[2], s.val[3]));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::setTo(Mat value, Mat mask = Mat())
    //

    //javadoc: Mat::setTo(value, mask)
    public  Mat setTo(Mat value, Mat mask)
    {
        
        Mat retVal = new Mat(n_setTo(nativeObj, value.nativeObj, mask.nativeObj));
        
        return retVal;
    }

    //javadoc: Mat::setTo(value)
    public  Mat setTo(Mat value)
    {
        
        Mat retVal = new Mat(n_setTo(nativeObj, value.nativeObj));
        
        return retVal;
    }


    //
    // C++:  Size Mat::size()
    //

    //javadoc: Mat::size()
    public  Size size()
    {
        
        Size retVal = new Size(n_size(nativeObj));
        
        return retVal;
    }


    //
    // C++:  size_t Mat::step1(int i = 0)
    //

    //javadoc: Mat::step1(i)
    public  long step1(int i)
    {
        
        long retVal = n_step1(nativeObj, i);
        
        return retVal;
    }

    //javadoc: Mat::step1()
    public  long step1()
    {
        
        long retVal = n_step1(nativeObj);
        
        return retVal;
    }


    //
    // C++:  Mat Mat::operator()(int rowStart, int rowEnd, int colStart, int colEnd)
    //

    //javadoc: Mat::operator()(rowStart, rowEnd, colStart, colEnd)
    public  Mat submat(int rowStart, int rowEnd, int colStart, int colEnd)
    {
        
        Mat retVal = new Mat(n_submat_rr(nativeObj, rowStart, rowEnd, colStart, colEnd));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::operator()(Range rowRange, Range colRange)
    //

    //javadoc: Mat::operator()(rowRange, colRange)
    public  Mat submat(Range rowRange, Range colRange)
    {
        
        Mat retVal = new Mat(n_submat_rr(nativeObj, rowRange.start, rowRange.end, colRange.start, colRange.end));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::operator()(Rect roi)
    //

    //javadoc: Mat::operator()(roi)
    public  Mat submat(Rect roi)
    {
        
        Mat retVal = new Mat(n_submat(nativeObj, roi.x, roi.y, roi.width, roi.height));
        
        return retVal;
    }


    //
    // C++:  Mat Mat::t()
    //

    //javadoc: Mat::t()
    public  Mat t()
    {
        
        Mat retVal = new Mat(n_t(nativeObj));
        
        return retVal;
    }


    //
    // C++:  size_t Mat::total()
    //

    //javadoc: Mat::total()
    public  long total()
    {
        
        long retVal = n_total(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int Mat::type()
    //

    //javadoc: Mat::type()
    public  int type()
    {
        
        int retVal = n_type(nativeObj);
        
        return retVal;
    }


    //
    // C++: static Mat Mat::zeros(int rows, int cols, int type)
    //

    //javadoc: Mat::zeros(rows, cols, type)
    public static Mat zeros(int rows, int cols, int type)
    {
        
        Mat retVal = new Mat(n_zeros(rows, cols, type));
        
        return retVal;
    }


    //
    // C++: static Mat Mat::zeros(Size size, int type)
    //

    //javadoc: Mat::zeros(size, type)
    public static Mat zeros(Size size, int type)
    {
        
        Mat retVal = new Mat(n_zeros(size.width, size.height, type));
        
        return retVal;
    }


    @Override
    protected void finalize() throws Throwable {
        n_delete(nativeObj);
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
	
    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, double...data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
         return nPutD(nativeObj, row, col, data.length, data);
    }

    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, float[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_32F) {
            return nPutF(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::put(row,col,data) 
    public int put(int row, int col, int[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_32S) {
            return nPutI(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }
    
    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, short[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_16U || CvType.depth(t) == CvType.CV_16S) {
            return nPutS(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }
    
    //javadoc:Mat::put(row,col,data)
    public int put(int row, int col, byte[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_8U || CvType.depth(t) == CvType.CV_8S) {
            return nPutB(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }
    
    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, byte[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_8U || CvType.depth(t) == CvType.CV_8S) {
            return nGetB(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, short[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_16U || CvType.depth(t) == CvType.CV_16S) {
            return nGetS(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, int[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_32S) {
            return nGetI(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, float[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_32F) {
            return nGetF(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col,data)
    public int get(int row, int col, double[] data) {
        int t = type();
    	if(data == null || data.length % CvType.channels(t) != 0)
    		throw new java.lang.UnsupportedOperationException(
    				"Provided data element number (" +
    				(data == null ? 0 : data.length) + 
    				") should be multiple of the Mat channels count (" +
    				CvType.channels(t) + ")");
        if(CvType.depth(t) == CvType.CV_64F) {
            return nGetD(nativeObj, row, col, data.length, data);
        }
        throw new java.lang.UnsupportedOperationException("Mat data type is not compatible: " + t);
    }

    //javadoc:Mat::get(row,col)
    public double[] get(int row, int col) {
        return nGet(nativeObj, row, col);
    }

    //javadoc:Mat::height()
    public int height() { 
    	return rows(); 
    }
	
    //javadoc:Mat::width()
    public int width() { 
    	return cols(); 
    }

    //javadoc:Mat::getNativeObjAddr()
    public long getNativeObjAddr() {
        return nativeObj;
    }


    //
    // native stuff
    //
    static { System.loadLibrary("opencv_java"); }

    // C++:   Mat::Mat()
    private static native long n_Mat();

    // C++:   Mat::Mat(int rows, int cols, int type)
    private static native long n_Mat(int rows, int cols, int type);

    // C++:   Mat::Mat(Size size, int type)
    private static native long n_Mat(double size_width, double size_height, int type);

    // C++:   Mat::Mat(int rows, int cols, int type, Scalar s)
    private static native long n_Mat(int rows, int cols, int type, double s_val0, double s_val1, double s_val2, double s_val3);

    // C++:   Mat::Mat(Size size, int type, Scalar s)
    private static native long n_Mat(double size_width, double size_height, int type, double s_val0, double s_val1, double s_val2, double s_val3);

    // C++:   Mat::Mat(Mat m, Range rowRange, Range colRange = Range::all())
    private static native long n_Mat(long m_nativeObj, int rowRange_start, int rowRange_end, int colRange_start, int colRange_end);
    private static native long n_Mat(long m_nativeObj, int rowRange_start, int rowRange_end);

    // C++:  Mat Mat::adjustROI(int dtop, int dbottom, int dleft, int dright)
    private static native long n_adjustROI(long nativeObj, int dtop, int dbottom, int dleft, int dright);

    // C++:  void Mat::assignTo(Mat m, int type = -1)
    private static native void n_assignTo(long nativeObj, long m_nativeObj, int type);
    private static native void n_assignTo(long nativeObj, long m_nativeObj);

    // C++:  int Mat::channels()
    private static native int n_channels(long nativeObj);

    // C++:  int Mat::checkVector(int elemChannels, int depth = -1, bool requireContinuous = true)
    private static native int n_checkVector(long nativeObj, int elemChannels, int depth, boolean requireContinuous);
    private static native int n_checkVector(long nativeObj, int elemChannels, int depth);
    private static native int n_checkVector(long nativeObj, int elemChannels);

    // C++:  Mat Mat::clone()
    private static native long n_clone(long nativeObj);

    // C++:  Mat Mat::col(int x)
    private static native long n_col(long nativeObj, int x);

    // C++:  Mat Mat::colRange(int startcol, int endcol)
    private static native long n_colRange(long nativeObj, int startcol, int endcol);

    // C++:  int Mat::cols()
    private static native int n_cols(long nativeObj);

    // C++:  void Mat::convertTo(Mat& m, int rtype, double alpha = 1, double beta = 0)
    private static native void n_convertTo(long nativeObj, long m_nativeObj, int rtype, double alpha, double beta);
    private static native void n_convertTo(long nativeObj, long m_nativeObj, int rtype, double alpha);
    private static native void n_convertTo(long nativeObj, long m_nativeObj, int rtype);

    // C++:  void Mat::copyTo(Mat& m)
    private static native void n_copyTo(long nativeObj, long m_nativeObj);

    // C++:  void Mat::copyTo(Mat& m, Mat mask)
    private static native void n_copyTo(long nativeObj, long m_nativeObj, long mask_nativeObj);

    // C++:  void Mat::create(int rows, int cols, int type)
    private static native void n_create(long nativeObj, int rows, int cols, int type);

    // C++:  void Mat::create(Size size, int type)
    private static native void n_create(long nativeObj, double size_width, double size_height, int type);

    // C++:  Mat Mat::cross(Mat m)
    private static native long n_cross(long nativeObj, long m_nativeObj);

    // C++:  long Mat::dataAddr()
    private static native long n_dataAddr(long nativeObj);

    // C++:  int Mat::depth()
    private static native int n_depth(long nativeObj);

    // C++:  Mat Mat::diag(int d = 0)
    private static native long n_diag(long nativeObj, int d);

    // C++: static Mat Mat::diag(Mat d)
    private static native long n_diag(long d_nativeObj);

    // C++:  double Mat::dot(Mat m)
    private static native double n_dot(long nativeObj, long m_nativeObj);

    // C++:  size_t Mat::elemSize()
    private static native long n_elemSize(long nativeObj);

    // C++:  size_t Mat::elemSize1()
    private static native long n_elemSize1(long nativeObj);

    // C++:  bool Mat::empty()
    private static native boolean n_empty(long nativeObj);

    // C++: static Mat Mat::eye(int rows, int cols, int type)
    private static native long n_eye(int rows, int cols, int type);

    // C++: static Mat Mat::eye(Size size, int type)
    private static native long n_eye(double size_width, double size_height, int type);

    // C++:  Mat Mat::inv(int method = DECOMP_LU)
    private static native long n_inv(long nativeObj, int method);
    private static native long n_inv(long nativeObj);

    // C++:  bool Mat::isContinuous()
    private static native boolean n_isContinuous(long nativeObj);

    // C++:  bool Mat::isSubmatrix()
    private static native boolean n_isSubmatrix(long nativeObj);

    // C++:  void Mat::locateROI(Size wholeSize, Point ofs)
    private static native void n_locateROI(long nativeObj, double wholeSize_width, double wholeSize_height, double ofs_x, double ofs_y);

    // C++:  Mat Mat::mul(Mat m, double scale = 1)
    private static native long n_mul(long nativeObj, long m_nativeObj, double scale);
    private static native long n_mul(long nativeObj, long m_nativeObj);

    // C++: static Mat Mat::ones(int rows, int cols, int type)
    private static native long n_ones(int rows, int cols, int type);

    // C++: static Mat Mat::ones(Size size, int type)
    private static native long n_ones(double size_width, double size_height, int type);

    // C++:  void Mat::push_back(Mat m)
    private static native void n_push_back(long nativeObj, long m_nativeObj);

    // C++:  void Mat::release()
    private static native void n_release(long nativeObj);

    // C++:  Mat Mat::reshape(int cn, int rows = 0)
    private static native long n_reshape(long nativeObj, int cn, int rows);
    private static native long n_reshape(long nativeObj, int cn);

    // C++:  Mat Mat::row(int y)
    private static native long n_row(long nativeObj, int y);

    // C++:  Mat Mat::rowRange(int startrow, int endrow)
    private static native long n_rowRange(long nativeObj, int startrow, int endrow);

    // C++:  int Mat::rows()
    private static native int n_rows(long nativeObj);

    // C++:  Mat Mat::operator =(Scalar s)
    private static native long n_setTo(long nativeObj, double s_val0, double s_val1, double s_val2, double s_val3);

    // C++:  Mat Mat::setTo(Mat value, Mat mask = Mat())
    private static native long n_setTo(long nativeObj, long value_nativeObj, long mask_nativeObj);
    private static native long n_setTo(long nativeObj, long value_nativeObj);

    // C++:  Size Mat::size()
    private static native double[] n_size(long nativeObj);

    // C++:  size_t Mat::step1(int i = 0)
    private static native long n_step1(long nativeObj, int i);
    private static native long n_step1(long nativeObj);
	
	// C++:  Mat Mat::operator()(Range rowRange, Range colRange)
    private static native long n_submat_rr(long nativeObj, int rowRange_start, int rowRange_end, int colRange_start, int colRange_end);

    // C++:  Mat Mat::operator()(Rect roi)
    private static native long n_submat(long nativeObj, int roi_x, int roi_y, int roi_width, int roi_height);

    // C++:  Mat Mat::t()
    private static native long n_t(long nativeObj);

    // C++:  size_t Mat::total()
    private static native long n_total(long nativeObj);

    // C++:  int Mat::type()
    private static native int n_type(long nativeObj);

    // C++: static Mat Mat::zeros(int rows, int cols, int type)
    private static native long n_zeros(int rows, int cols, int type);

    // C++: static Mat Mat::zeros(Size size, int type)
    private static native long n_zeros(double size_width, double size_height, int type);

    // native support for java finalize()
    private static native void n_delete(long nativeObj);
	
	
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
    private static native String nDump(long self);
}
