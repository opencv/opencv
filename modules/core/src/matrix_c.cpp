#include "opencv2/core/mat.hpp"
#include "opencv2/core/types_c.h"
#include "precomp.hpp"

// glue

CvMatND cvMatND(const cv::Mat& m)
{
    CvMatND self;
    cvInitMatNDHeader(&self, m.dims, m.size, m.type(), m.data );
    int i, d = m.dims;
    for( i = 0; i < d; i++ )
        self.dim[i].step = (int)m.step[i];
    self.type |= m.flags & cv::Mat::CONTINUOUS_FLAG;
    return self;
}

_IplImage cvIplImage(const cv::Mat& m)
{
    _IplImage self;
    CV_Assert( m.dims <= 2 );
    cvInitImageHeader(&self, cvSize(m.size()), cvIplDepth(m.flags), m.channels());
    cvSetData(&self, m.data, (int)m.step[0]);
    return self;
}

namespace cv {

static Mat cvMatToMat(const CvMat* m, bool copyData)
{
    Mat thiz;

    if( !m )
        return thiz;

    if( !copyData )
    {
        thiz.flags = Mat::MAGIC_VAL + (m->type & (CV_MAT_TYPE_MASK|CV_MAT_CONT_FLAG));
        thiz.dims = 2;
        thiz.rows = m->rows;
        thiz.cols = m->cols;
        thiz.datastart = thiz.data = m->data.ptr;
        size_t esz = CV_ELEM_SIZE(m->type), minstep = thiz.cols*esz, _step = m->step;
        if( _step == 0 )
            _step = minstep;
        thiz.datalimit = thiz.datastart + _step*thiz.rows;
        thiz.dataend = thiz.datalimit - _step + minstep;
        thiz.step[0] = _step; thiz.step[1] = esz;
    }
    else
    {
        thiz.datastart = thiz.dataend = thiz.data = 0;
        Mat(m->rows, m->cols, m->type, m->data.ptr, m->step).copyTo(thiz);
    }

    return thiz;
}

static Mat cvMatNDToMat(const CvMatND* m, bool copyData)
{
    Mat thiz;

    if( !m )
        return thiz;
    thiz.datastart = thiz.data = m->data.ptr;
    thiz.flags |= CV_MAT_TYPE(m->type);
    int _sizes[CV_MAX_DIM];
    size_t _steps[CV_MAX_DIM];

    int d = m->dims;
    for( int i = 0; i < d; i++ )
    {
        _sizes[i] = m->dim[i].size;
        _steps[i] = m->dim[i].step;
    }

    setSize(thiz, d, _sizes, _steps);
    finalizeHdr(thiz);

    if( copyData )
    {
        Mat temp(thiz);
        thiz.release();
        temp.copyTo(thiz);
    }

    return thiz;
}

static Mat iplImageToMat(const IplImage* img, bool copyData)
{
    Mat m;

    if( !img )
        return m;

    m.dims = 2;
    CV_DbgAssert(CV_IS_IMAGE(img) && img->imageData != 0);

    int imgdepth = IPL2CV_DEPTH(img->depth);
    size_t esz;
    m.step[0] = img->widthStep;

    if(!img->roi)
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL);
        m.flags = Mat::MAGIC_VAL + CV_MAKETYPE(imgdepth, img->nChannels);
        m.rows = img->height;
        m.cols = img->width;
        m.datastart = m.data = (uchar*)img->imageData;
        esz = CV_ELEM_SIZE(m.flags);
    }
    else
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL || img->roi->coi != 0);
        bool selectedPlane = img->roi->coi && img->dataOrder == IPL_DATA_ORDER_PLANE;
        m.flags = Mat::MAGIC_VAL + CV_MAKETYPE(imgdepth, selectedPlane ? 1 : img->nChannels);
        m.rows = img->roi->height;
        m.cols = img->roi->width;
        esz = CV_ELEM_SIZE(m.flags);
        m.datastart = m.data = (uchar*)img->imageData +
            (selectedPlane ? (img->roi->coi - 1)*m.step*img->height : 0) +
            img->roi->yOffset*m.step[0] + img->roi->xOffset*esz;
    }
    m.datalimit = m.datastart + m.step.p[0]*m.rows;
    m.dataend = m.datastart + m.step.p[0]*(m.rows-1) + esz*m.cols;
    m.step[1] = esz;
    m.updateContinuityFlag();

    if( copyData )
    {
        Mat m2 = m;
        m.release();
        if( !img->roi || !img->roi->coi ||
            img->dataOrder == IPL_DATA_ORDER_PLANE)
            m2.copyTo(m);
        else
        {
            int ch[] = {img->roi->coi - 1, 0};
            m.create(m2.rows, m2.cols, m2.type());
            mixChannels(&m2, 1, &m, 1, ch, 1);
        }
    }

    return m;
}

Mat cvarrToMat(const CvArr* arr, bool copyData,
               bool /*allowND*/, int coiMode, AutoBuffer<double>* abuf )
{
    if( !arr )
        return Mat();
    if( CV_IS_MAT_HDR_Z(arr) )
        return cvMatToMat((const CvMat*)arr, copyData);
    if( CV_IS_MATND(arr) )
        return cvMatNDToMat((const CvMatND*)arr, copyData );
    if( CV_IS_IMAGE(arr) )
    {
        const IplImage* iplimg = (const IplImage*)arr;
        if( coiMode == 0 && iplimg->roi && iplimg->roi->coi > 0 )
            CV_Error(CV_BadCOI, "COI is not supported by the function");
        return iplImageToMat(iplimg, copyData);
    }
    if( CV_IS_SEQ(arr) )
    {
        CvSeq* seq = (CvSeq*)arr;
        int total = seq->total, type = CV_MAT_TYPE(seq->flags), esz = seq->elem_size;
        if( total == 0 )
            return Mat();
        CV_Assert(total > 0 && CV_ELEM_SIZE(seq->flags) == esz);
        if(!copyData && seq->first->next == seq->first)
            return Mat(total, 1, type, seq->first->data);
        if( abuf )
        {
            abuf->allocate(((size_t)total*esz + sizeof(double)-1)/sizeof(double));
            double* bufdata = abuf->data();
            cvCvtSeqToArray(seq, bufdata, CV_WHOLE_SEQ);
            return Mat(total, 1, type, bufdata);
        }

        Mat buf(total, 1, type);
        cvCvtSeqToArray(seq, buf.ptr(), CV_WHOLE_SEQ);
        return buf;
    }
    CV_Error(CV_StsBadArg, "Unknown array type");
}

void extractImageCOI(const CvArr* arr, OutputArray _ch, int coi)
{
    Mat mat = cvarrToMat(arr, false, true, 1);
    _ch.create(mat.dims, mat.size, mat.depth());
    Mat ch = _ch.getMat();
    if(coi < 0)
    {
        CV_Assert( CV_IS_IMAGE(arr) );
        coi = cvGetImageCOI((const IplImage*)arr)-1;
    }
    CV_Assert(0 <= coi && coi < mat.channels());
    int _pairs[] = { coi, 0 };
    mixChannels( &mat, 1, &ch, 1, _pairs, 1 );
}

void insertImageCOI(InputArray _ch, CvArr* arr, int coi)
{
    Mat ch = _ch.getMat(), mat = cvarrToMat(arr, false, true, 1);
    if(coi < 0)
    {
        CV_Assert( CV_IS_IMAGE(arr) );
        coi = cvGetImageCOI((const IplImage*)arr)-1;
    }
    CV_Assert(ch.size == mat.size && ch.depth() == mat.depth() && 0 <= coi && coi < mat.channels());
    int _pairs[] = { 0, coi };
    mixChannels( &ch, 1, &mat, 1, _pairs, 1 );
}

} // cv::

// operations

CV_IMPL void cvSetIdentity( CvArr* arr, CvScalar value )
{
    cv::Mat m = cv::cvarrToMat(arr);
    cv::setIdentity(m, value);
}


CV_IMPL CvScalar cvTrace( const CvArr* arr )
{
    return cvScalar(cv::trace(cv::cvarrToMat(arr)));
}


CV_IMPL void cvTranspose( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.rows == dst.cols && src.cols == dst.rows && src.type() == dst.type() );
    transpose( src, dst );
}


CV_IMPL void cvCompleteSymm( CvMat* matrix, int LtoR )
{
    cv::Mat m = cv::cvarrToMat(matrix);
    cv::completeSymm( m, LtoR != 0 );
}


CV_IMPL void cvCrossProduct( const CvArr* srcAarr, const CvArr* srcBarr, CvArr* dstarr )
{
    cv::Mat srcA = cv::cvarrToMat(srcAarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( srcA.size() == dst.size() && srcA.type() == dst.type() );
    srcA.cross(cv::cvarrToMat(srcBarr)).copyTo(dst);
}


CV_IMPL void
cvReduce( const CvArr* srcarr, CvArr* dstarr, int dim, int op )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    if( dim < 0 )
        dim = src.rows > dst.rows ? 0 : src.cols > dst.cols ? 1 : dst.cols == 1;

    if( dim > 1 )
        CV_Error( CV_StsOutOfRange, "The reduced dimensionality index is out of range" );

    if( (dim == 0 && (dst.cols != src.cols || dst.rows != 1)) ||
        (dim == 1 && (dst.rows != src.rows || dst.cols != 1)) )
        CV_Error( CV_StsBadSize, "The output array size is incorrect" );

    if( src.channels() != dst.channels() )
        CV_Error( CV_StsUnmatchedFormats, "Input and output arrays must have the same number of channels" );

    cv::reduce(src, dst, dim, op, dst.type());
}


CV_IMPL CvArr*
cvRange( CvArr* arr, double start, double end )
{
    CvMat stub, *mat = (CvMat*)arr;
    int step;
    double val = start;

    if( !CV_IS_MAT(mat) )
        mat = cvGetMat( mat, &stub);

    int rows = mat->rows;
    int cols = mat->cols;
    int type = CV_MAT_TYPE(mat->type);
    double delta = (end-start)/(rows*cols);

    if( CV_IS_MAT_CONT(mat->type) )
    {
        cols *= rows;
        rows = 1;
        step = 1;
    }
    else
        step = mat->step / CV_ELEM_SIZE(type);

    if( type == CV_32SC1 )
    {
        int* idata = mat->data.i;
        int ival = cvRound(val), idelta = cvRound(delta);

        if( fabs(val - ival) < DBL_EPSILON &&
            fabs(delta - idelta) < DBL_EPSILON )
        {
            for( int i = 0; i < rows; i++, idata += step )
                for( int j = 0; j < cols; j++, ival += idelta )
                    idata[j] = ival;
        }
        else
        {
            for( int i = 0; i < rows; i++, idata += step )
                for( int j = 0; j < cols; j++, val += delta )
                    idata[j] = cvRound(val);
        }
    }
    else if( type == CV_32FC1 )
    {
        float* fdata = mat->data.fl;
        for( int i = 0; i < rows; i++, fdata += step )
            for( int j = 0; j < cols; j++, val += delta )
                fdata[j] = (float)val;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "The function only supports 32sC1 and 32fC1 datatypes" );

    return arr;
}


CV_IMPL void
cvSort( const CvArr* _src, CvArr* _dst, CvArr* _idx, int flags )
{
    cv::Mat src = cv::cvarrToMat(_src);

    if( _idx )
    {
        cv::Mat idx0 = cv::cvarrToMat(_idx), idx = idx0;
        CV_Assert( src.size() == idx.size() && idx.type() == CV_32S && src.data != idx.data );
        cv::sortIdx( src, idx, flags );
        CV_Assert( idx0.data == idx.data );
    }

    if( _dst )
    {
        cv::Mat dst0 = cv::cvarrToMat(_dst), dst = dst0;
        CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
        cv::sort( src, dst, flags );
        CV_Assert( dst0.data == dst.data );
    }
}


CV_IMPL int
cvKMeans2( const CvArr* _samples, int cluster_count, CvArr* _labels,
           CvTermCriteria termcrit, int attempts, CvRNG*,
           int flags, CvArr* _centers, double* _compactness )
{
    cv::Mat data = cv::cvarrToMat(_samples), labels = cv::cvarrToMat(_labels), centers;
    if( _centers )
    {
        centers = cv::cvarrToMat(_centers);

        centers = centers.reshape(1);
        data = data.reshape(1);

        CV_Assert( !centers.empty() );
        CV_Assert( centers.rows == cluster_count );
        CV_Assert( centers.cols == data.cols );
        CV_Assert( centers.depth() == data.depth() );
    }
    CV_Assert( labels.isContinuous() && labels.type() == CV_32S &&
        (labels.cols == 1 || labels.rows == 1) &&
        labels.cols + labels.rows - 1 == data.rows );

    double compactness = cv::kmeans(data, cluster_count, labels, termcrit, attempts,
                                    flags, _centers ? cv::_OutputArray(centers) : cv::_OutputArray() );
    if( _compactness )
        *_compactness = compactness;
    return 1;
}
