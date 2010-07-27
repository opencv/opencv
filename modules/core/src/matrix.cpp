/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

/****************************************************************************************\
*                           [scaled] Identity matrix initialization                      *
\****************************************************************************************/

namespace cv {

Mat::Mat(const IplImage* img, bool copyData)
{
    CV_DbgAssert(CV_IS_IMAGE(img) && img->imageData != 0);
    
    int depth = IPL2CV_DEPTH(img->depth);
    size_t esz;
    step = img->widthStep;
    refcount = 0;

    if(!img->roi)
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL);
        flags = MAGIC_VAL + CV_MAKETYPE(depth, img->nChannels);
        rows = img->height; cols = img->width;
        datastart = data = (uchar*)img->imageData;
        esz = elemSize(); 
    }
    else
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL || img->roi->coi != 0);
        bool selectedPlane = img->roi->coi && img->dataOrder == IPL_DATA_ORDER_PLANE;
        flags = MAGIC_VAL + CV_MAKETYPE(depth, selectedPlane ? 1 : img->nChannels);
        rows = img->roi->height; cols = img->roi->width;
        esz = elemSize();
        data = datastart = (uchar*)img->imageData +
			(selectedPlane ? (img->roi->coi - 1)*step*img->height : 0) +
			img->roi->yOffset*step + img->roi->xOffset*esz;        
    }
    dataend = datastart + step*(rows-1) + esz*cols;
    flags |= (cols*esz == step || rows == 1 ? CONTINUOUS_FLAG : 0);

    if( copyData )
    {
        Mat m = *this;
        release();
        if( !img->roi || !img->roi->coi ||
            img->dataOrder == IPL_DATA_ORDER_PLANE)
            m.copyTo(*this);
        else
        {
            int ch[] = {img->roi->coi - 1, 0};
            create(m.rows, m.cols, m.type());
            mixChannels(&m, 1, this, 1, ch, 1);
        }
    }
}

    
Mat::operator IplImage() const
{
    IplImage img;
    cvInitImageHeader(&img, size(), cvIplDepth(flags), channels());
    cvSetData(&img, data, (int)step);
    return img;
}
    
    
Mat cvarrToMat(const CvArr* arr, bool copyData,
               bool allowND, int coiMode)
{
    if( CV_IS_MAT(arr) )
        return Mat((const CvMat*)arr, copyData );
    else if( CV_IS_IMAGE(arr) )
    {
        const IplImage* iplimg = (const IplImage*)arr;
        if( coiMode == 0 && iplimg->roi && iplimg->roi->coi > 0 )
            CV_Error(CV_BadCOI, "COI is not supported by the function");
        return Mat(iplimg, copyData);
    }
    else if( CV_IS_SEQ(arr) )
    {
        CvSeq* seq = (CvSeq*)arr;
        CV_Assert(seq->total > 0 && CV_ELEM_SIZE(seq->flags) == seq->elem_size);
        if(!copyData && seq->first->next == seq->first)
            return Mat(seq->total, 1, CV_MAT_TYPE(seq->flags), seq->first->data);
        Mat buf(seq->total, 1, CV_MAT_TYPE(seq->flags));
        cvCvtSeqToArray(seq, buf.data, CV_WHOLE_SEQ);
        return buf;
    }
    else
    {
        CvMat hdr, *cvmat = cvGetMat( arr, &hdr, 0, allowND ? 1 : 0 );
        if( cvmat )
            return Mat(cvmat, copyData);
    }
    return Mat();
}
    

void extractImageCOI(const CvArr* arr, Mat& ch, int coi)
{
    Mat mat = cvarrToMat(arr, false, true, 1);
    ch.create(mat.size(), mat.depth());
    if(coi < 0) 
        CV_Assert( CV_IS_IMAGE(arr) && (coi = cvGetImageCOI((const IplImage*)arr)-1) >= 0 );
    CV_Assert(0 <= coi && coi < mat.channels());
    int _pairs[] = { coi, 0 };
    mixChannels( &mat, 1, &ch, 1, _pairs, 1 );
}
    
void insertImageCOI(const Mat& ch, CvArr* arr, int coi)
{
    Mat mat = cvarrToMat(arr, false, true, 1);
    if(coi < 0) 
        CV_Assert( CV_IS_IMAGE(arr) && (coi = cvGetImageCOI((const IplImage*)arr)-1) >= 0 );
    CV_Assert(ch.size() == mat.size() && ch.depth() == mat.depth() && 0 <= coi && coi < mat.channels());
    int _pairs[] = { 0, coi };
    mixChannels( &ch, 1, &mat, 1, _pairs, 1 );
}
    

Mat Mat::reshape(int new_cn, int new_rows) const
{
    Mat hdr = *this;

    int cn = channels();
    if( new_cn == 0 )
        new_cn = cn;

    int total_width = cols * cn;

    if( (new_cn > total_width || total_width % new_cn != 0) && new_rows == 0 )
        new_rows = rows * total_width / new_cn;

    if( new_rows != 0 && new_rows != rows )
    {
        int total_size = total_width * rows;
        if( !isContinuous() )
            CV_Error( CV_BadStep,
            "The matrix is not continuous, thus its number of rows can not be changed" );

        if( (unsigned)new_rows > (unsigned)total_size )
            CV_Error( CV_StsOutOfRange, "Bad new number of rows" );

        total_width = total_size / new_rows;

        if( total_width * new_rows != total_size )
            CV_Error( CV_StsBadArg, "The total number of matrix elements "
                                    "is not divisible by the new number of rows" );

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if( new_width * new_cn != total_width )
        CV_Error( CV_BadNumChannels,
        "The total width is not divisible by the new number of channels" );

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
    return hdr;
}


void
setIdentity( Mat& m, const Scalar& s )
{
    int i, j, rows = m.rows, cols = m.cols, type = m.type();
    
    if( type == CV_32FC1 )
    {
        float* data = (float*)m.data;
        float val = (float)s[0];
        size_t step = m.step/sizeof(data[0]);

        for( i = 0; i < rows; i++, data += step )
        {
            for( j = 0; j < cols; j++ )
                data[j] = 0;
            if( i < cols )
                data[i] = val;
        }
    }
    else if( type == CV_64FC1 )
    {
        double* data = (double*)m.data;
        double val = s[0];
        size_t step = m.step/sizeof(data[0]);

        for( i = 0; i < rows; i++, data += step )
        {
            for( j = 0; j < cols; j++ )
                data[j] = j == i ? val : 0;
        }
    }
    else
    {
        m = Scalar(0);
        m.diag() = s;
    }
}

Scalar trace( const Mat& m )
{
    int i, type = m.type();
    int nm = std::min(m.rows, m.cols);
    
    if( type == CV_32FC1 )
    {
        const float* ptr = (const float*)m.data;
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }
    
    if( type == CV_64FC1 )
    {
        const double* ptr = (const double*)m.data;
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }
    
    return cv::sum(m.diag());
}


/****************************************************************************************\
*                                       transpose                                        *
\****************************************************************************************/

template<typename T> static void
transposeI_( Mat& mat )
{
    int rows = mat.rows, cols = mat.cols;
    uchar* data = mat.data;
    size_t step = mat.step;

    for( int i = 0; i < rows; i++ )
    {
        T* row = (T*)(data + step*i);
        uchar* data1 = data + i*sizeof(T);
        for( int j = i+1; j < cols; j++ )
            std::swap( row[j], *(T*)(data1 + step*j) );
    }
}

template<typename T> static void
transpose_( const Mat& src, Mat& dst )
{
    int rows = dst.rows, cols = dst.cols;
    uchar* data = src.data;
    size_t step = src.step;

    for( int i = 0; i < rows; i++ )
    {
        T* row = (T*)(dst.data + dst.step*i);
        uchar* data1 = data + i*sizeof(T);
        for( int j = 0; j < cols; j++ )
            row[j] = *(T*)(data1 + step*j);
    }
}

typedef void (*TransposeInplaceFunc)( Mat& mat );
typedef void (*TransposeFunc)( const Mat& src, Mat& dst );

void transpose( const Mat& src, Mat& dst )
{
    TransposeInplaceFunc itab[] =
    {
        0,
        transposeI_<uchar>, // 1
        transposeI_<ushort>, // 2
        transposeI_<Vec<uchar,3> >, // 3
        transposeI_<int>, // 4
        0,
        transposeI_<Vec<ushort,3> >, // 6
        0,
        transposeI_<int64>, // 8
        0, 0, 0,
        transposeI_<Vec<int,3> >, // 12
        0, 0, 0,
        transposeI_<Vec<int64,2> >, // 16
        0, 0, 0, 0, 0, 0, 0,
        transposeI_<Vec<int64,3> >, // 24
        0, 0, 0, 0, 0, 0, 0,
        transposeI_<Vec<int64,4> > // 32
    };

    TransposeFunc tab[] =
    {
        0,
        transpose_<uchar>, // 1
        transpose_<ushort>, // 2
        transpose_<Vec<uchar,3> >, // 3
        transpose_<int>, // 4
        0,
        transpose_<Vec<ushort,3> >, // 6
        0,
        transpose_<int64>, // 8
        0, 0, 0,
        transpose_<Vec<int,3> >, // 12
        0, 0, 0,
        transpose_<Vec<int64,2> >, // 16
        0, 0, 0, 0, 0, 0, 0,
        transpose_<Vec<int64,3> >, // 24
        0, 0, 0, 0, 0, 0, 0,
        transpose_<Vec<int64,4> > // 32
    };

    size_t esz = src.elemSize();
    CV_Assert( esz <= (size_t)32 );

    if( dst.data == src.data && dst.cols == dst.rows )
    {
        TransposeInplaceFunc func = itab[esz];
        CV_Assert( func != 0 );
        func( dst );
    }
    else
    {
        dst.create( src.cols, src.rows, src.type() );
        TransposeFunc func = tab[esz];
        CV_Assert( func != 0 );
        func( src, dst );
    }
}


void completeSymm( Mat& matrix, bool LtoR )
{
    int i, j, nrows = matrix.rows, type = matrix.type();
    int j0 = 0, j1 = nrows;
    CV_Assert( matrix.rows == matrix.cols );

    if( type == CV_32FC1 || type == CV_32SC1 )
    {
        int* data = (int*)matrix.data;
        size_t step = matrix.step/sizeof(data[0]);
        for( i = 0; i < nrows; i++ )
        {
            if( !LtoR ) j1 = i; else j0 = i+1;
            for( j = j0; j < j1; j++ )
                data[i*step + j] = data[j*step + i];
        }
    }
    else if( type == CV_64FC1 )
    {
        double* data = (double*)matrix.data;
        size_t step = matrix.step/sizeof(data[0]);
        for( i = 0; i < nrows; i++ )
        {
            if( !LtoR ) j1 = i; else j0 = i+1;
            for( j = j0; j < j1; j++ )
                data[i*step + j] = data[j*step + i];
        }
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "" );
}

Mat Mat::cross(const Mat& m) const
{
    int t = type(), d = CV_MAT_DEPTH(t);
    CV_Assert( size() == m.size() && t == m.type() &&
        ((rows == 3 && cols == 1) || (cols*channels() == 3 && rows == 1)));
    Mat result(rows, cols, t);

    if( d == CV_32F )
    {
        const float *a = (const float*)data, *b = (const float*)m.data;
        float* c = (float*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }
    else if( d == CV_64F )
    {
        const double *a = (const double*)data, *b = (const double*)m.data;
        double* c = (double*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }

    return result;
}


/****************************************************************************************\
*                                Reduce Mat to vector                                 *
\****************************************************************************************/

template<typename T, typename ST, class Op> static void
reduceR_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    size.width *= srcmat.channels();
    AutoBuffer<WT> buffer(size.width);
    WT* buf = buffer;
    ST* dst = (ST*)dstmat.data;
    const T* src = (const T*)srcmat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    int i;
    Op op;

    for( i = 0; i < size.width; i++ )
        buf[i] = src[i];

    for( ; --size.height; )
    {
        src += srcstep;
        for( i = 0; i <= size.width - 4; i += 4 )
        {
            WT s0, s1;
            s0 = op(buf[i], (WT)src[i]);
            s1 = op(buf[i+1], (WT)src[i+1]);
            buf[i] = s0; buf[i+1] = s1;

            s0 = op(buf[i+2], (WT)src[i+2]);
            s1 = op(buf[i+3], (WT)src[i+3]);
            buf[i+2] = s0; buf[i+3] = s1;
        }

        for( ; i < size.width; i++ )
            buf[i] = op(buf[i], (WT)src[i]);
    }

    for( i = 0; i < size.width; i++ )
        dst[i] = (ST)buf[i];
}


template<typename T, typename ST, class Op> static void
reduceC_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    int i, k, cn = srcmat.channels();
    size.width *= cn;
    Op op;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        ST* dst = (ST*)(dstmat.data + dstmat.step*y);
        if( size.width == cn )
            for( k = 0; k < cn; k++ )
                dst[k] = src[k];
        else
        {
            for( k = 0; k < cn; k++ )
            {
                WT a0 = src[k], a1 = src[k+cn];
                for( i = 2*cn; i <= size.width - 4*cn; i += 4*cn )
                {
                    a0 = op(a0, (WT)src[i+k]);
                    a1 = op(a1, (WT)src[i+k+cn]);
                    a0 = op(a0, (WT)src[i+k+cn*2]);
                    a1 = op(a1, (WT)src[i+k+cn*3]);
                }

                for( ; i < size.width; i += cn )
                {
                    a0 = op(a0, (WT)src[i]);
                }
                a0 = op(a0, a1);
                dst[k] = (ST)a0;
            }
        }
    }
}

typedef void (*ReduceFunc)( const Mat& src, Mat& dst );

void reduce(const Mat& src, Mat& dst, int dim, int op, int dtype)
{
    int op0 = op;
    int stype = src.type(), sdepth = src.depth();
    if( dtype < 0 )
        dtype = stype;
    int ddepth = CV_MAT_DEPTH(dtype);

    dst.create(dim == 0 ? 1 : src.rows, dim == 0 ? src.cols : 1, dtype >= 0 ? dtype : stype);
    Mat temp = dst;
    
    CV_Assert( op == CV_REDUCE_SUM || op == CV_REDUCE_MAX ||
        op == CV_REDUCE_MIN || op == CV_REDUCE_AVG );
    CV_Assert( src.channels() == dst.channels() );

    if( op == CV_REDUCE_AVG )
    {
        op = CV_REDUCE_SUM;
        if( sdepth < CV_32S && ddepth < CV_32S )
            temp.create(dst.rows, dst.cols, CV_32SC(src.channels()));
    }

    ReduceFunc func = 0;
    if( dim == 0 )
    {
        if( op == CV_REDUCE_SUM )
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = reduceR_<uchar,int,OpAdd<int> >;
            if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceR_<uchar,float,OpAdd<int> >;
            if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceR_<uchar,double,OpAdd<int> >;
            if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceR_<ushort,float,OpAdd<float> >;
            if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceR_<ushort,double,OpAdd<double> >;
            if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceR_<short,float,OpAdd<float> >;
            if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceR_<short,double,OpAdd<double> >;
            if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceR_<float,float,OpAdd<float> >;
            if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceR_<float,double,OpAdd<double> >;
            if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceR_<double,double,OpAdd<double> >;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceR_<uchar, uchar, OpMax<uchar> >;
            if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceR_<float, float, OpMax<float> >;
            if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceR_<double, double, OpMax<double> >;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceR_<uchar, uchar, OpMin<uchar> >;
            if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceR_<float, float, OpMin<float> >;
            if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceR_<double, double, OpMin<double> >;
        }
    }
    else
    {
        if(op == CV_REDUCE_SUM)
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = reduceC_<uchar,int,OpAdd<int> >;
            if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceC_<uchar,float,OpAdd<int> >;
            if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceC_<uchar,double,OpAdd<int> >;
            if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceC_<ushort,float,OpAdd<float> >;
            if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceC_<ushort,double,OpAdd<double> >;
            if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceC_<short,float,OpAdd<float> >;
            if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceC_<short,double,OpAdd<double> >;
            if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceC_<float,float,OpAdd<float> >;
            if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceC_<float,double,OpAdd<double> >;
            if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceC_<double,double,OpAdd<double> >;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceC_<uchar, uchar, OpMax<uchar> >;
            if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceC_<float, float, OpMax<float> >;
            if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceC_<double, double, OpMax<double> >;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceC_<uchar, uchar, OpMin<uchar> >;
            if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceC_<float, float, OpMin<float> >;
            if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceC_<double, double, OpMin<double> >;
        }
    }

    if( !func )
        CV_Error( CV_StsUnsupportedFormat,
        "Unsupported combination of input and output array formats" );

    func( src, temp );

    if( op0 == CV_REDUCE_AVG )
        temp.convertTo(dst, dst.type(), 1./(dim == 0 ? src.rows : src.cols));
}


template<typename T> static void sort_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    T* bptr;
    int i, j, n, len;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool inplace = src.data == dst.data;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;
    
    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
    }
    bptr = (T*)buf;

    for( i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        if( sortRows )
        {
            T* dptr = (T*)(dst.data + dst.step*i);
            if( !inplace )
            {
                const T* sptr = (const T*)(src.data + src.step*i);
                for( j = 0; j < len; j++ )
                    dptr[j] = sptr[j];
            }
            ptr = dptr;
        }
        else
        {
            for( j = 0; j < len; j++ )
                ptr[j] = ((const T*)(src.data + src.step*j))[i];
        }
        std::sort( ptr, ptr + len, LessThan<T>() );
        if( sortDescending )
            for( j = 0; j < len/2; j++ )
                std::swap(ptr[j], ptr[len-1-j]);
        if( !sortRows )
            for( j = 0; j < len; j++ )
                ((T*)(dst.data + dst.step*j))[i] = ptr[j];
    }
}


template<typename T> static void sortIdx_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    AutoBuffer<int> ibuf;
    T* bptr;
    int* _iptr;
    int i, j, n, len;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;

    CV_Assert( src.data != dst.data );
    
    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
        ibuf.allocate(len);
    }
    bptr = (T*)buf;
    _iptr = (int*)ibuf;

    for( i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        int* iptr = _iptr;

        if( sortRows )
        {
            ptr = (T*)(src.data + src.step*i);
            iptr = (int*)(dst.data + dst.step*i);
        }
        else
        {
            for( j = 0; j < len; j++ )
                ptr[j] = ((const T*)(src.data + src.step*j))[i];
        }
        for( j = 0; j < len; j++ )
            iptr[j] = j;
        std::sort( iptr, iptr + len, LessThanIdx<T>(ptr) );
        if( sortDescending )
            for( j = 0; j < len/2; j++ )
                std::swap(iptr[j], iptr[len-1-j]);
        if( !sortRows )
            for( j = 0; j < len; j++ )
                ((int*)(dst.data + dst.step*j))[i] = iptr[j];
    }
}

typedef void (*SortFunc)(const Mat& src, Mat& dst, int flags);

void sort( const Mat& src, Mat& dst, int flags )
{
    static SortFunc tab[] =
    {
        sort_<uchar>, sort_<schar>, sort_<ushort>, sort_<short>,
        sort_<int>, sort_<float>, sort_<double>, 0
    };
    SortFunc func = tab[src.depth()];
    CV_Assert( src.channels() == 1 && func != 0 );
    dst.create( src.size(), src.type() );
    func( src, dst, flags );
}

void sortIdx( const Mat& src, Mat& dst, int flags )
{
    static SortFunc tab[] =
    {
        sortIdx_<uchar>, sortIdx_<schar>, sortIdx_<ushort>, sortIdx_<short>,
        sortIdx_<int>, sortIdx_<float>, sortIdx_<double>, 0
    };
    SortFunc func = tab[src.depth()];
    CV_Assert( src.channels() == 1 && func != 0 );
    if( dst.data == src.data )
        dst.release();
    dst.create( src.size(), CV_32S );
    func( src, dst, flags );
}

static void generateRandomCenter(const vector<Vec2f>& box, float* center, RNG& rng)
{
    size_t j, dims = box.size();
    float margin = 1.f/dims;
    for( j = 0; j < dims; j++ )
        center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
}


static inline float distance(const float* a, const float* b, int n, bool simd)
{
    int j = 0; float d = 0.f;
#if CV_SSE
    if( simd )
    {
        float CV_DECL_ALIGNED(16) buf[4];
        __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();

        for( ; j <= n - 8; j += 8 )
        {
            __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
            __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
            d0 = _mm_add_ps(d0, _mm_mul_ps(t0, t0));
            d1 = _mm_add_ps(d1, _mm_mul_ps(t1, t1));
        }
        _mm_store_ps(buf, _mm_add_ps(d0, d1));
        d = buf[0] + buf[1] + buf[2] + buf[3];
    }
    else
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            float t0 = a[j] - b[j], t1 = a[j+1] - b[j+1], t2 = a[j+2] - b[j+2], t3 = a[j+3] - b[j+3];
            d += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        }
    }

    for( ; j < n; j++ )
    {
        float t = a[j] - b[j];
        d += t*t;
    }
    return d;
}

/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
                              int K, RNG& rng, int trials)
{
    int i, j, k, dims = _data.cols, N = _data.rows;
    const float* data = _data.ptr<float>(0);
    int step = (int)(_data.step/sizeof(data[0]));
    vector<int> _centers(K);
    int* centers = &_centers[0];
    vector<float> _dist(N*3);
    float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
    double sum0 = 0;
    bool simd = checkHardwareSupport(CV_CPU_SSE);

    centers[0] = (unsigned)rng % N;

    for( i = 0; i < N; i++ )
    {
        dist[i] = distance(data + step*i, data + step*centers[0], dims, simd);
        sum0 += dist[i];
    }
    
    for( k = 1; k < K; k++ )
    {
        double bestSum = DBL_MAX;
        int bestCenter = -1;

        for( j = 0; j < trials; j++ )
        {
            double p = (double)rng*sum0, s = 0;
            for( i = 0; i < N-1; i++ )
                if( (p -= dist[i]) <= 0 )
                    break;
            int ci = i;
            for( i = 0; i < N; i++ )
            {
                tdist2[i] = std::min(distance(data + step*i, data + step*ci, dims, simd), dist[i]);
                s += tdist2[i];
            }
            
            if( s < bestSum )
            {
                bestSum = s;
                bestCenter = ci;
                std::swap(tdist, tdist2);
            }
        }
        centers[k] = bestCenter;
        sum0 = bestSum;
        std::swap(dist, tdist);
    }

    for( k = 0; k < K; k++ )
    {
        const float* src = data + step*centers[k];
        float* dst = _out_centers.ptr<float>(k);
        for( j = 0; j < dims; j++ )
            dst[j] = src[j];
    }
}

double kmeans( const Mat& data, int K, Mat& best_labels,
               TermCriteria criteria, int attempts,
               int flags, Mat* _centers )
{
    const int SPP_TRIALS = 3;
    int N = data.rows > 1 ? data.rows : data.cols;
    int dims = (data.rows > 1 ? data.cols : 1)*data.channels();
    int type = data.depth();
    bool simd = checkHardwareSupport(CV_CPU_SSE);

    attempts = std::max(attempts, 1);
    CV_Assert( type == CV_32F && K > 0 );

    Mat _labels;
    if( flags & CV_KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                  best_labels.cols*best_labels.rows == N &&
                  best_labels.type() == CV_32S &&
                  best_labels.isContinuous());
        best_labels.copyTo(_labels);
    }
    else
    {
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
             best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S);
        _labels.create(best_labels.size(), best_labels.type());
    }
    int* labels = _labels.ptr<int>();

    Mat centers(K, dims, type), old_centers(K, dims, type);
    vector<int> counters(K);
    vector<Vec2f> _box(dims);
    Vec2f* box = &_box[0];

    double best_compactness = DBL_MAX, compactness = 0;
    RNG& rng = theRNG();
    int a, iter, i, j, k;

    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

    const float* sample = data.ptr<float>(0);
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);

    for( i = 1; i < N; i++ )
    {
        sample = data.ptr<float>(i);
        for( j = 0; j < dims; j++ )
        {
            float v = sample[j];
            box[j][0] = std::min(box[j][0], v);
            box[j][1] = std::max(box[j][1], v);
        }
    }

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = DBL_MAX;
        for( iter = 0; iter < criteria.maxCount && max_center_shift > criteria.epsilon; iter++ )
        {
            swap(centers, old_centers);

            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
            {
                if( flags & KMEANS_PP_CENTERS )
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                else
                {
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }
            
                // compute centers
                centers = Scalar(0);
                for( k = 0; k < K; k++ )
                    counters[k] = 0;

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<float>(i);
                    k = labels[i];
                    float* center = centers.ptr<float>(k);
                    for( j = 0; j <= dims - 4; j += 4 )
                    {
                        float t0 = center[j] + sample[j];
                        float t1 = center[j+1] + sample[j+1];

                        center[j] = t0;
                        center[j+1] = t1;

                        t0 = center[j+2] + sample[j+2];
                        t1 = center[j+3] + sample[j+3];

                        center[j+2] = t0;
                        center[j+3] = t1;
                    }
                    for( ; j < dims; j++ )
                        center[j] += sample[j];
                    counters[k]++;
                }

                if( iter > 0 )
                    max_center_shift = 0;

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    if( counters[k] != 0 )
                    {
                        float scale = 1.f/counters[k];
                        for( j = 0; j < dims; j++ )
                            center[j] *= scale;
                    }
                    else
                        generateRandomCenter(_box, center, rng);
                    
                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        for( j = 0; j < dims; j++ )
                        {
                            double t = center[j] - old_center[j];
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }

            // assign labels
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                sample = data.ptr<float>(i);
                int k_best = 0;
                double min_dist = DBL_MAX;

                for( k = 0; k < K; k++ )
                {
                    const float* center = centers.ptr<float>(k);
                    double dist = distance(sample, center, dims, simd);

                    if( min_dist > dist )
                    {
                        min_dist = dist;
                        k_best = k;
                    }
                }

                compactness += min_dist;
                labels[i] = k_best;
            }
        }

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            if( _centers )
                centers.copyTo(*_centers);
            _labels.copyTo(best_labels);
        }
    }

    return best_compactness;
}

}


CV_IMPL void cvSetIdentity( CvArr* arr, CvScalar value )
{
    cv::Mat m = cv::cvarrToMat(arr);
    cv::setIdentity(m, value);
}


CV_IMPL CvScalar cvTrace( const CvArr* arr )
{
    return cv::trace(cv::cvarrToMat(arr));
}


CV_IMPL void cvTranspose( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.rows == dst.cols && src.cols == dst.rows && src.type() == dst.type() );
    transpose( src, dst );
}


CV_IMPL void cvCompleteSymm( CvMat* matrix, int LtoR )
{
    cv::Mat m(matrix);
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
    int ok = 0;
    
    CvMat stub, *mat = (CvMat*)arr;
    double delta;
    int type, step;
    double val = start;
    int i, j;
    int rows, cols;
    
    if( !CV_IS_MAT(mat) )
        mat = cvGetMat( mat, &stub);

    rows = mat->rows;
    cols = mat->cols;
    type = CV_MAT_TYPE(mat->type);
    delta = (end-start)/(rows*cols);

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
            for( i = 0; i < rows; i++, idata += step )
                for( j = 0; j < cols; j++, ival += idelta )
                    idata[j] = ival;
        }
        else
        {
            for( i = 0; i < rows; i++, idata += step )
                for( j = 0; j < cols; j++, val += delta )
                    idata[j] = cvRound(val);
        }
    }
    else if( type == CV_32FC1 )
    {
        float* fdata = mat->data.fl;
        for( i = 0; i < rows; i++, fdata += step )
            for( j = 0; j < cols; j++, val += delta )
                fdata[j] = (float)val;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "The function only supports 32sC1 and 32fC1 datatypes" );

    ok = 1;
    return ok ? arr : 0;
}


CV_IMPL void
cvSort( const CvArr* _src, CvArr* _dst, CvArr* _idx, int flags )
{
    cv::Mat src = cv::cvarrToMat(_src), dst, idx;
    
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
        centers = cv::cvarrToMat(_centers);
    CV_Assert( labels.isContinuous() && labels.type() == CV_32S &&
        (labels.cols == 1 || labels.rows == 1) &&
        labels.cols + labels.rows - 1 == data.rows );
    double compactness = cv::kmeans(data, cluster_count, labels, termcrit, attempts,
                                    flags, _centers ? &centers : 0 );
    if( _compactness )
        *_compactness = compactness;
    return 1;
}

///////////////////////////// n-dimensional matrices ////////////////////////////

namespace cv
{

//////////////////////////////// MatND ///////////////////////////////////

MatND::MatND(const MatND& m, const Range* ranges)
 : flags(MAGIC_VAL), dims(0), refcount(0), data(0), datastart(0), dataend(0)
{
    int i, j, d = m.dims;

    CV_Assert(ranges);
    for( i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        CV_Assert( r == Range::all() ||
            (0 <= r.start && r.start < r.end && r.end <= m.size[i]) );
    }
    *this = m;
    for( i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        if( r != Range::all() )
        {
            size[i] = r.end - r.start;
            data += r.start*step[i];
        }
    }
    
    for( i = 0; i < d; i++ )
    {
        if( size[i] != 1 )
            break;
    }

    CV_Assert( step[d-1] == elemSize() );
    for( j = d-1; j > i; j-- )
    {
        if( step[j]*size[j] < step[j-1] )
            break;
    }
    flags = (flags & ~CONTINUOUS_FLAG) | (j <= i ? CONTINUOUS_FLAG : 0);
}

void MatND::create(int d, const int* _sizes, int _type)
{
    CV_Assert(d > 0 && _sizes);
    int i;
    _type = CV_MAT_TYPE(_type);
    if( data && d == dims && _type == type() )
    {
        for( i = 0; i < d; i++ )
            if( size[i] != _sizes[i] )
                break;
        if( i == d )
            return;
    }
    
    release();
    
    flags = (_type & CV_MAT_TYPE_MASK) | MAGIC_VAL | CONTINUOUS_FLAG;
    size_t total = elemSize();
    int64 total1;
    
    for( i = d-1; i >= 0; i-- )
    {
        int sz = _sizes[i];
        size[i] = sz;
        step[i] = total;
        total1 = (int64)total*sz;
        CV_Assert( sz > 0 );
        if( (uint64)total1 != (size_t)total1 )
            CV_Error( CV_StsOutOfRange, "The total matrix size does not fit to \"size_t\" type" );
        total = (size_t)total1;
    }
    total = alignSize(total, (int)sizeof(*refcount));
    data = datastart = (uchar*)fastMalloc(total + (int)sizeof(*refcount));
    dataend = datastart + step[0]*size[0];
    refcount = (int*)(data + total);
    *refcount = 1;
    dims = d;
}

void MatND::copyTo( MatND& m ) const
{
    m.create( dims, size, type() );
    NAryMatNDIterator it(*this, m);

    for( int i = 0; i < it.nplanes; i++, ++it )
        it.planes[0].copyTo(it.planes[1]); 
}

void MatND::copyTo( MatND& m, const MatND& mask ) const
{
    m.create( dims, size, type() );
    NAryMatNDIterator it(*this, m, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        it.planes[0].copyTo(it.planes[1], it.planes[2]); 
}

void MatND::convertTo( MatND& m, int rtype, double alpha, double beta ) const
{
    rtype = rtype < 0 ? type() : CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());
    m.create( dims, size, rtype );
    NAryMatNDIterator it(*this, m);

    for( int i = 0; i < it.nplanes; i++, ++it )
        it.planes[0].convertTo(it.planes[1], rtype, alpha, beta);
}

MatND& MatND::operator = (const Scalar& s)
{
    NAryMatNDIterator it(*this);
    for( int i = 0; i < it.nplanes; i++, ++it )
        it.planes[0] = s;

    return *this;
}

MatND& MatND::setTo(const Scalar& s, const MatND& mask)
{
    NAryMatNDIterator it(*this, mask);
    for( int i = 0; i < it.nplanes; i++, ++it )
        it.planes[0].setTo(s, it.planes[1]);

    return *this;
}

MatND MatND::reshape(int, int, const int*) const
{
    CV_Error(CV_StsNotImplemented, "");
    // TBD
    return MatND();
}

MatND::operator Mat() const
{
    int i, d = dims, d1, rows, cols;
    size_t _step = Mat::AUTO_STEP;
    
    if( d <= 2 )
    {
        rows = size[0];
        cols = d == 2 ? size[1] : 1;
        if( d == 2 )
            _step = step[0];
    }
    else
    {
        rows = 1;
        cols = size[d-1];

        for( d1 = 0; d1 < d; d1++ )
            if( size[d1] > 1 )
                break;

        for( i = d-1; i > d1; i-- )
        {
            int64 cols1 = (int64)cols*size[i-1];
            if( cols1 != (int)cols1 || size[i]*step[i] != step[i-1] )
                break;
            cols = (int)cols1;
        }

        if( i > d1 )
        {
            --i;
            _step = step[i];
            rows = size[i];
            for( ; i > d1; i-- )
            {
                int64 rows1 = (int64)rows*size[i-1];
                if( rows1 != (int)rows1 || size[i]*step[i] != step[i-1] )
                    break;
                rows = (int)rows1;
            }

            if( i > d1 )
                CV_Error( CV_StsBadArg,
                "The nD matrix can not be represented as 2D matrix due "
                "to its layout in memory; you may use (Mat)the_matnd.clone() instead" );
        }
    }

    Mat m(rows, cols, type(), data, _step);
    m.datastart = datastart;
    m.dataend = dataend;
    m.refcount = refcount;
    m.addref();
    return m;
}

MatND::operator CvMatND() const
{
    CvMatND mat;
    cvInitMatNDHeader( &mat, dims, size, type(), data );
    int i, d = dims;
    for( i = 0; i < d; i++ )
        mat.dim[i].step = (int)step[i];
    mat.type |= flags & CONTINUOUS_FLAG;
    return mat;
}

NAryMatNDIterator::NAryMatNDIterator(const MatND** _arrays, size_t count)
{
    init(_arrays, count);
}

NAryMatNDIterator::NAryMatNDIterator(const MatND* _arrays, size_t count)
{
    AutoBuffer<const MatND*, 32> buf(count);
    for( size_t i = 0; i < count; i++ )
        buf[i] = _arrays + i;
    init(buf, count);
}

    
NAryMatNDIterator::NAryMatNDIterator(const MatND& m1)
{
    const MatND* mm[] = {&m1};
    init(mm, 1);
}

NAryMatNDIterator::NAryMatNDIterator(const MatND& m1, const MatND& m2)
{
    const MatND* mm[] = {&m1, &m2};
    init(mm, 2);
}

NAryMatNDIterator::NAryMatNDIterator(const MatND& m1, const MatND& m2, const MatND& m3)
{
    const MatND* mm[] = {&m1, &m2, &m3};
    init(mm, 3);
}

NAryMatNDIterator::NAryMatNDIterator(const MatND& m1, const MatND& m2,
                                     const MatND& m3, const MatND& m4)
{
    const MatND* mm[] = {&m1, &m2, &m3, &m4};
    init(mm, 4);
}
    
NAryMatNDIterator::NAryMatNDIterator(const MatND& m1, const MatND& m2,
                                     const MatND& m3, const MatND& m4,
                                     const MatND& m5)
{
    const MatND* mm[] = {&m1, &m2, &m3, &m4, &m5};
    init(mm, 5);
}
    
NAryMatNDIterator::NAryMatNDIterator(const MatND& m1, const MatND& m2,
                                     const MatND& m3, const MatND& m4,
                                     const MatND& m5, const MatND& m6)
{
    const MatND* mm[] = {&m1, &m2, &m3, &m4, &m5, &m6};
    init(mm, 6);
}    
    
void NAryMatNDIterator::init(const MatND** _arrays, size_t count)
{
    CV_Assert( _arrays && count > 0 );
    arrays.resize(count);
    int i, j, d1=0, i0 = -1, d = -1, n = (int)count;

    iterdepth = 0;

    for( i = 0; i < n; i++ )
    {
        if( !_arrays[i] || !_arrays[i]->data )
        {
            arrays[i] = MatND();
            continue;
        }
        const MatND& A = arrays[i] = *_arrays[i];
        
        if( i0 < 0 )
        {
            i0 = i;
            d = A.dims;
            
            // find the first dimensionality which is different from 1;
            // in any of the arrays the first "d1" steps do not affect the continuity
            for( d1 = 0; d1 < d; d1++ )
                if( A.size[d1] > 1 )
                    break;
        }
        else
        {
            CV_Assert( A.dims == d );
            for( j = 0; j < d; j++ )
                CV_Assert( A.size[j] == arrays[i0].size[j] );
        }

        if( !A.isContinuous() )
        {
            CV_Assert( A.step[d-1] == A.elemSize() );
            for( j = d-1; j > d1; j-- )
                if( A.step[j]*A.size[j] < A.step[j-1] )
                    break;
            iterdepth = std::max(iterdepth, j);
        }
    }

    if( i0 < 0 )
        CV_Error( CV_StsBadArg, "All the input arrays are empty" );

    int total = arrays[i0].size[d-1];
    for( j = d-1; j > iterdepth; j-- )
    {
        int64 total1 = (int64)total*arrays[i0].size[j-1];
        if( total1 != (int)total1 )
            break;
        total = (int)total1;
    }

    iterdepth = j;
    if( iterdepth == d1 )
        iterdepth = 0;

    planes.resize(n);
    for( i = 0; i < n; i++ )
    {
        if( !arrays[i].data )
        {
            planes[i] = Mat();
            continue;
        }
        planes[i] = Mat( 1, total, arrays[i].type(), arrays[i].data );
        planes[i].datastart = arrays[i].datastart;
        planes[i].dataend = arrays[i].dataend;
        planes[i].refcount = arrays[i].refcount;
        planes[i].addref();
    }

    idx = 0;
    nplanes = 1;
    for( j = iterdepth-1; j >= 0; j-- )
        nplanes *= arrays[i0].size[j];
}


NAryMatNDIterator& NAryMatNDIterator::operator ++()
{
    if( idx >= nplanes-1 )
        return *this;
    ++idx;

    for( size_t i = 0; i < arrays.size(); i++ )
    {
        const MatND& A = arrays[i];
        Mat& M = planes[i];
        if( !A.data )
            continue;
        int _idx = idx;
        uchar* data = A.data;
        for( int j = iterdepth-1; j >= 0 && _idx > 0; j-- )
        {
            int szi = A.size[j], t = _idx/szi;
            data += (_idx - t * szi)*A.step[j];
            _idx = t;
        }
        M.data = data;
    }
    
    return *this;
}

NAryMatNDIterator NAryMatNDIterator::operator ++(int)
{
    NAryMatNDIterator it = *this;
    ++*this;
    return it;
}

void add(const MatND& a, const MatND& b, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        add( it.planes[0], it.planes[1], it.planes[2], it.planes[3] ); 
}

void subtract(const MatND& a, const MatND& b, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        subtract( it.planes[0], it.planes[1], it.planes[2], it.planes[3] ); 
}

void add(const MatND& a, const MatND& b, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        add( it.planes[0], it.planes[1], it.planes[2] ); 
}


void subtract(const MatND& a, const MatND& b, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        subtract( it.planes[0], it.planes[1], it.planes[2] ); 
}

void add(const MatND& a, const Scalar& s, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        add( it.planes[0], s, it.planes[1], it.planes[2] ); 
}

void subtract(const Scalar& s, const MatND& a, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        subtract( s, it.planes[0], it.planes[1], it.planes[2] ); 
}

void multiply(const MatND& a, const MatND& b, MatND& c, double scale)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        multiply( it.planes[0], it.planes[1], it.planes[2], scale ); 
}

void divide(const MatND& a, const MatND& b, MatND& c, double scale)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        divide( it.planes[0], it.planes[1], it.planes[2], scale ); 
}

void divide(double scale, const MatND& b, MatND& c)
{
    c.create(b.dims, b.size, b.type());
    NAryMatNDIterator it(b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        divide( scale, it.planes[0], it.planes[1] ); 
}

void scaleAdd(const MatND& a, double alpha, const MatND& b, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        scaleAdd( it.planes[0], alpha, it.planes[1], it.planes[2] ); 
}

void addWeighted(const MatND& a, double alpha, const MatND& b,
                 double beta, double gamma, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        addWeighted( it.planes[0], alpha, it.planes[1], beta, gamma, it.planes[2] );
}

Scalar sum(const MatND& m)
{
    NAryMatNDIterator it(m);
    Scalar s;

    for( int i = 0; i < it.nplanes; i++, ++it )
        s += sum(it.planes[0]);
    return s;
}

int countNonZero( const MatND& m )
{
    NAryMatNDIterator it(m);
    int nz = 0;

    for( int i = 0; i < it.nplanes; i++, ++it )
        nz += countNonZero(it.planes[0]);
    return nz;
}

Scalar mean(const MatND& m)
{
    NAryMatNDIterator it(m);
    double total = 1;
    for( int i = 0; i < m.dims; i++ )
        total *= m.size[i];
    return sum(m)*(1./total);
}

Scalar mean(const MatND& m, const MatND& mask)
{
    if( !mask.data )
        return mean(m);
    NAryMatNDIterator it(m, mask);
    double total = 0;
    Scalar s;
    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        int n = countNonZero(it.planes[1]);
        s += mean(it.planes[0], it.planes[1])*(double)n;
        total += n;
    }
    return s *= 1./std::max(total, 1.);
}

void meanStdDev(const MatND& m, Scalar& mean, Scalar& stddev, const MatND& mask)
{
    NAryMatNDIterator it(m, mask);
    double total = 0;
    Scalar s, sq;
    int k, cn = m.channels();

    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        Scalar _mean, _stddev;
        meanStdDev(it.planes[0], _mean, _stddev, it.planes[1]);
        double nz = mask.data ? countNonZero(it.planes[1]) :
            (double)it.planes[0].rows*it.planes[0].cols;
        for( k = 0; k < cn; k++ )
        {
            s[k] += _mean[k]*nz;
            sq[k] += (_stddev[k]*_stddev[k] + _mean[k]*_mean[k])*nz;
        }
        total += nz;
    }

    mean = stddev = Scalar();
    total = 1./std::max(total, 1.);
    for( k = 0; k < cn; k++ )
    {
        mean[k] = s[k]*total;
        stddev[k] = std::sqrt(std::max(sq[k]*total - mean[k]*mean[k], 0.));
    }
}

double norm(const MatND& a, int normType, const MatND& mask)
{
    NAryMatNDIterator it(a, mask);
    double total = 0;

    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        double n = norm(it.planes[0], normType, it.planes[1]);
        if( normType == NORM_INF )
            total = std::max(total, n);
        else if( normType == NORM_L1 )
            total += n;
        else
            total += n*n;
    }

    return normType != NORM_L2 ? total : std::sqrt(total);
}

double norm(const MatND& a, const MatND& b,
            int normType, const MatND& mask)
{
    bool isRelative = (normType & NORM_RELATIVE) != 0;
    normType &= 7;

    NAryMatNDIterator it(a, b, mask);
    double num = 0, denom = 0;

    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        double n = norm(it.planes[0], it.planes[1], normType, it.planes[2]);
        double d = !isRelative ? 0 : norm(it.planes[1], normType, it.planes[2]);
        if( normType == NORM_INF )
        {
            num = std::max(num, n);
            denom = std::max(denom, d);
        }
        else if( normType == NORM_L1 )
        {
            num += n;
            denom += d;
        }
        else
        {
            num += n*n;
            denom += d*d;
        }
    }

    if( normType == NORM_L2 )
    {
        num = std::sqrt(num);
        denom = std::sqrt(denom);
    }

    return !isRelative ? num : num/std::max(denom,DBL_EPSILON);
}

void normalize( const MatND& src, MatND& dst, double a, double b,
                int norm_type, int rtype, const MatND& mask )
{
    double scale = 1, shift = 0;
    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = std::min( a, b ), dmax = std::max( a, b );
        minMaxLoc( src, &smin, &smax, 0, 0, mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
        shift = dmin - smin*scale;
    }
    else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( src, norm_type, mask );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
        shift = 0;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );
    
    if( !mask.data )
        src.convertTo( dst, rtype, scale, shift );
    else
    {
        MatND temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( dst, mask );
    }
}

static void ofs2idx(const MatND& a, size_t ofs, int* idx)
{
    int i, d = a.dims;
    for( i = 0; i < d; i++ )
    {
        idx[i] = (int)(ofs / a.step[i]);
        ofs %= a.step[i];
    }
}
    
    
void minMaxLoc(const MatND& a, double* minVal,
               double* maxVal, int* minLoc, int* maxLoc,
               const MatND& mask)
{
    NAryMatNDIterator it(a, mask);
    double minval = DBL_MAX, maxval = -DBL_MAX;
    size_t minofs = 0, maxofs = 0, esz = a.elemSize();
    
    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        double val0 = 0, val1 = 0;
        Point pt0, pt1;
        minMaxLoc( it.planes[0], &val0, &val1, &pt0, &pt1, it.planes[1] );
        if( val0 < minval )
        {
            minval = val0;
            minofs = (it.planes[0].data - a.data) + pt0.x*esz;
        }
        if( val1 > maxval )
        {
            maxval = val1;
            maxofs = (it.planes[0].data - a.data) + pt1.x*esz;
        }
    }

    if( minVal )
        *minVal = minval;
    if( maxVal )
        *maxVal = maxval;
    if( minLoc )
        ofs2idx(a, minofs, minLoc);
    if( maxLoc )
        ofs2idx(a, maxofs, maxLoc);
}

void merge(const MatND* mv, size_t n, MatND& dst)
{
    size_t k;
    CV_Assert( n > 0 );
    vector<MatND> v(n + 1);
    int total_cn = 0;
    for( k = 0; k < n; k++ )
    {
        total_cn += mv[k].channels();
        v[k] = mv[k];
    }
    dst.create( mv[0].dims, mv[0].size, CV_MAKETYPE(mv[0].depth(), total_cn) );
    v[n] = dst;
    NAryMatNDIterator it(&v[0], v.size());

    for( int i = 0; i < it.nplanes; i++, ++it )
        merge( &it.planes[0], n, it.planes[n] );
}

void split(const MatND& m, MatND* mv)
{
    size_t k, n = m.channels();
    CV_Assert( n > 0 );
    vector<MatND> v(n + 1);
    for( k = 0; k < n; k++ )
    {
        mv[k].create( m.dims, m.size, CV_MAKETYPE(m.depth(), 1) );
        v[k] = mv[k];
    }
    v[n] = m;
    NAryMatNDIterator it(&v[0], v.size());

    for( int i = 0; i < it.nplanes; i++, ++it )
        split( it.planes[n], &it.planes[0] );
}

void mixChannels(const MatND* src, int nsrcs, MatND* dst, int ndsts,
                 const int* fromTo, size_t npairs)
{
    size_t k, m = nsrcs, n = ndsts;
    CV_Assert( n > 0 && m > 0 );
    vector<MatND> v(m + n);
    for( k = 0; k < m; k++ )
        v[k] = src[k];
    for( k = 0; k < n; k++ )
        v[m + k] = dst[k];
    NAryMatNDIterator it(&v[0], v.size());

    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        Mat* pptr = &it.planes[0];
        mixChannels( pptr, m, pptr + m, n, fromTo, npairs );
    }
}

void bitwise_and(const MatND& a, const MatND& b, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        bitwise_and( it.planes[0], it.planes[1], it.planes[2], it.planes[3] ); 
}

void bitwise_or(const MatND& a, const MatND& b, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        bitwise_or( it.planes[0], it.planes[1], it.planes[2], it.planes[3] ); 
}

void bitwise_xor(const MatND& a, const MatND& b, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        bitwise_xor( it.planes[0], it.planes[1], it.planes[2], it.planes[3] ); 
}

void bitwise_and(const MatND& a, const Scalar& s, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        bitwise_and( it.planes[0], s, it.planes[1], it.planes[2] ); 
}

void bitwise_or(const MatND& a, const Scalar& s, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        bitwise_or( it.planes[0], s, it.planes[1], it.planes[2] ); 
}

void bitwise_xor(const MatND& a, const Scalar& s, MatND& c, const MatND& mask)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c, mask);

    for( int i = 0; i < it.nplanes; i++, ++it )
        bitwise_xor( it.planes[0], s, it.planes[1], it.planes[2] ); 
}

void bitwise_not(const MatND& a, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        bitwise_not( it.planes[0], it.planes[1] ); 
}

void absdiff(const MatND& a, const MatND& b, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        absdiff( it.planes[0], it.planes[1], it.planes[2] ); 
}

void absdiff(const MatND& a, const Scalar& s, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        absdiff( it.planes[0], s, it.planes[1] ); 
}

void inRange(const MatND& src, const MatND& lowerb,
             const MatND& upperb, MatND& dst)
{
    dst.create(src.dims, src.size, CV_8UC1);
    NAryMatNDIterator it(src, lowerb, upperb, dst);

    for( int i = 0; i < it.nplanes; i++, ++it )
        inRange( it.planes[0], it.planes[1], it.planes[2], it.planes[3] ); 
}

void inRange(const MatND& src, const Scalar& lowerb,
             const Scalar& upperb, MatND& dst)
{
    dst.create(src.dims, src.size, CV_8UC1);
    NAryMatNDIterator it(src, dst);

    for( int i = 0; i < it.nplanes; i++, ++it )
        inRange( it.planes[0], lowerb, upperb, it.planes[1] ); 
}

void compare(const MatND& a, const MatND& b, MatND& c, int cmpop)
{
    c.create(a.dims, a.size, CV_8UC1);
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        compare( it.planes[0], it.planes[1], it.planes[2], cmpop ); 
}

void compare(const MatND& a, double s, MatND& c, int cmpop)
{
    c.create(a.dims, a.size, CV_8UC1);
    NAryMatNDIterator it(a, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        compare( it.planes[0], s, it.planes[1], cmpop ); 
}

void min(const MatND& a, const MatND& b, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        min( it.planes[0], it.planes[1], it.planes[2] );
}

void min(const MatND& a, double alpha, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        min( it.planes[0], alpha, it.planes[1] );
}

void max(const MatND& a, const MatND& b, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        max( it.planes[0], it.planes[1], it.planes[2] ); 
}

void max(const MatND& a, double alpha, MatND& c)
{
    c.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, c);

    for( int i = 0; i < it.nplanes; i++, ++it )
        max( it.planes[0], alpha, it.planes[1] );
}

void sqrt(const MatND& a, MatND& b)
{
    b.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b);

    for( int i = 0; i < it.nplanes; i++, ++it )
        sqrt( it.planes[0], it.planes[1] );
}

void pow(const MatND& a, double power, MatND& b)
{
    b.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b);

    for( int i = 0; i < it.nplanes; i++, ++it )
        pow( it.planes[0], power, it.planes[1] );
}

void exp(const MatND& a, MatND& b)
{
    b.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b);

    for( int i = 0; i < it.nplanes; i++, ++it )
        exp( it.planes[0], it.planes[1] );
}

void log(const MatND& a, MatND& b)
{
    b.create(a.dims, a.size, a.type());
    NAryMatNDIterator it(a, b);

    for( int i = 0; i < it.nplanes; i++, ++it )
        log( it.planes[0], it.planes[1] );
}

bool checkRange(const MatND& a, bool quiet, int*,
                double minVal, double maxVal)
{
    NAryMatNDIterator it(a);

    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        Point pt;
        if( !checkRange( it.planes[0], quiet, &pt, minVal, maxVal ))
        {
            // todo: set index properly
            return false;
        }
    }
    return true;
}


//////////////////////////////// SparseMat ////////////////////////////////

template<typename T1, typename T2> void
convertData_(const void* _from, void* _to, int cn)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]);
}

template<typename T1, typename T2> void
convertScaleData_(const void* _from, void* _to, int cn, double alpha, double beta)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from*alpha + beta);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]*alpha + beta);
}

ConvertData getConvertData(int fromType, int toType)
{
    static ConvertData tab[][8] =
    {{ convertData_<uchar, uchar>, convertData_<uchar, schar>,
      convertData_<uchar, ushort>, convertData_<uchar, short>,
      convertData_<uchar, int>, convertData_<uchar, float>,
      convertData_<uchar, double>, 0 },

    { convertData_<schar, uchar>, convertData_<schar, schar>,
      convertData_<schar, ushort>, convertData_<schar, short>,
      convertData_<schar, int>, convertData_<schar, float>,
      convertData_<schar, double>, 0 },

    { convertData_<ushort, uchar>, convertData_<ushort, schar>,
      convertData_<ushort, ushort>, convertData_<ushort, short>,
      convertData_<ushort, int>, convertData_<ushort, float>,
      convertData_<ushort, double>, 0 },

    { convertData_<short, uchar>, convertData_<short, schar>,
      convertData_<short, ushort>, convertData_<short, short>,
      convertData_<short, int>, convertData_<short, float>,
      convertData_<short, double>, 0 },

    { convertData_<int, uchar>, convertData_<int, schar>,
      convertData_<int, ushort>, convertData_<int, short>,
      convertData_<int, int>, convertData_<int, float>,
      convertData_<int, double>, 0 },

    { convertData_<float, uchar>, convertData_<float, schar>,
      convertData_<float, ushort>, convertData_<float, short>,
      convertData_<float, int>, convertData_<float, float>,
      convertData_<float, double>, 0 },

    { convertData_<double, uchar>, convertData_<double, schar>,
      convertData_<double, ushort>, convertData_<double, short>,
      convertData_<double, int>, convertData_<double, float>,
      convertData_<double, double>, 0 },

    { 0, 0, 0, 0, 0, 0, 0, 0 }};

    ConvertData func = tab[CV_MAT_DEPTH(fromType)][CV_MAT_DEPTH(toType)];
    CV_Assert( func != 0 );
    return func;
}

ConvertScaleData getConvertScaleData(int fromType, int toType)
{
    static ConvertScaleData tab[][8] =
    {{ convertScaleData_<uchar, uchar>, convertScaleData_<uchar, schar>,
      convertScaleData_<uchar, ushort>, convertScaleData_<uchar, short>,
      convertScaleData_<uchar, int>, convertScaleData_<uchar, float>,
      convertScaleData_<uchar, double>, 0 },

    { convertScaleData_<schar, uchar>, convertScaleData_<schar, schar>,
      convertScaleData_<schar, ushort>, convertScaleData_<schar, short>,
      convertScaleData_<schar, int>, convertScaleData_<schar, float>,
      convertScaleData_<schar, double>, 0 },

    { convertScaleData_<ushort, uchar>, convertScaleData_<ushort, schar>,
      convertScaleData_<ushort, ushort>, convertScaleData_<ushort, short>,
      convertScaleData_<ushort, int>, convertScaleData_<ushort, float>,
      convertScaleData_<ushort, double>, 0 },

    { convertScaleData_<short, uchar>, convertScaleData_<short, schar>,
      convertScaleData_<short, ushort>, convertScaleData_<short, short>,
      convertScaleData_<short, int>, convertScaleData_<short, float>,
      convertScaleData_<short, double>, 0 },

    { convertScaleData_<int, uchar>, convertScaleData_<int, schar>,
      convertScaleData_<int, ushort>, convertScaleData_<int, short>,
      convertScaleData_<int, int>, convertScaleData_<int, float>,
      convertScaleData_<int, double>, 0 },

    { convertScaleData_<float, uchar>, convertScaleData_<float, schar>,
      convertScaleData_<float, ushort>, convertScaleData_<float, short>,
      convertScaleData_<float, int>, convertScaleData_<float, float>,
      convertScaleData_<float, double>, 0 },

    { convertScaleData_<double, uchar>, convertScaleData_<double, schar>,
      convertScaleData_<double, ushort>, convertScaleData_<double, short>,
      convertScaleData_<double, int>, convertScaleData_<double, float>,
      convertScaleData_<double, double>, 0 },

    { 0, 0, 0, 0, 0, 0, 0, 0 }};

    ConvertScaleData func = tab[CV_MAT_DEPTH(fromType)][CV_MAT_DEPTH(toType)];
    CV_Assert( func != 0 );
    return func;
}

enum { HASH_SIZE0 = 8 };

static inline void copyElem(const uchar* from, uchar* to, size_t elemSize)
{
    size_t i;
    for( i = 0; (int)i <= (int)(elemSize - sizeof(int)); i += sizeof(int) )
        *(int*)(to + i) = *(const int*)(from + i);
    for( ; i < elemSize; i++ )
        to[i] = from[i];
}

static inline bool isZeroElem(const uchar* data, size_t elemSize)
{
    size_t i;
    for( i = 0; i <= elemSize - sizeof(int); i += sizeof(int) )
        if( *(int*)(data + i) != 0 )
            return false;
    for( ; i < elemSize; i++ )
        if( data[i] != 0 )
            return false;
    return true;
}

SparseMat::Hdr::Hdr( int _dims, const int* _sizes, int _type )
{
    refcount = 1;

    dims = _dims;
    valueOffset = (int)alignSize(sizeof(SparseMat::Node) +
        sizeof(int)*std::max(dims - CV_MAX_DIM, 0), CV_ELEM_SIZE1(_type));
    nodeSize = alignSize(valueOffset +
        CV_ELEM_SIZE(_type), (int)sizeof(size_t));
   
    int i;
    for( i = 0; i < dims; i++ )
        size[i] = _sizes[i];
    for( ; i < CV_MAX_DIM; i++ )
        size[i] = 0;
    clear();
}

void SparseMat::Hdr::clear()
{
    hashtab.clear();
    hashtab.resize(HASH_SIZE0);
    pool.clear();
    pool.resize(nodeSize);
    nodeCount = freeList = 0;
}


SparseMat::SparseMat(const Mat& m, bool try1d)
: flags(MAGIC_VAL), hdr(0)
{
    bool is1d = try1d && m.cols == 1;
    
    if( is1d )
    {
        int i, M = m.rows;
        const uchar* data = m.data;
        size_t step =  m.step, esz = m.elemSize();
        create( 1, &M, m.type() );
        for( i = 0; i < M; i++ )
        {
            const uchar* from = data + step*i;
            if( isZeroElem(from, esz) )
                continue;
            uchar* to = newNode(&i, hash(i));
            copyElem(from, to, esz);
        }
    }
    else
    {
        int i, j, size[] = {m.rows, m.cols};
        const uchar* data = m.data;
        size_t step =  m.step, esz = m.elemSize();
        create( 2, size, m.type() );
        for( i = 0; i < m.rows; i++ )
        {
            for( j = 0; j < m.cols; j++ )
            {
                const uchar* from = data + step*i + esz*j;
                if( isZeroElem(from, esz) )
                    continue;
                int idx[] = {i, j};
                uchar* to = newNode(idx, hash(i, j));
                copyElem(from, to, esz);
            }
        }
    }
}

SparseMat::SparseMat(const MatND& m)
: flags(MAGIC_VAL), hdr(0)
{
    create( m.dims, m.size, m.type() );

    int i, idx[CV_MAX_DIM] = {0}, d = m.dims, lastSize = m.size[d - 1];
    size_t esz = m.elemSize();
    uchar* ptr = m.data;

    for(;;)
    {
        for( i = 0; i < lastSize; i++, ptr += esz )
        {
            if( isZeroElem(ptr, esz) )
                continue;
            idx[d-1] = i;
            uchar* to = newNode(idx, hash(idx));
            copyElem( ptr, to, esz );
        }
        
        for( i = d - 2; i >= 0; i-- )
        {
            ptr += m.step[i] - m.size[i+1]*m.step[i+1];
            if( ++idx[i] < m.size[i] )
                break;
            idx[i] = 0;
        }
        if( i < 0 )
            break;
    }
}
                
SparseMat::SparseMat(const CvSparseMat* m)
: flags(MAGIC_VAL), hdr(0)
{
    CV_Assert(m);
    create( m->dims, &m->size[0], m->type );

    CvSparseMatIterator it;
    CvSparseNode* n = cvInitSparseMatIterator(m, &it);
    size_t esz = elemSize();

    for( ; n != 0; n = cvGetNextSparseNode(&it) )
    {
        const int* idx = CV_NODE_IDX(m, n);
        uchar* to = newNode(idx, hash(idx));
        copyElem((const uchar*)CV_NODE_VAL(m, n), to, esz);
    }
}

void SparseMat::create(int d, const int* _sizes, int _type)
{
    int i;
    CV_Assert( _sizes && 0 < d && d <= CV_MAX_DIM );
    for( i = 0; i < d; i++ )
        CV_Assert( _sizes[i] > 0 );
    _type = CV_MAT_TYPE(_type);
    if( hdr && _type == type() && hdr->dims == d && hdr->refcount == 1 )
    {
        for( i = 0; i < d; i++ )
            if( _sizes[i] != hdr->size[i] )
                break;
        if( i == d )
        {
            clear();
            return;
        }
    }
    release();
    flags = MAGIC_VAL | _type;
    hdr = new Hdr(d, _sizes, _type);
}

void SparseMat::copyTo( SparseMat& m ) const
{
    if( hdr == m.hdr )
        return;
    if( !hdr )
    {
        m.release();
        return;
    }
    m.create( hdr->dims, hdr->size, type() );
    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        uchar* to = m.newNode(n->idx, n->hashval);
        copyElem( from.ptr, to, esz );
    }
}

void SparseMat::copyTo( Mat& m ) const
{
    CV_Assert( hdr && hdr->dims <= 2 );
    m.create( hdr->size[0], hdr->dims == 2 ? hdr->size[1] : 1, type() );
    m = Scalar(0);

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = elemSize();

    if( hdr->dims == 2 )
    {
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.data + m.step*n->idx[0] + esz*n->idx[1];
            copyElem( from.ptr, to, esz );
        }
    }
    else
    {
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.data + esz*n->idx[0];
            copyElem( from.ptr, to, esz );
        }
    }
}

void SparseMat::copyTo( MatND& m ) const
{
    CV_Assert( hdr );
    m.create( dims(), hdr->size, type() );
    m = Scalar(0);

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        copyElem( from.ptr, m.ptr(n->idx), esz);
    }
}


void SparseMat::convertTo( SparseMat& m, int rtype, double alpha ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);
    if( hdr == m.hdr && rtype != type()  )
    {
        SparseMat temp;
        convertTo(temp, rtype, alpha);
        m = temp;
        return;
    }
    
    CV_Assert(hdr != 0);
    if( hdr != m.hdr )
        m.create( hdr->dims, hdr->size, rtype );
    
    SparseMatConstIterator from = begin();
    size_t i, N = nzcount();

    if( alpha == 1 )
    {
        ConvertData cvtfunc = getConvertData(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn ); 
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleData(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn, alpha, 0 ); 
        }
    }
}


void SparseMat::convertTo( Mat& m, int rtype, double alpha, double beta ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);
    
    CV_Assert( hdr && hdr->dims <= 2 );
    m.create( hdr->size[0], hdr->dims == 2 ? hdr->size[1] : 1, type() );
    m = Scalar(beta);

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = CV_ELEM_SIZE(rtype);

    if( alpha == 1 && beta == 0 )
    {
        ConvertData cvtfunc = getConvertData(type(), rtype);

        if( hdr->dims == 2 )
        {
            for( i = 0; i < N; i++, ++from )
            {
                const Node* n = from.node();
                uchar* to = m.data + m.step*n->idx[0] + esz*n->idx[1];
                cvtfunc( from.ptr, to, cn );
            }
        }
        else
        {
            for( i = 0; i < N; i++, ++from )
            {
                const Node* n = from.node();
                uchar* to = m.data + esz*n->idx[0];
                cvtfunc( from.ptr, to, cn );
            }
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleData(type(), rtype);

        if( hdr->dims == 2 )
        {
            for( i = 0; i < N; i++, ++from )
            {
                const Node* n = from.node();
                uchar* to = m.data + m.step*n->idx[0] + esz*n->idx[1];
                cvtfunc( from.ptr, to, cn, alpha, beta );
            }
        }
        else
        {
            for( i = 0; i < N; i++, ++from )
            {
                const Node* n = from.node();
                uchar* to = m.data + esz*n->idx[0];
                cvtfunc( from.ptr, to, cn, alpha, beta );
            }
        }
    }
}

void SparseMat::convertTo( MatND& m, int rtype, double alpha, double beta ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);
    
    CV_Assert( hdr );
    m.create( dims(), hdr->size, rtype );
    m = Scalar(beta);

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount();

    if( alpha == 1 && beta == 0 )
    {
        ConvertData cvtfunc = getConvertData(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn );
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleData(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn, alpha, beta );
        }
    }
}

void SparseMat::clear()
{
    if( hdr )
        hdr->clear();
}

SparseMat::operator CvSparseMat*() const
{
    if( !hdr )
        return 0;
    CvSparseMat* m = cvCreateSparseMat(hdr->dims, hdr->size, type());

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        uchar* to = cvPtrND(m, n->idx, 0, -2, 0);
        copyElem(from.ptr, to, esz);
    }
    return m;
}

uchar* SparseMat::ptr(int i0, int i1, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1 };
        return newNode( idx, h );
    }
    return 0;
}

uchar* SparseMat::ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1, i2 };
        return newNode( idx, h );
    }
    return 0;
}

uchar* SparseMat::ptr(const int* idx, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                return &value<uchar>(elem);
        }
        nidx = elem->next;
    }

    return createMissing ? newNode(idx, h) : 0;
}

void SparseMat::erase(int i0, int i1, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(int i0, int i1, int i2, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(const int* idx, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                break;
        }
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::resizeHashTab(size_t newsize)
{
    newsize = std::max(newsize, (size_t)8);
    if((newsize & (newsize-1)) != 0)
        newsize = (size_t)1 << cvCeil(std::log((double)newsize)/CV_LOG2);

    size_t i, hsize = hdr->hashtab.size();
    vector<size_t> _newh(newsize);
    size_t* newh = &_newh[0];
    for( i = 0; i < newsize; i++ )
        newh[i] = 0;
    uchar* pool = &hdr->pool[0];
    for( i = 0; i < hsize; i++ )
    {
        size_t nidx = hdr->hashtab[i];
        while( nidx )
        {
            Node* elem = (Node*)(pool + nidx);
            size_t next = elem->next;
            size_t newhidx = elem->hashval & (newsize - 1);
            elem->next = newh[newhidx];
            newh[newhidx] = nidx;
            nidx = next;
        }
    }
    hdr->hashtab = _newh;
}

uchar* SparseMat::newNode(const int* idx, size_t hashval)
{
    const int HASH_MAX_FILL_FACTOR=3;
    assert(hdr);
    size_t hsize = hdr->hashtab.size();
    if( ++hdr->nodeCount > hsize*HASH_MAX_FILL_FACTOR )
    {
        resizeHashTab(std::max(hsize*2, (size_t)8));
        hsize = hdr->hashtab.size();
    }
    
    if( !hdr->freeList )
    {
        size_t i, nsz = hdr->nodeSize, psize = hdr->pool.size(),
            newpsize = std::max(psize*2, 8*nsz);
        hdr->pool.resize(newpsize);
        uchar* pool = &hdr->pool[0];
        hdr->freeList = std::max(psize, nsz);
        for( i = hdr->freeList; i < newpsize - nsz; i += nsz )
            ((Node*)(pool + i))->next = i + nsz;
        ((Node*)(pool + i))->next = 0;
    }
    size_t nidx = hdr->freeList;
    Node* elem = (Node*)&hdr->pool[nidx];
    hdr->freeList = elem->next;
    elem->hashval = hashval;
    size_t hidx = hashval & (hsize - 1);
    elem->next = hdr->hashtab[hidx];
    hdr->hashtab[hidx] = nidx;

    int i, d = hdr->dims;
    for( i = 0; i < d; i++ )
        elem->idx[i] = idx[i];
    size_t esz = elemSize();
    uchar* p = &value<uchar>(elem);
    if( esz == sizeof(float) )
        *((float*)p) = 0.f;
    else if( esz == sizeof(double) )
        *((double*)p) = 0.;
    else
        memset(p, 0, esz);
    
    return p;
}


void SparseMat::removeNode(size_t hidx, size_t nidx, size_t previdx)
{
    Node* n = node(nidx);
    if( previdx )
    {
        Node* prev = node(previdx);
        prev->next = n->next;
    }
    else
        hdr->hashtab[hidx] = n->next;
    n->next = hdr->freeList;
    hdr->freeList = nidx;
    --hdr->nodeCount;
}


SparseMatConstIterator::SparseMatConstIterator(const SparseMat* _m)
: m((SparseMat*)_m), hashidx(0), ptr(0)
{
    if(!_m || !_m->hdr)
        return;
    SparseMat::Hdr& hdr = *m->hdr;
    const vector<size_t>& htab = hdr.hashtab;
    size_t i, hsize = htab.size();
    for( i = 0; i < hsize; i++ )
    {
        size_t nidx = htab[i];
        if( nidx )
        {
            hashidx = i;
            ptr = &hdr.pool[nidx] + hdr.valueOffset;
            return;
        }
    }
}

SparseMatConstIterator& SparseMatConstIterator::operator ++()
{
    if( !ptr || !m || !m->hdr )
        return *this;
    SparseMat::Hdr& hdr = *m->hdr;
    size_t next = ((const SparseMat::Node*)(ptr - hdr.valueOffset))->next;
    if( next )
    {
        ptr = &hdr.pool[next] + hdr.valueOffset;
        return *this;
    }
    size_t i = hashidx + 1, sz = hdr.hashtab.size();
    for( ; i < sz; i++ )
    {
        size_t nidx = hdr.hashtab[i];
        if( nidx )
        {
            hashidx = i;
            ptr = &hdr.pool[nidx] + hdr.valueOffset;
            return *this;
        }
    }
    hashidx = sz;
    ptr = 0;
    return *this;
}


double norm( const SparseMat& src, int normType )
{
    SparseMatConstIterator it = src.begin();
    
    size_t i, N = src.nzcount();
    normType &= NORM_TYPE_MASK;
    int type = src.type();
    double result = 0;
    
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );
    
    if( type == CV_32F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
                result = std::max(result, std::abs((double)*(const float*)it.ptr));
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
                result += std::abs(*(const float*)it.ptr);
        else
            for( i = 0; i < N; i++, ++it )
            {
                double v = *(const float*)it.ptr; 
                result += v*v;
            }
    }
    else if( type == CV_64F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
                result = std::max(result, std::abs(*(const double*)it.ptr));
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
                result += std::abs(*(const double*)it.ptr);
        else
            for( i = 0; i < N; i++, ++it )
            {
                double v = *(const double*)it.ptr; 
                result += v*v;
            }
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );
    
    if( normType == NORM_L2 )
        result = std::sqrt(result);
    return result;
}
    
void minMaxLoc( const SparseMat& src, double* _minval, double* _maxval, int* _minidx, int* _maxidx )
{
    SparseMatConstIterator it = src.begin();
    size_t i, N = src.nzcount(), d = src.hdr ? src.hdr->dims : 0;
    int type = src.type();
    const int *minidx = 0, *maxidx = 0;
    
    if( type == CV_32F )
    {
        float minval = FLT_MAX, maxval = -FLT_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            float v = *(const float*)it.ptr;
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else if( type == CV_64F )
    {
        double minval = DBL_MAX, maxval = -DBL_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            double v = *(const double*)it.ptr;
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );
    
    if( _minidx )
        for( i = 0; i < d; i++ )
            _minidx[i] = minidx[i];
    if( _maxidx )
        for( i = 0; i < d; i++ )
            _maxidx[i] = maxidx[i];
}

    
void normalize( const SparseMat& src, SparseMat& dst, double a, int norm_type )
{
    double scale = 1;
    if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( src, norm_type );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );
    
    src.convertTo( dst, -1, scale );
}
    
    
}

/* End of file. */
