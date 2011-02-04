#include "precomp.hpp"
#include <float.h>
#include <limits.h>

using namespace cv;

namespace cvtest
{

Size randomSize(RNG& rng, double maxSizeLog)
{
    double width_log = rng.uniform(0., maxSizeLog);
    double height_log = rng.uniform(0., maxSizeLog - width_log);
    if( (unsigned)rng % 2 != 0 )
        std::swap(width_log, height_log);
    Size sz;
    sz.width = cvRound(exp(width_log));
    sz.height = cvRound(exp(height_log));
    return sz;
}

void randomSize(RNG& rng, int minDims, int maxDims, double maxSizeLog, vector<int>& sz)
{
    int i, dims = rng.uniform(minDims, maxDims+1);
    sz.resize(dims);
    for( i = 0; i < dims; i++ )
    {
        double v = rng.uniform(0., maxSizeLog);
        maxSizeLog -= v;
        sz[i] = cvRound(exp(v));
    }
    for( i = 0; i < dims; i++ )
    {
        int j = rng.uniform(0, dims);
        int k = rng.uniform(0, dims);
        std::swap(sz[j], sz[k]);
    }
}

int randomType(RNG& rng, int typeMask, int minChannels, int maxChannels)
{
    int channels = rng.uniform(minChannels, maxChannels+1);
    int depth = 0;
    CV_Assert((typeMask & TYPE_MASK_ALL) != 0);
    for(;;)
    {
        depth = rng.uniform(CV_8U, CV_64F+1);
        if( ((1 << depth) & typeMask) != 0 )
            break;
    }
    return CV_MAKETYPE(depth, channels);
}

double getMinVal(int depth)
{
    double val = depth == CV_8U ? 0 : depth == CV_8S ? SCHAR_MIN : depth == CV_16U ? 0 :
    depth == CV_16S ? SHRT_MIN : depth == CV_32S ? INT_MIN :
    depth == CV_32F ? -FLT_MAX : depth == CV_64F ? -DBL_MAX : -1;
    CV_Assert(val != -1);
    return val;
}

double getMaxVal(int depth)
{
    double val = depth == CV_8U ? UCHAR_MAX : depth == CV_8S ? SCHAR_MAX : depth == CV_16U ? USHRT_MAX :
    depth == CV_16S ? SHRT_MAX : depth == CV_32S ? INT_MAX :
    depth == CV_32F ? FLT_MAX : depth == CV_64F ? DBL_MAX : -1;
    CV_Assert(val != -1);
    return val;
}    
    
Mat randomMat(RNG& rng, Size size, int type, double minVal, double maxVal, bool useRoi)
{
    Size size0 = size;
    if( useRoi )
    {
        size0.width += std::max(rng.uniform(0, 10) - 5, 0);
        size0.height += std::max(rng.uniform(0, 10) - 5, 0);
    }
        
    Mat m(size0, type);
    
    rng.fill(m, RNG::UNIFORM, Scalar::all(minVal), Scalar::all(maxVal));
    if( size0 == size )
        return m;
    return m(Rect((size0.width-size.width)/2, (size0.height-size.height)/2, size.width, size.height));
}

Mat randomMat(RNG& rng, const vector<int>& size, int type, double minVal, double maxVal, bool useRoi)
{
    int i, dims = (int)size.size();
    vector<int> size0(dims);
    vector<Range> r(dims);
    bool eqsize = true;
    for( i = 0; i < dims; i++ )
    {
        size0[i] = size[i];
        r[i] = Range::all();
        if( useRoi )
        {
            size0[i] += std::max(rng.uniform(0, 5) - 2, 0);
            r[i] = Range((size0[i] - size[i])/2, (size0[i] - size[i])/2 + size[i]);
        }
        eqsize = eqsize && size[i] == size0[i];
    }
    
    Mat m(dims, &size0[0], type);
    
    rng.fill(m, RNG::UNIFORM, Scalar::all(minVal), Scalar::all(maxVal));
    if( eqsize )
        return m;
    return m(&r[0]);
}
    
void add(const Mat& _a, double alpha, const Mat& _b, double beta,
        Scalar gamma, Mat& c, int ctype, bool calcAbs)
{
    Mat a = _a, b = _b;
    if( a.empty() || alpha == 0 )
    {
        // both alpha and beta can be 0, but at least one of a and b must be non-empty array,
        // otherwise we do not know the size of the output (and may be type of the output, when ctype<0)
        CV_Assert( !a.empty() || !b.empty() );
        if( !b.empty() )
        {
            a = b;
            alpha = beta;
            b = Mat();
            beta = 0;
        }
    }
    if( b.empty() || beta == 0 )
    {
        b = Mat();
        beta = 0;
    }
    else
        CV_Assert(a.size == b.size);
    
    if( ctype < 0 )
        ctype = a.depth();
    ctype = CV_MAKETYPE(CV_MAT_DEPTH(ctype), a.channels());
    c.create(a.dims, &a.size[0], ctype);
    const Mat *arrays[3];
    Mat planes[3], buf[3];
    arrays[0] = &a;
    arrays[1] = b.empty() ? 0 : &b;
    arrays[2] = &c;
    
    NAryMatIterator it(arrays, planes, 3);
    int i, nplanes = it.nplanes, cn=a.channels();
    size_t total = planes[0].total(), maxsize = std::min((size_t)12*12*std::max(12/cn, 1), total);
    
    CV_Assert(planes[0].rows == 1);
    buf[0].create(1, (int)maxsize, CV_64FC(cn));
    if(!b.empty())
        buf[1].create(1, maxsize, CV_64FC(cn));
    buf[2].create(1, maxsize, CV_64FC(cn));
    scalarToRawData(gamma, buf[2].data, CV_64FC(cn), (int)(maxsize*cn));
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        for( size_t j = 0; j < total; j += maxsize )
        {
            size_t j2 = min(j + maxsize, total);
            Mat apart0 = planes[0].colRange((int)j, (int)j2);
            Mat cpart0 = planes[2].colRange((int)j, (int)j2);
            Mat apart = buf[0].colRange(0, (int)(j2 - j));
            
            apart0.convertTo(apart, apart.type(), alpha);
            size_t k, n = (j2 - j)*cn;
            double* aptr = (double*)apart.data;
            const double* gptr = (const double*)buf[2].data;
            
            if( b.empty() )
            {
                for( k = 0; k < n; k++ )
                    aptr[k] += gptr[k];
            }
            else
            {
                Mat bpart0 = planes[1].colRange((int)j, (int)j2);
                Mat bpart = buf[1].colRange(0, (int)(j2 - j));
                bpart0.convertTo(bpart, bpart.type(), beta);
                const double* bptr = (const double*)bpart.data;
                
                for( k = 0; k < n; k++ )
                    aptr[k] += bptr[k] + gptr[k];
            }
            if( calcAbs )
                for( k = 0; k < n; k++ )
                    aptr[k] = fabs(aptr[k]);
            apart.convertTo(cpart0, cpart0.type(), 1, 0);
        }
    }
}


template<typename _Tp1, typename _Tp2> inline void
convert_(const _Tp1* src, _Tp2* dst, size_t total, double alpha, double beta)
{
    size_t i;
    if( alpha == 1 && beta == 0 )
        for( i = 0; i < total; i++ )
            dst[i] = saturate_cast<_Tp2>(src[i]);
    else if( beta == 0 )
        for( i = 0; i < total; i++ )
            dst[i] = saturate_cast<_Tp2>(src[i]*alpha);
    else
        for( i = 0; i < total; i++ )
            dst[i] = saturate_cast<_Tp2>(src[i]*alpha + beta);
}

template<typename _Tp> inline void
convertTo(const _Tp* src, void* dst, int dtype, size_t total, double alpha, double beta)
{
    switch( CV_MAT_DEPTH(dtype) )
    {
    case CV_8U:
        convert_(src, (uchar*)dst, total, alpha, beta);
        break;
    case CV_8S:
        convert_(src, (schar*)dst, total, alpha, beta);
        break;
    case CV_16U:
        convert_(src, (ushort*)dst, total, alpha, beta);
        break;
    case CV_16S:
        convert_(src, (short*)dst, total, alpha, beta);
        break;
    case CV_32S:
        convert_(src, (int*)dst, total, alpha, beta);
        break;
    case CV_32F:
        convert_(src, (float*)dst, total, alpha, beta);
        break;
    case CV_64F:
        convert_(src, (double*)dst, total, alpha, beta);
        break;
    default:
        CV_Assert(0);
    }
}
    
void convert(const Mat& src, Mat& dst, int dtype, double alpha, double beta)
{
    dtype = CV_MAKETYPE(CV_MAT_DEPTH(dtype), src.channels());
    dst.create(src.dims, &src.size[0], dtype);
    if( alpha == 0 )
    {
        set( dst, Scalar::all(beta) );
        return;
    }
    if( dtype == src.type() && alpha == 1 && beta == 0 )
    {
        copy( src, dst );
        return;
    }
    
    const Mat *arrays[]={&src, &dst};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t total = planes[0].total()*planes[0].channels();
    int i, nplanes = it.nplanes;
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        
        switch( src.depth() )
        {
        case CV_8U:
            convertTo((const uchar*)sptr, dptr, dtype, total, alpha, beta);
            break;
        case CV_8S:
            convertTo((const schar*)sptr, dptr, dtype, total, alpha, beta);
            break;
        case CV_16U:
            convertTo((const ushort*)sptr, dptr, dtype, total, alpha, beta);
            break;
        case CV_16S:
            convertTo((const short*)sptr, dptr, dtype, total, alpha, beta);
            break;
        case CV_32S:
            convertTo((const int*)sptr, dptr, dtype, total, alpha, beta);
            break;
        case CV_32F:
            convertTo((const float*)sptr, dptr, dtype, total, alpha, beta);
            break;
        case CV_64F:
            convertTo((const double*)sptr, dptr, dtype, total, alpha, beta);
            break;
        }
    }
}
    
     
void copy(const Mat& src, Mat& dst, const Mat& mask, bool invertMask)
{
    dst.create(src.dims, &src.size[0], src.type());
    
    if(mask.empty())
    {
        const Mat* arrays[] = {&src, &dst};
        Mat planes[2];
        NAryMatIterator it(arrays, planes, 2);
        int i, nplanes = it.nplanes;
        size_t planeSize = planes[0].total()*src.elemSize();
        
        for( i = 0; i < nplanes; i++, ++it )
            memcpy(planes[1].data, planes[0].data, planeSize);
        
        return;
    }
    
    CV_Assert( src.size == mask.size && mask.type() == CV_8U );
    
    const Mat *arrays[3]={&src, &dst, &mask};
    Mat planes[3];
    
    NAryMatIterator it(arrays, planes, 3);
    size_t j, k, elemSize = src.elemSize(), total = planes[0].total();
    int i, nplanes = it.nplanes;
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        const uchar* mptr = planes[2].data;
        
        for( j = 0; j < total; j++, sptr += elemSize, dptr += elemSize )
        {
            if( (mptr[j] != 0) ^ invertMask )
                for( k = 0; k < elemSize; k++ )
                    dptr[k] = sptr[k];
        }
    }
}

    
void set(Mat& dst, const Scalar& gamma, const Mat& mask)
{
    double buf[12];
    scalarToRawData(gamma, &buf, dst.type(), dst.channels());
    const uchar* gptr = (const uchar*)&buf[0];
    
    if(mask.empty())
    {
        const Mat* arrays[] = {&dst};
        Mat plane;
        NAryMatIterator it(arrays, &plane, 1);
        int i, nplanes = it.nplanes;
        size_t j, k, elemSize = dst.elemSize(), planeSize = plane.total()*elemSize;
        
        for( k = 1; k < elemSize; k++ )
            if( gptr[k] != gptr[0] )
                break;
        bool uniform = k >= elemSize;
        
        for( i = 0; i < nplanes; i++, ++it )
        {
            uchar* dptr = plane.data;
            if( uniform )
                memset( dptr, gptr[0], planeSize );
            else if( i == 0 )
            {
                for( j = 0; j < planeSize; j += elemSize, dptr += elemSize )
                    for( k = 0; k < elemSize; k++ )
                        dptr[k] = gptr[k];
            }
            else
                memcpy(dptr, dst.data, planeSize);
        }
        return;
    }
    
    CV_Assert( dst.size == mask.size && mask.type() == CV_8U );
    
    const Mat *arrays[2]={&dst, &mask};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t j, k, elemSize = dst.elemSize(), total = planes[0].total();
    int i, nplanes = it.nplanes;
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        uchar* dptr = planes[0].data;
        const uchar* mptr = planes[1].data;
        
        for( j = 0; j < total; j++, dptr += elemSize )
        {
            if( mptr[j] )
                for( k = 0; k < elemSize; k++ )
                    dptr[k] = gptr[k];
        }
    }
}

    
template<typename _Tp> static void
erode_(const Mat& src, Mat& dst, const vector<int>& ofsvec)
{
    int width = src.cols*src.channels(), n = (int)ofsvec.size();
    const int* ofs = &ofsvec[0];
    
    for( int y = 0; y < dst.rows; y++ )
    {
        const _Tp* sptr = src.ptr<_Tp>(y);
        _Tp* dptr = dst.ptr<_Tp>(y);
        
        for( int x = 0; x < width; x++ )
        {
            _Tp result = sptr[x + ofs[0]];
            for( int i = 1; i < n; i++ )
                result = std::min(result, sptr[x + ofs[i]]);
            dptr[x] = result;
        }
    }
}

    
template<typename _Tp> static void
dilate_(const Mat& src, Mat& dst, const vector<int>& ofsvec)
{
    int width = src.cols*src.channels(), n = (int)ofsvec.size();
    const int* ofs = &ofsvec[0];
    
    for( int y = 0; y < dst.rows; y++ )
    {
        const _Tp* sptr = src.ptr<_Tp>(y);
        _Tp* dptr = dst.ptr<_Tp>(y);
        
        for( int x = 0; x < width; x++ )
        {
            _Tp result = sptr[x + ofs[0]];
            for( int i = 1; i < n; i++ )
                result = std::max(result, sptr[x + ofs[i]]);
            dptr[x] = result;
        }
    }
}
    
    
void erode(const Mat& _src, Mat& dst, const Mat& _kernel, Point anchor,
           int borderType, const Scalar& _borderValue)
{
    Mat kernel = _kernel, src;
    Scalar borderValue = _borderValue;
    if( kernel.empty() )
        kernel = Mat::ones(3, 3, CV_8U);
    else
    {
        CV_Assert( kernel.type() == CV_8U );
    }
    if( anchor == Point(-1,-1) )
        anchor = Point(kernel.cols/2, kernel.rows/2);
    if( borderType == IPL_BORDER_CONSTANT )
        borderValue = getMaxVal(src.depth());
    copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y - 1,
                   anchor.x, kernel.cols - anchor.y - 1,
                   borderType, borderValue);
    dst.create( _src.size(), src.type() );
    
    vector<int> ofs;
    int step = (int)(src.step/src.elemSize1()), cn = src.channels();
    for( int i = 0; i < kernel.rows; i++ )
        for( int j = 0; j < kernel.cols; j++ )
            if( kernel.at<uchar>(i, j) != 0 )
                ofs.push_back(i*step + j*cn);
    if( ofs.empty() )
        ofs.push_back(anchor.y*step + anchor.x*cn);
    
    switch( src.depth() )
    {
    case CV_8U:
        erode_<uchar>(src, dst, ofs);
        break;
    case CV_8S:
        erode_<schar>(src, dst, ofs);
        break;
    case CV_16U:
        erode_<ushort>(src, dst, ofs);
        break;
    case CV_16S:
        erode_<short>(src, dst, ofs);
        break;
    case CV_32S:
        erode_<int>(src, dst, ofs);
        break;
    case CV_32F:
        erode_<float>(src, dst, ofs);
        break;
    case CV_64F:
        erode_<double>(src, dst, ofs);
        break;
    default:
        CV_Assert(0);
    }
}

void dilate(const Mat& _src, Mat& dst, const Mat& _kernel, Point anchor,
            int borderType, const Scalar& _borderValue)
{
    Mat kernel = _kernel, src;
    Scalar borderValue = _borderValue;
    if( kernel.empty() )
        kernel = Mat::ones(3, 3, CV_8U);
    else
    {
        CV_Assert( kernel.type() == CV_8U );
    }
    if( anchor == Point(-1,-1) )
        anchor = Point(kernel.cols/2, kernel.rows/2);
    if( borderType == IPL_BORDER_CONSTANT )
        borderValue = getMinVal(src.depth());
    copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y - 1,
                   anchor.x, kernel.cols - anchor.y - 1,
                   borderType, borderValue);
    dst.create( _src.size(), src.type() );
    
    vector<int> ofs;
    int step = (int)(src.step/src.elemSize1()), cn = src.channels();
    for( int i = 0; i < kernel.rows; i++ )
        for( int j = 0; j < kernel.cols; j++ )
            if( kernel.at<uchar>(i, j) != 0 )
                ofs.push_back(i*step + j*cn);
    if( ofs.empty() )
        ofs.push_back(anchor.y*step + anchor.x*cn);
    
    switch( src.depth() )
    {
    case CV_8U:
        dilate_<uchar>(src, dst, ofs);
        break;
    case CV_8S:
        dilate_<schar>(src, dst, ofs);
        break;
    case CV_16U:
        dilate_<ushort>(src, dst, ofs);
        break;
    case CV_16S:
        dilate_<short>(src, dst, ofs);
        break;
    case CV_32S:
        dilate_<int>(src, dst, ofs);
        break;
    case CV_32F:
        dilate_<float>(src, dst, ofs);
        break;
    case CV_64F:
        dilate_<double>(src, dst, ofs);
        break;
    default:
        CV_Assert(0);
    }
}    

    
template<typename _Tp> static void
filter2D_(const Mat& src, Mat& dst, const vector<int>& ofsvec, const vector<double>& coeffvec)
{
    const int* ofs = &ofsvec[0];
    const double* coeff = &coeffvec[0];
    int width = dst.cols*dst.channels(), ncoeffs = (int)ofsvec.size();
    
    for( int y = 0; y < dst.rows; y++ )
    {
        const _Tp* sptr = src.ptr<_Tp>(y);
        double* dptr = dst.ptr<double>(y);
        
        for( int x = 0; x < width; x++ )
        {
            double s = 0;
            for( int i = 0; i < ncoeffs; i++ )
                s += sptr[ofs[i]]*coeff[i];
            dptr[x] = s;
        }
    }
}
    
    
void filter2D(const Mat& _src, Mat& dst, int ddepth, const Mat& _kernel,
              Point anchor, double delta, int borderType, const Scalar& _borderValue)
{
    Mat kernel = _kernel, src, _dst;
    Scalar borderValue = _borderValue;
    CV_Assert( kernel.type() == CV_32F || kernel.type() == CV_64F );
    if( anchor == Point(-1,-1) )
        anchor = Point(kernel.cols/2, kernel.rows/2);
    if( borderType == IPL_BORDER_CONSTANT )
        borderValue = getMinVal(src.depth());
    copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y - 1,
                   anchor.x, kernel.cols - anchor.y - 1,
                   borderType, borderValue);
    _dst.create( _src.size(), CV_MAKETYPE(CV_64F, src.channels()) );
    
    vector<int> ofs;
    vector<double> coeff(kernel.rows*kernel.cols);
    Mat cmat(kernel.rows, kernel.cols, CV_64F, &coeff[0]);
    convert(kernel, cmat, cmat.type());
    
    int step = (int)(src.step/src.elemSize1()), cn = src.channels();
    for( int i = 0; i < kernel.rows; i++ )
        for( int j = 0; j < kernel.cols; j++ )
                ofs.push_back(i*step + j*cn);
    
    switch( src.depth() )
    {
    case CV_8U:
        filter2D_<uchar>(src, _dst, ofs, coeff);
        break;
    case CV_8S:
        filter2D_<schar>(src, _dst, ofs, coeff);
        break;
    case CV_16U:
        filter2D_<ushort>(src, _dst, ofs, coeff);
        break;
    case CV_16S:
        filter2D_<short>(src, _dst, ofs, coeff);
        break;
    case CV_32S:
        filter2D_<int>(src, _dst, ofs, coeff);
        break;
    case CV_32F:
        filter2D_<float>(src, _dst, ofs, coeff);
        break;
    case CV_64F:
        filter2D_<double>(src, _dst, ofs, coeff);
        break;
    default:
        CV_Assert(0);
    }
    
    convert(_dst, dst, ddepth, 1, delta);
}


static int borderInterpolate( int p, int len, int borderType )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == IPL_BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == IPL_BORDER_REFLECT || borderType == IPL_BORDER_REFLECT_101 )
    {
        int delta = borderType == IPL_BORDER_REFLECT_101;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == IPL_BORDER_WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    else if( borderType == IPL_BORDER_CONSTANT )
        p = -1;
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported border type" );
    return p;
}    
    
    
void copyMakeBorder(const Mat& src, Mat& dst, int top, int bottom, int left, int right,
                    int borderType, const Scalar& borderValue)
{
    dst.create(src.rows + top + bottom, src.cols + left + right, src.type());
    int i, j, k, esz = (int)src.elemSize();
    int width = src.cols*esz, width1 = dst.cols*esz;
    
    if( borderType == IPL_BORDER_CONSTANT )
    {
        vector<uchar> valvec((src.cols + left + right)*esz);
        uchar* val = &valvec[0];
        scalarToRawData(borderValue, val, src.type(), (src.cols + left + right)*src.channels());
        
        left *= esz;
        right *= esz;
        for( i = 0; i < src.rows; i++ )
        {
            const uchar* sptr = src.ptr(i);
            uchar* dptr = dst.ptr(i + top) + left;
            for( j = 0; j < left; j++ )
                dptr[j - left] = val[j];
            if( dptr != sptr )
                for( j = 0; j < width; j++ )
                    dptr[j] = sptr[j];
            for( j = 0; j < right; j++ )
                dptr[j + width] = val[j];
        }
        
        for( i = 0; i < top; i++ )
        {
            uchar* dptr = dst.ptr(i);
            for( j = 0; j < width1; j++ )
                dptr[j] = val[j];
        }
        
        for( i = 0; i < bottom; i++ )
        {
            uchar* dptr = dst.ptr(i + top + src.rows);
            for( j = 0; j < width1; j++ )
                dptr[j] = val[j];
        }
    }
    else
    {
        vector<int> tabvec((left + right)*esz);
        int* ltab = &tabvec[0];
        int* rtab = &tabvec[left*esz];
        for( i = 0; i < left; i++ )
        {
            j = borderInterpolate(i - left, src.cols, borderType)*esz;
            for( k = 0; k < esz; k++ )
                ltab[i*esz + k] = j + k;
        }
        for( i = 0; i < right; i++ )
        {
            j = borderInterpolate(src.cols + i, src.cols, borderType)*esz;
            for( k = 0; k < esz; k++ )
                rtab[i*esz + k] = j + k;
        }
        
        left *= esz;
        right *= esz;
        for( i = 0; i < src.rows; i++ )
        {
            const uchar* sptr = src.ptr(i);
            uchar* dptr = dst.ptr(i + top);
            
            for( j = 0; j < left; j++ )
                dptr[j] = sptr[ltab[j]];
            if( dptr + left != sptr )
            {
                for( j = 0; j < width; j++ )
                    dptr[j + left] = sptr[j];
            }
            for( j = 0; j < right; j++ )
                dptr[j + left + width] = sptr[rtab[j]];
        }
        
        for( i = 0; i < top; i++ )
        {
            j = borderInterpolate(i - top, src.rows, borderType);
            const uchar* sptr = dst.ptr(j + top);
            uchar* dptr = dst.ptr(i);
            
            for( k = 0; k < width1; k++ )
                dptr[k] = sptr[k];
        }
        
        for( i = 0; i < bottom; i++ )
        {
            j = borderInterpolate(i + src.rows, src.rows, borderType);
            const uchar* sptr = dst.ptr(j + top);
            uchar* dptr = dst.ptr(i + top + src.rows);
            
            for( k = 0; k < width1; k++ )
                dptr[k] = sptr[k];
        }
    }
}


template<typename _Tp> static void
minMaxLoc_(const _Tp* src, size_t total, size_t startidx,
           double* _maxval, double* _minval,
           size_t* _minpos, size_t* _maxpos,
           const uchar* mask)
{
    _Tp maxval = saturate_cast<_Tp>(*_maxval), minval = saturate_cast<_Tp>(*_minval);
    size_t minpos = *_minpos, maxpos = *_maxpos;
    
    if( !mask )
    {
        for( size_t i = 0; i < total; i++ )
        {
            _Tp val = src[i];
            if( minval > val )
            {
                minval = val;
                minpos = startidx + i;
            }
            if( maxval < val )
            {
                maxval = val;
                maxpos = startidx + i;
            }
        }
    }
    else
    {
        for( size_t i = 0; i < total; i++ )
        {
            _Tp val = src[i];
            if( minval > val && mask[i] )
            {
                minval = val;
                minpos = startidx + i;
            }
            if( maxval < val && mask[i] )
            {
                maxval = val;
                maxpos = startidx + i;
            }
        }
    }
    
    *_maxval = maxval;
    *_minval = minval;
    *_maxpos = maxpos;
    *_minpos = minpos;
}


static void setpos( const Mat& mtx, vector<int>& pos, size_t idx )
{
    pos.resize(mtx.dims);
    for( int i = mtx.dims-1; i >= 0; i-- )
    {
        int sz = mtx.size[i]*(i == mtx.dims-1 ? mtx.channels() : 1);
        pos[i] = idx % sz;
        idx /= sz;
    }
}
    
void minMaxLoc(const Mat& src, double* _maxval, double* _minval,
               vector<int>* _maxloc, vector<int>* _minloc,
               const Mat& mask)
{
    CV_Assert( src.channels() == 1 );
    const Mat *arrays[2]={&src, &mask};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t startidx = 1, total = planes[0].total();
    int i, nplanes = it.nplanes, depth = src.depth();
    double maxval = depth < CV_32F ? INT_MIN : depth == CV_32F ? -FLT_MAX : -DBL_MAX;
    double minval = depth < CV_32F ? INT_MAX : depth == CV_32F ? FLT_MAX : DBL_MAX;
    size_t maxidx = 0, minidx = 0;
    
    for( i = 0; i < nplanes; i++, ++it, startidx += total )
    {
        const uchar* sptr = planes[0].data;
        const uchar* mptr = planes[1].data;
        
        switch( depth )
        {
        case CV_8U:
            minMaxLoc_((const uchar*)sptr, total, startidx,
                       &maxval, &minval, &maxidx, &minidx, mptr);
            break;
        case CV_8S:
            minMaxLoc_((const schar*)sptr, total, startidx,
                       &maxval, &minval, &maxidx, &minidx, mptr);
            break;
        case CV_16U:
            minMaxLoc_((const ushort*)sptr, total, startidx,
                       &maxval, &minval, &maxidx, &minidx, mptr);
            break;
        case CV_16S:
            minMaxLoc_((const short*)sptr, total, startidx,
                       &maxval, &minval, &maxidx, &minidx, mptr);
            break;
        case CV_32S:
            minMaxLoc_((const int*)sptr, total, startidx,
                       &maxval, &minval, &maxidx, &minidx, mptr);
            break;
        case CV_32F:
            minMaxLoc_((const float*)sptr, total, startidx,
                       &maxval, &minval, &maxidx, &minidx, mptr);
            break;
        case CV_64F:
            minMaxLoc_((const double*)sptr, total, startidx,
                       &maxval, &minval, &maxidx, &minidx, mptr);
            break;
        default:
            CV_Assert(0);
        }
    }
    
    if( _maxval )
        *_maxval = maxval;
    if( _minval )
        *_minval = minval;
    if( _maxloc )
        setpos( src, *_maxloc, maxidx );
    if( _minloc )
        setpos( src, *_minloc, minidx );
}

    
template<typename _Tp> static double
norm_(const _Tp* src, size_t total, int normType, double startval, const uchar* mask)
{
    size_t i;
    double result = startval;
    
    if( normType == NORM_INF )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result = std::max(result, (double)std::abs(src[i]));
        else
            for( i = 0; i < total; i++ )
                if( mask[i] )
                    result = std::max(result, (double)std::abs(src[i]));
    }
    else if( normType == NORM_L1 )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result += std::abs(src[i]);
        else
            for( i = 0; i < total; i++ )
                if( mask[i] )
                    result += std::abs(src[i]);
    }
    else
    {
        if( !mask )
            for( i = 0; i < total; i++ )
            {
                double v = src[i];
                result += v*v;
            }
        else
            for( i = 0; i < total; i++ )
                if( mask[i] )
                {
                    double v = src[i];
                    result += v*v;
                }
    }
    return result;
}


template<typename _Tp> static double
norm_(const _Tp* src1, const _Tp* src2, size_t total, int normType, double startval, const uchar* mask)
{
    size_t i;
    double result = startval;
    
    if( normType == NORM_INF )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result = std::max(result, (double)std::abs(src1[i] - src2[i]));
        else
            for( i = 0; i < total; i++ )
                if( mask[i] )
                    result = std::max(result, (double)std::abs(src1[i] - src2[i]));
    }
    else if( normType == NORM_L1 )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result += std::abs(src1[i] - src2[i]);
        else
            for( i = 0; i < total; i++ )
                if( mask[i] )
                    result += std::abs(src1[i] - src2[i]);
    }
    else
    {
        if( !mask )
            for( i = 0; i < total; i++ )
            {
                double v = src1[i] - src2[i];
                result += v*v;
            }
        else
            for( i = 0; i < total; i++ )
                if( mask[i] )
                {
                    double v = src1[i] - src2[i];
                    result += v*v;
                }
    }
    return result;
}
    
    
double norm(const Mat& src, int normType, const Mat& mask)
{
    CV_Assert( mask.empty() || (src.size == mask.size && mask.type() == CV_8U) );
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );
    const Mat *arrays[2]={&src, &mask};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t total = planes[0].total()*planes[0].channels();
    int i, nplanes = it.nplanes, depth = src.depth();
    double result = 0;
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].data;
        const uchar* mptr = planes[1].data;
        
        switch( depth )
        {
        case CV_8U:
            result = norm_((const uchar*)sptr, total, normType, result, mptr);
            break;
        case CV_8S:
            result = norm_((const schar*)sptr, total, normType, result, mptr);
            break;
        case CV_16U:
            result = norm_((const ushort*)sptr, total, normType, result, mptr);
            break;
        case CV_16S:
            result = norm_((const short*)sptr, total, normType, result, mptr);
            break;
        case CV_32S:
            result = norm_((const int*)sptr, total, normType, result, mptr);
            break;
        case CV_32F:
            result = norm_((const float*)sptr, total, normType, result, mptr);
            break;
        case CV_64F:
            result = norm_((const double*)sptr, total, normType, result, mptr);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        };
    }
    if( normType == NORM_L2 )
        result = sqrt(result);
    return result;
}
    
double norm(const Mat& src1, const Mat& src2, int normType, const Mat& mask)
{
    CV_Assert( src1.type() == src2.type() && src1.size == src2.size );
    CV_Assert( mask.empty() || (src1.size == mask.size && mask.type() == CV_8U) );
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );
    const Mat *arrays[3]={&src1, &src2, &mask};
    Mat planes[3];
    
    NAryMatIterator it(arrays, planes, 3);
    size_t total = planes[0].total()*planes[0].channels();
    int i, nplanes = it.nplanes, depth = src1.depth();
    double result = 0;
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].data;
        const uchar* sptr2 = planes[1].data;
        const uchar* mptr = planes[2].data;
        
        switch( depth )
        {
        case CV_8U:
            result = norm_((const uchar*)sptr1, (const uchar*)sptr2, total, normType, result, mptr);
            break;
        case CV_8S:
            result = norm_((const schar*)sptr1, (const schar*)sptr2, total, normType, result, mptr);
            break;
        case CV_16U:
            result = norm_((const ushort*)sptr1, (const ushort*)sptr2, total, normType, result, mptr);
            break;
        case CV_16S:
            result = norm_((const short*)sptr1, (const short*)sptr2, total, normType, result, mptr);
            break;
        case CV_32S:
            result = norm_((const int*)sptr1, (const int*)sptr2, total, normType, result, mptr);
            break;
        case CV_32F:
            result = norm_((const float*)sptr1, (const float*)sptr2, total, normType, result, mptr);
            break;
        case CV_64F:
            result = norm_((const double*)sptr1, (const double*)sptr2, total, normType, result, mptr);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        };
    }
    if( normType == NORM_L2 )
        result = sqrt(result);
    return result;
}
    
bool cmpEps(const Mat& src1, const Mat& src2, int int_maxdiff, int flt_maxulp, vector<int>* loc);

static void
logicOp_(const uchar* src1, const uchar* src2, uchar* dst, size_t total, char c)
{
    size_t i;
    if( c == '&' )
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] & src2[i];
    else if( c == '|' )
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] | src2[i];
    else
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] ^ src2[i];
}

static void
logicOpS_(const uchar* src, const uchar* scalar, uchar* dst, size_t total, char c)
{
    const size_t blockSize = 96;
    size_t i, j;
    if( c == '&' )
        for( i = 0; i < total; i += blockSize, dst += blockSize, src += blockSize )
        {
            size_t sz = std::min(total - i, blockSize);
            for( j = 0; j < sz; j++ )
                dst[j] = src[j] & scalar[j];
        }
    else if( c == '|' )
        for( i = 0; i < total; i += blockSize, dst += blockSize, src += blockSize )
        {
            size_t sz = std::min(total - i, blockSize);
            for( j = 0; j < sz; j++ )
                dst[j] = src[j] | scalar[j];
        }
    else if( c == '^' )
    {
        for( i = 0; i < total; i += blockSize, dst += blockSize, src += blockSize )
        {
            size_t sz = std::min(total - i, blockSize);
            for( j = 0; j < sz; j++ )
                dst[j] = src[j] ^ scalar[j];
        }
    }
    else
        for( i = 0; i < total; i++ )
            dst[i] = ~src[i];
}    
    
void logicOp( const Mat& src1, const Mat& src2, Mat& dst, char op )
{
    CV_Assert( op == '&' || op == '|' || op == '^' );
    CV_Assert( src1.type() == src2.type() && src1.size == src2.size );
    dst.create( src1.dims, &src1.size[0], src1.type() );
    const Mat *arrays[3]={&src1, &src2, &dst};
    Mat planes[3];
    
    NAryMatIterator it(arrays, planes, 3);
    size_t total = planes[0].total()*planes[0].elemSize();
    int i, nplanes = it.nplanes;
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].data;
        const uchar* sptr2 = planes[1].data;
        uchar* dptr = planes[2].data;
        
        logicOp_(sptr1, sptr2, dptr, total, op);
    }
}
    
    
void logicOp(const Mat& src, const Scalar& s, Mat& dst, char op)
{
    CV_Assert( op == '&' || op == '|' || op == '^' );
    dst.create( src.dims, &src.size[0], src.type() );
    const Mat *arrays[2]={&src, &dst};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t total = planes[0].total()*planes[0].elemSize();
    int i, nplanes = it.nplanes;
    double buf[12];
    scalarToRawData(s, buf, src.type(), 12);
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        
        logicOpS_(sptr, (uchar*)&buf[0], dptr, total, op);
    }
}


template<typename _Tp> static void
compare_(const _Tp* src1, const _Tp* src2, uchar* dst, size_t total, int cmpop)
{
    size_t i;
    switch( cmpop )
    {
    case CMP_LT:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] < src2[i] ? 255 : 0;
        break;
    case CMP_LE:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] <= src2[i] ? 255 : 0;
        break;
    case CMP_EQ:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] == src2[i] ? 255 : 0;
        break;
    case CMP_NE:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] != src2[i] ? 255 : 0;
        break;
    case CMP_GE:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] >= src2[i] ? 255 : 0;
        break;
    case CMP_GT:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] > src2[i] ? 255 : 0;
        break;
    default:
        CV_Error(CV_StsBadArg, "Unknown comparison operation");
    }
}


template<typename _Tp, typename _WTp> static void
compareS_(const _Tp* src1, _WTp value, uchar* dst, size_t total, int cmpop)
{
    size_t i;
    switch( cmpop )
    {
    case CMP_LT:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] < value ? 255 : 0;
        break;
    case CMP_LE:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] <= value ? 255 : 0;
        break;
    case CMP_EQ:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] == value ? 255 : 0;
        break;
    case CMP_NE:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] != value ? 255 : 0;
        break;
    case CMP_GE:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] >= value ? 255 : 0;
        break;
    case CMP_GT:
        for( i = 0; i < total; i++ )
            dst[i] = src1[i] > value ? 255 : 0;
        break;
    default:
        CV_Error(CV_StsBadArg, "Unknown comparison operation");
    }
}    
    
    
void compare(const Mat& src1, const Mat& src2, Mat& dst, int cmpop)
{
    CV_Assert( src1.type() == src2.type() && src1.channels() == 1 && src1.size == src2.size );
    dst.create( src1.dims, &src1.size[0], CV_8U );
    const Mat *arrays[3]={&src1, &src2, &dst};
    Mat planes[3];
    
    NAryMatIterator it(arrays, planes, 3);
    size_t total = planes[0].total()*planes[0].elemSize();
    int i, nplanes = it.nplanes, depth = src1.depth();
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].data;
        const uchar* sptr2 = planes[1].data;
        uchar* dptr = planes[2].data;
        
        switch( depth )
        {
        case CV_8U:
            compare_((const uchar*)sptr1, (const uchar*)sptr2, dptr, total, cmpop);
            break;
        case CV_8S:
            compare_((const schar*)sptr1, (const schar*)sptr2, dptr, total, cmpop);
            break;
        case CV_16U:
            compare_((const ushort*)sptr1, (const ushort*)sptr2, dptr, total, cmpop);
            break;
        case CV_16S:
            compare_((const short*)sptr1, (const short*)sptr2, dptr, total, cmpop);
            break;
        case CV_32S:
            compare_((const int*)sptr1, (const int*)sptr2, dptr, total, cmpop);
            break;
        case CV_32F:
            compare_((const float*)sptr1, (const float*)sptr2, dptr, total, cmpop);
            break;
        case CV_64F:
            compare_((const double*)sptr1, (const double*)sptr2, dptr, total, cmpop);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        }
    }
}
    
void compare(const Mat& src, double value, Mat& dst, int cmpop)
{
    CV_Assert( src.channels() == 1 );
    dst.create( src.dims, &src.size[0], CV_8U );
    const Mat *arrays[2]={&src, &dst};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t total = planes[0].total()*planes[0].elemSize();
    int i, nplanes = it.nplanes, depth = src.depth();
    int ivalue = saturate_cast<int>(value);
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        
        switch( depth )
        {
        case CV_8U:
            compareS_((const uchar*)sptr, ivalue, dptr, total, cmpop);
            break;
        case CV_8S:
            compareS_((const schar*)sptr, ivalue, dptr, total, cmpop);
            break;
        case CV_16U:
            compareS_((const ushort*)sptr, ivalue, dptr, total, cmpop);
            break;
        case CV_16S:
            compareS_((const short*)sptr, ivalue, dptr, total, cmpop);
            break;
        case CV_32S:
            compareS_((const int*)sptr, ivalue, dptr, total, cmpop);
            break;
        case CV_32F:
            compareS_((const float*)sptr, value, dptr, total, cmpop);
            break;
        case CV_64F:
            compareS_((const double*)sptr, value, dptr, total, cmpop);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        }
    }
}

    
template<typename _Tp> static bool
cmpEpsInt_(const _Tp* src1, const _Tp* src2, int imaxdiff, size_t total, size_t startidx, size_t& idx)
{
    size_t i;
    for( i = 0; i < total; i++ )
        if( std::abs(src1[i] - src2[i]) > imaxdiff )
        {
            idx = i + startidx;
            return false;
        }
    return true;
}


template<typename _Tp> static bool
cmpEpsFlt_(const _Tp* src1, const _Tp* src2, size_t total, int imaxdiff, size_t startidx, size_t& idx)
{
    const _Tp C = ((_Tp)1 << (sizeof(_Tp)*8-1)) - 1;
    size_t i;
    for( i = 0; i < total; i++ )
    {
        _Tp a = src1[i], b = src2[i];
        if( a < 0 ) a ^= C; if( b < 0 ) b ^= C;
        _Tp d = std::abs(double(a - b));
        if( d > imaxdiff )
        {
            idx = i + startidx;
            return false;
        }
    }
    return true;
}
    
    
bool cmpEps(const Mat& src1, const Mat& src2, int imaxDiff, vector<int>* loc)
{
    CV_Assert( src1.type() == src2.type() && src1.size == src2.size );
    const Mat *arrays[2]={&src1, &src2};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t total = planes[0].total()*planes[0].channels();
    int i, nplanes = it.nplanes, depth = src1.depth();
    size_t startidx = 0, idx = -1;
    bool ok = true;
    
    for( i = 0; i < nplanes; i++, ++it, startidx += total )
    {
        const uchar* sptr1 = planes[0].data;
        const uchar* sptr2 = planes[1].data;

        switch( depth )
        {
        case CV_8U:
            ok = cmpEpsInt_((const uchar*)sptr1, (const uchar*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_8S:
            ok = cmpEpsInt_((const schar*)sptr1, (const schar*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_16U:
            ok = cmpEpsInt_((const ushort*)sptr1, (const ushort*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_16S:
            ok = cmpEpsInt_((const short*)sptr1, (const short*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_32S:
            ok = cmpEpsInt_((const int*)sptr1, (const int*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_32F:
            ok = cmpEpsFlt_((const int*)sptr1, (const int*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_64F:
            ok = cmpEpsFlt_((const int64*)sptr1, (const int64*)sptr2, total, imaxDiff, startidx, idx);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        }
        if(!ok)
            break;
    }
    if(!ok && loc)
        setpos(src1, *loc, idx);
    return ok;
}

}
