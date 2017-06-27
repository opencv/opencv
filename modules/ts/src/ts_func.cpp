#include "precomp.hpp"
#include <float.h>
#include <limits.h>
#include "opencv2/imgproc/types_c.h"

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "tegra.hpp"
#endif

using namespace cv;

namespace cvtest
{

const char* getTypeName( int type )
{
    static const char* type_names[] = { "8u", "8s", "16u", "16s", "32s", "32f", "64f", "ptr" };
    return type_names[CV_MAT_DEPTH(type)];
}

int typeByName( const char* name )
{
    int i;
    for( i = 0; i < CV_DEPTH_MAX; i++ )
        if( strcmp(name, getTypeName(i)) == 0 )
            return i;
    return -1;
}

string vec2str( const string& sep, const int* v, size_t nelems )
{
    char buf[32];
    string result = "";
    for( size_t i = 0; i < nelems; i++ )
    {
        sprintf(buf, "%d", v[i]);
        result += string(buf);
        if( i < nelems - 1 )
            result += sep;
    }
    return result;
}


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
    CV_Assert((typeMask & _OutputArray::DEPTH_MASK_ALL) != 0);
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
    depth = CV_MAT_DEPTH(depth);
    double val = depth == CV_8U ? 0 : depth == CV_8S ? SCHAR_MIN : depth == CV_16U ? 0 :
    depth == CV_16S ? SHRT_MIN : depth == CV_32S ? INT_MIN :
    depth == CV_32F ? -FLT_MAX : depth == CV_64F ? -DBL_MAX : -1;
    CV_Assert(val != -1);
    return val;
}

double getMaxVal(int depth)
{
    depth = CV_MAT_DEPTH(depth);
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

    rng.fill(m, RNG::UNIFORM, minVal, maxVal);
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

    rng.fill(m, RNG::UNIFORM, minVal, maxVal);
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
    const Mat *arrays[] = {&a, &b, &c, 0};
    Mat planes[3], buf[3];

    NAryMatIterator it(arrays, planes);
    size_t i, nplanes = it.nplanes;
    int cn=a.channels();
    int total = (int)planes[0].total(), maxsize = std::min(12*12*std::max(12/cn, 1), total);

    CV_Assert(planes[0].rows == 1);
    buf[0].create(1, maxsize, CV_64FC(cn));
    if(!b.empty())
        buf[1].create(1, maxsize, CV_64FC(cn));
    buf[2].create(1, maxsize, CV_64FC(cn));
    scalarToRawData(gamma, buf[2].ptr(), CV_64FC(cn), (int)(maxsize*cn));

    for( i = 0; i < nplanes; i++, ++it)
    {
        for( int j = 0; j < total; j += maxsize )
        {
            int j2 = std::min(j + maxsize, total);
            Mat apart0 = planes[0].colRange(j, j2);
            Mat cpart0 = planes[2].colRange(j, j2);
            Mat apart = buf[0].colRange(0, j2 - j);

            apart0.convertTo(apart, apart.type(), alpha);
            size_t k, n = (j2 - j)*cn;
            double* aptr = apart.ptr<double>();
            const double* gptr = buf[2].ptr<double>();

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
                const double* bptr = bpart.ptr<double>();

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

void convert(const Mat& src, cv::OutputArray _dst, int dtype, double alpha, double beta)
{
    if (dtype < 0) dtype = _dst.depth();

    dtype = CV_MAKETYPE(CV_MAT_DEPTH(dtype), src.channels());
    _dst.create(src.dims, &src.size[0], dtype);
    Mat dst = _dst.getMat();
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

    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].channels();
    size_t i, nplanes = it.nplanes;

    for( i = 0; i < nplanes; i++, ++it)
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

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
        const Mat* arrays[] = {&src, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        size_t i, nplanes = it.nplanes;
        size_t planeSize = planes[0].total()*src.elemSize();

        for( i = 0; i < nplanes; i++, ++it )
            memcpy(planes[1].ptr(), planes[0].ptr(), planeSize);

        return;
    }

    CV_Assert( src.size == mask.size && mask.type() == CV_8U );

    const Mat *arrays[]={&src, &dst, &mask, 0};
    Mat planes[3];

    NAryMatIterator it(arrays, planes);
    size_t j, k, elemSize = src.elemSize(), total = planes[0].total();
    size_t i, nplanes = it.nplanes;

    for( i = 0; i < nplanes; i++, ++it)
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();
        const uchar* mptr = planes[2].ptr();

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
        const Mat* arrays[] = {&dst, 0};
        Mat plane;
        NAryMatIterator it(arrays, &plane);
        size_t i, nplanes = it.nplanes;
        size_t j, k, elemSize = dst.elemSize(), planeSize = plane.total()*elemSize;

        for( k = 1; k < elemSize; k++ )
            if( gptr[k] != gptr[0] )
                break;
        bool uniform = k >= elemSize;

        for( i = 0; i < nplanes; i++, ++it )
        {
            uchar* dptr = plane.ptr();
            if( uniform )
                memset( dptr, gptr[0], planeSize );
            else if( i == 0 )
            {
                for( j = 0; j < planeSize; j += elemSize, dptr += elemSize )
                    for( k = 0; k < elemSize; k++ )
                        dptr[k] = gptr[k];
            }
            else
                memcpy(dptr, dst.ptr(), planeSize);
        }
        return;
    }

    CV_Assert( dst.size == mask.size && mask.type() == CV_8U );

    const Mat *arrays[]={&dst, &mask, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t j, k, elemSize = dst.elemSize(), total = planes[0].total();
    size_t i, nplanes = it.nplanes;

    for( i = 0; i < nplanes; i++, ++it)
    {
        uchar* dptr = planes[0].ptr();
        const uchar* mptr = planes[1].ptr();

        for( j = 0; j < total; j++, dptr += elemSize )
        {
            if( mptr[j] )
                for( k = 0; k < elemSize; k++ )
                    dptr[k] = gptr[k];
        }
    }
}


void insert(const Mat& src, Mat& dst, int coi)
{
    CV_Assert( dst.size == src.size && src.depth() == dst.depth() &&
              0 <= coi && coi < dst.channels() );

    const Mat* arrays[] = {&src, &dst, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    size_t i, nplanes = it.nplanes;
    size_t j, k, size0 = src.elemSize(), size1 = dst.elemSize(), total = planes[0].total();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr() + coi*size0;

        for( j = 0; j < total; j++, sptr += size0, dptr += size1 )
        {
            for( k = 0; k < size0; k++ )
                dptr[k] = sptr[k];
        }
    }
}


void extract(const Mat& src, Mat& dst, int coi)
{
    dst.create( src.dims, &src.size[0], src.depth() );
    CV_Assert( 0 <= coi && coi < src.channels() );

    const Mat* arrays[] = {&src, &dst, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    size_t i, nplanes = it.nplanes;
    size_t j, k, size0 = src.elemSize(), size1 = dst.elemSize(), total = planes[0].total();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr() + coi*size1;
        uchar* dptr = planes[1].ptr();

        for( j = 0; j < total; j++, sptr += size0, dptr += size1 )
        {
            for( k = 0; k < size1; k++ )
                dptr[k] = sptr[k];
        }
    }
}


void transpose(const Mat& src, Mat& dst)
{
    CV_Assert(src.dims == 2);
    dst.create(src.cols, src.rows, src.type());
    int i, j, k, esz = (int)src.elemSize();

    for( i = 0; i < dst.rows; i++ )
    {
        const uchar* sptr = src.ptr(0) + i*esz;
        uchar* dptr = dst.ptr(i);

        for( j = 0; j < dst.cols; j++, sptr += src.step[0], dptr += esz )
        {
            for( k = 0; k < esz; k++ )
                dptr[k] = sptr[k];
        }
    }
}


template<typename _Tp> static void
randUniInt_(RNG& rng, _Tp* data, size_t total, int cn, const Scalar& scale, const Scalar& delta)
{
    for( size_t i = 0; i < total; i += cn )
        for( int k = 0; k < cn; k++ )
        {
            int val = cvFloor( randInt(rng)*scale[k] + delta[k] );
            data[i + k] = saturate_cast<_Tp>(val);
        }
}


template<typename _Tp> static void
randUniFlt_(RNG& rng, _Tp* data, size_t total, int cn, const Scalar& scale, const Scalar& delta)
{
    for( size_t i = 0; i < total; i += cn )
        for( int k = 0; k < cn; k++ )
        {
            double val = randReal(rng)*scale[k] + delta[k];
            data[i + k] = saturate_cast<_Tp>(val);
        }
}


void randUni( RNG& rng, Mat& a, const Scalar& param0, const Scalar& param1 )
{
    Scalar scale = param0;
    Scalar delta = param1;
    double C = a.depth() < CV_32F ? 1./(65536.*65536.) : 1.;

    for( int k = 0; k < 4; k++ )
    {
        double s = scale.val[k] - delta.val[k];
        if( s >= 0 )
            scale.val[k] = s;
        else
        {
            delta.val[k] = scale.val[k];
            scale.val[k] = -s;
        }
        scale.val[k] *= C;
    }

    const Mat *arrays[]={&a, 0};
    Mat plane;

    NAryMatIterator it(arrays, &plane);
    size_t i, nplanes = it.nplanes;
    int depth = a.depth(), cn = a.channels();
    size_t total = plane.total()*cn;

    for( i = 0; i < nplanes; i++, ++it )
    {
        switch( depth )
        {
        case CV_8U:
            randUniInt_(rng, plane.ptr<uchar>(), total, cn, scale, delta);
            break;
        case CV_8S:
            randUniInt_(rng, plane.ptr<schar>(), total, cn, scale, delta);
            break;
        case CV_16U:
            randUniInt_(rng, plane.ptr<ushort>(), total, cn, scale, delta);
            break;
        case CV_16S:
            randUniInt_(rng, plane.ptr<short>(), total, cn, scale, delta);
            break;
        case CV_32S:
            randUniInt_(rng, plane.ptr<int>(), total, cn, scale, delta);
            break;
        case CV_32F:
            randUniFlt_(rng, plane.ptr<float>(), total, cn, scale, delta);
            break;
        case CV_64F:
            randUniFlt_(rng, plane.ptr<double>(), total, cn, scale, delta);
            break;
        default:
            CV_Assert(0);
        }
    }
}


template<typename _Tp> static void
erode_(const Mat& src, Mat& dst, const vector<int>& ofsvec)
{
    int width = dst.cols*src.channels(), n = (int)ofsvec.size();
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
    int width = dst.cols*src.channels(), n = (int)ofsvec.size();
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
    //if( _src.type() == CV_16UC3 && _src.size() == Size(1, 2) )
    //    putchar('*');
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
    if( borderType == BORDER_CONSTANT )
        borderValue = getMaxVal(src.depth());
    copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y - 1,
                   anchor.x, kernel.cols - anchor.x - 1,
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
    if( borderType == BORDER_CONSTANT )
        borderValue = getMinVal(src.depth());
    copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y - 1,
                   anchor.x, kernel.cols - anchor.x - 1,
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
                s += sptr[x + ofs[i]]*coeff[i];
            dptr[x] = s;
        }
    }
}


void filter2D(const Mat& _src, Mat& dst, int ddepth, const Mat& kernel,
              Point anchor, double delta, int borderType, const Scalar& _borderValue)
{
    Mat src, _dst;
    Scalar borderValue = _borderValue;
    CV_Assert( kernel.type() == CV_32F || kernel.type() == CV_64F );
    if( anchor == Point(-1,-1) )
        anchor = Point(kernel.cols/2, kernel.rows/2);
    if( borderType == BORDER_CONSTANT )
        borderValue = getMinVal(src.depth());
    copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y - 1,
                   anchor.x, kernel.cols - anchor.x - 1,
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
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
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
    else if( borderType == BORDER_WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    else if( borderType == BORDER_CONSTANT )
        p = -1;
    else
        CV_Error( Error::StsBadArg, "Unknown/unsupported border type" );
    return p;
}


void copyMakeBorder(const Mat& src, Mat& dst, int top, int bottom, int left, int right,
                    int borderType, const Scalar& borderValue)
{
    dst.create(src.rows + top + bottom, src.cols + left + right, src.type());
    int i, j, k, esz = (int)src.elemSize();
    int width = src.cols*esz, width1 = dst.cols*esz;

    if( borderType == BORDER_CONSTANT )
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
        vector<int> tabvec((left + right)*esz + 1);
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
           double* _minval, double* _maxval,
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
            if( minval > val || !minpos )
            {
                minval = val;
                minpos = startidx + i;
            }
            if( maxval < val || !maxpos )
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
            if( (minval > val || !minpos) && mask[i] )
            {
                minval = val;
                minpos = startidx + i;
            }
            if( (maxval < val || !maxpos) && mask[i] )
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
    if( idx > 0 )
    {
        idx--;
        for( int i = mtx.dims-1; i >= 0; i-- )
        {
            int sz = mtx.size[i]*(i == mtx.dims-1 ? mtx.channels() : 1);
            pos[i] = (int)(idx % sz);
            idx /= sz;
        }
    }
    else
    {
        for( int i = mtx.dims-1; i >= 0; i-- )
            pos[i] = -1;
    }
}

void minMaxLoc(const Mat& src, double* _minval, double* _maxval,
               vector<int>* _minloc, vector<int>* _maxloc,
               const Mat& mask)
{
    CV_Assert( src.channels() == 1 );
    const Mat *arrays[]={&src, &mask, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t startidx = 1, total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth();
    double minval = 0;
    double maxval = 0;
    size_t maxidx = 0, minidx = 0;

    for( i = 0; i < nplanes; i++, ++it, startidx += total )
    {
        const uchar* sptr = planes[0].ptr();
        const uchar* mptr = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            minMaxLoc_((const uchar*)sptr, total, startidx,
                       &minval, &maxval, &minidx, &maxidx, mptr);
            break;
        case CV_8S:
            minMaxLoc_((const schar*)sptr, total, startidx,
                       &minval, &maxval, &minidx, &maxidx, mptr);
            break;
        case CV_16U:
            minMaxLoc_((const ushort*)sptr, total, startidx,
                       &minval, &maxval, &minidx, &maxidx, mptr);
            break;
        case CV_16S:
            minMaxLoc_((const short*)sptr, total, startidx,
                       &minval, &maxval, &minidx, &maxidx, mptr);
            break;
        case CV_32S:
            minMaxLoc_((const int*)sptr, total, startidx,
                       &minval, &maxval, &minidx, &maxidx, mptr);
            break;
        case CV_32F:
            minMaxLoc_((const float*)sptr, total, startidx,
                       &minval, &maxval, &minidx, &maxidx, mptr);
            break;
        case CV_64F:
            minMaxLoc_((const double*)sptr, total, startidx,
                       &minval, &maxval, &minidx, &maxidx, mptr);
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


static int
normHamming(const uchar* src, size_t total, int cellSize)
{
    int result = 0;
    int mask = cellSize == 1 ? 1 : cellSize == 2 ? 3 : cellSize == 4 ? 15 : -1;
    CV_Assert( mask >= 0 );

    for( size_t i = 0; i < total; i++ )
    {
        unsigned a = src[i];
        for( ; a != 0; a >>= cellSize )
            result += (a & mask) != 0;
    }
    return result;
}


template<typename _Tp> static double
norm_(const _Tp* src, size_t total, int cn, int normType, double startval, const uchar* mask)
{
    size_t i;
    double result = startval;
    if( !mask )
        total *= cn;

    if( normType == NORM_INF )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result = std::max(result, (double)std::abs(0+src[i]));// trick with 0 used to quiet gcc warning
        else
            for( int c = 0; c < cn; c++ )
            {
                for( i = 0; i < total; i++ )
                    if( mask[i] )
                        result = std::max(result, (double)std::abs(0+src[i*cn + c]));
            }
    }
    else if( normType == NORM_L1 )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result += std::abs(0+src[i]);
        else
            for( int c = 0; c < cn; c++ )
            {
                for( i = 0; i < total; i++ )
                    if( mask[i] )
                        result += std::abs(0+src[i*cn + c]);
            }
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
            for( int c = 0; c < cn; c++ )
            {
                for( i = 0; i < total; i++ )
                    if( mask[i] )
                    {
                        double v = src[i*cn + c];
                        result += v*v;
                    }
            }
    }
    return result;
}


template<typename _Tp> static double
norm_(const _Tp* src1, const _Tp* src2, size_t total, int cn, int normType, double startval, const uchar* mask)
{
    size_t i;
    double result = startval;
    if( !mask )
        total *= cn;

    if( normType == NORM_INF )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result = std::max(result, (double)std::abs(src1[i] - src2[i]));
        else
            for( int c = 0; c < cn; c++ )
            {
                for( i = 0; i < total; i++ )
                    if( mask[i] )
                        result = std::max(result, (double)std::abs(src1[i*cn + c] - src2[i*cn + c]));
            }
    }
    else if( normType == NORM_L1 )
    {
        if( !mask )
            for( i = 0; i < total; i++ )
                result += std::abs(src1[i] - src2[i]);
        else
            for( int c = 0; c < cn; c++ )
            {
                for( i = 0; i < total; i++ )
                    if( mask[i] )
                        result += std::abs(src1[i*cn + c] - src2[i*cn + c]);
            }
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
            for( int c = 0; c < cn; c++ )
            {
                for( i = 0; i < total; i++ )
                    if( mask[i] )
                    {
                        double v = src1[i*cn + c] - src2[i*cn + c];
                        result += v*v;
                    }
            }
    }
    return result;
}


double norm(InputArray _src, int normType, InputArray _mask)
{
    Mat src = _src.getMat(), mask = _mask.getMat();
    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        if( !mask.empty() )
        {
            Mat temp;
            bitwise_and(src, mask, temp);
            return cvtest::norm(temp, normType, Mat());
        }

        CV_Assert( src.depth() == CV_8U );

        const Mat *arrays[]={&src, 0};
        Mat planes[1];

        NAryMatIterator it(arrays, planes);
        size_t total = planes[0].total();
        size_t i, nplanes = it.nplanes;
        double result = 0;
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        for( i = 0; i < nplanes; i++, ++it )
            result += normHamming(planes[0].ptr(), total, cellSize);
        return result;
    }
    int normType0 = normType;
    normType = normType == NORM_L2SQR ? NORM_L2 : normType;

    CV_Assert( mask.empty() || (src.size == mask.size && mask.type() == CV_8U) );
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    const Mat *arrays[]={&src, &mask, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth(), cn = planes[0].channels();
    double result = 0;

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        const uchar* mptr = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            result = norm_((const uchar*)sptr, total, cn, normType, result, mptr);
            break;
        case CV_8S:
            result = norm_((const schar*)sptr, total, cn, normType, result, mptr);
            break;
        case CV_16U:
            result = norm_((const ushort*)sptr, total, cn, normType, result, mptr);
            break;
        case CV_16S:
            result = norm_((const short*)sptr, total, cn, normType, result, mptr);
            break;
        case CV_32S:
            result = norm_((const int*)sptr, total, cn, normType, result, mptr);
            break;
        case CV_32F:
            result = norm_((const float*)sptr, total, cn, normType, result, mptr);
            break;
        case CV_64F:
            result = norm_((const double*)sptr, total, cn, normType, result, mptr);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        };
    }
    if( normType0 == NORM_L2 )
        result = sqrt(result);
    return result;
}


double norm(InputArray _src1, InputArray _src2, int normType, InputArray _mask)
{
    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();
    bool isRelative = (normType & NORM_RELATIVE) != 0;
    normType &= ~NORM_RELATIVE;

    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        Mat temp;
        bitwise_xor(src1, src2, temp);
        if( !mask.empty() )
            bitwise_and(temp, mask, temp);

        CV_Assert( temp.depth() == CV_8U );

        const Mat *arrays[]={&temp, 0};
        Mat planes[1];

        NAryMatIterator it(arrays, planes);
        size_t total = planes[0].total();
        size_t i, nplanes = it.nplanes;
        double result = 0;
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        for( i = 0; i < nplanes; i++, ++it )
            result += normHamming(planes[0].ptr(), total, cellSize);
        return result;
    }
    int normType0 = normType;
    normType = normType == NORM_L2SQR ? NORM_L2 : normType;

    CV_Assert( src1.type() == src2.type() && src1.size == src2.size );
    CV_Assert( mask.empty() || (src1.size == mask.size && mask.type() == CV_8U) );
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );
    const Mat *arrays[]={&src1, &src2, &mask, 0};
    Mat planes[3];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src1.depth(), cn = planes[0].channels();
    double result = 0;

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();
        const uchar* mptr = planes[2].ptr();

        switch( depth )
        {
        case CV_8U:
            result = norm_((const uchar*)sptr1, (const uchar*)sptr2, total, cn, normType, result, mptr);
            break;
        case CV_8S:
            result = norm_((const schar*)sptr1, (const schar*)sptr2, total, cn, normType, result, mptr);
            break;
        case CV_16U:
            result = norm_((const ushort*)sptr1, (const ushort*)sptr2, total, cn, normType, result, mptr);
            break;
        case CV_16S:
            result = norm_((const short*)sptr1, (const short*)sptr2, total, cn, normType, result, mptr);
            break;
        case CV_32S:
            result = norm_((const int*)sptr1, (const int*)sptr2, total, cn, normType, result, mptr);
            break;
        case CV_32F:
            result = norm_((const float*)sptr1, (const float*)sptr2, total, cn, normType, result, mptr);
            break;
        case CV_64F:
            result = norm_((const double*)sptr1, (const double*)sptr2, total, cn, normType, result, mptr);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        };
    }
    if( normType0 == NORM_L2 )
        result = sqrt(result);
    return isRelative ? result / (cvtest::norm(src2, normType) + DBL_EPSILON) : result;
}

double PSNR(InputArray _src1, InputArray _src2)
{
    CV_Assert( _src1.depth() == CV_8U );
    double diff = std::sqrt(cvtest::norm(_src1, _src2, NORM_L2SQR)/(_src1.total()*_src1.channels()));
    return 20*log10(255./(diff+DBL_EPSILON));
}

template<typename _Tp> static double
crossCorr_(const _Tp* src1, const _Tp* src2, size_t total)
{
    double result = 0;
    for( size_t i = 0; i < total; i++ )
        result += (double)src1[i]*src2[i];
    return result;
}

double crossCorr(const Mat& src1, const Mat& src2)
{
    CV_Assert( src1.size == src2.size && src1.type() == src2.type() );
    const Mat *arrays[]={&src1, &src2, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].channels();
    size_t i, nplanes = it.nplanes;
    int depth = src1.depth();
    double result = 0;

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            result += crossCorr_((const uchar*)sptr1, (const uchar*)sptr2, total);
            break;
        case CV_8S:
            result += crossCorr_((const schar*)sptr1, (const schar*)sptr2, total);
            break;
        case CV_16U:
            result += crossCorr_((const ushort*)sptr1, (const ushort*)sptr2, total);
            break;
        case CV_16S:
            result += crossCorr_((const short*)sptr1, (const short*)sptr2, total);
            break;
        case CV_32S:
            result += crossCorr_((const int*)sptr1, (const int*)sptr2, total);
            break;
        case CV_32F:
            result += crossCorr_((const float*)sptr1, (const float*)sptr2, total);
            break;
        case CV_64F:
            result += crossCorr_((const double*)sptr1, (const double*)sptr2, total);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        };
    }
    return result;
}


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
            size_t sz = MIN(total - i, blockSize);
            for( j = 0; j < sz; j++ )
                dst[j] = src[j] & scalar[j];
        }
    else if( c == '|' )
        for( i = 0; i < total; i += blockSize, dst += blockSize, src += blockSize )
        {
            size_t sz = MIN(total - i, blockSize);
            for( j = 0; j < sz; j++ )
                dst[j] = src[j] | scalar[j];
        }
    else if( c == '^' )
    {
        for( i = 0; i < total; i += blockSize, dst += blockSize, src += blockSize )
        {
            size_t sz = MIN(total - i, blockSize);
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
    const Mat *arrays[]={&src1, &src2, &dst, 0};
    Mat planes[3];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].elemSize();
    size_t i, nplanes = it.nplanes;

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();
        uchar* dptr = planes[2].ptr();

        logicOp_(sptr1, sptr2, dptr, total, op);
    }
}


void logicOp(const Mat& src, const Scalar& s, Mat& dst, char op)
{
    CV_Assert( op == '&' || op == '|' || op == '^' || op == '~' );
    dst.create( src.dims, &src.size[0], src.type() );
    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].elemSize();
    size_t i, nplanes = it.nplanes;
    double buf[12];
    scalarToRawData(s, buf, src.type(), (int)(96/planes[0].elemSize1()));

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

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
        CV_Error(Error::StsBadArg, "Unknown comparison operation");
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
        CV_Error(Error::StsBadArg, "Unknown comparison operation");
    }
}


void compare(const Mat& src1, const Mat& src2, Mat& dst, int cmpop)
{
    CV_Assert( src1.type() == src2.type() && src1.channels() == 1 && src1.size == src2.size );
    dst.create( src1.dims, &src1.size[0], CV_8U );
    const Mat *arrays[]={&src1, &src2, &dst, 0};
    Mat planes[3];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src1.depth();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();
        uchar* dptr = planes[2].ptr();

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
            CV_Error(Error::StsUnsupportedFormat, "");
        }
    }
}

void compare(const Mat& src, double value, Mat& dst, int cmpop)
{
    CV_Assert( src.channels() == 1 );
    dst.create( src.dims, &src.size[0], CV_8U );
    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth();
    int ivalue = saturate_cast<int>(value);

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

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
            CV_Error(Error::StsUnsupportedFormat, "");
        }
    }
}


template<typename _Tp> double
cmpUlpsInt_(const _Tp* src1, const _Tp* src2, size_t total, int imaxdiff,
           size_t startidx, size_t& idx)
{
    size_t i;
    int realmaxdiff = 0;
    for( i = 0; i < total; i++ )
    {
        int diff = std::abs(src1[i] - src2[i]);
        if( realmaxdiff < diff )
        {
            realmaxdiff = diff;
            if( diff > imaxdiff && idx == 0 )
                idx = i + startidx;
        }
    }
    return realmaxdiff;
}


template<> double cmpUlpsInt_<int>(const int* src1, const int* src2,
                                          size_t total, int imaxdiff,
                                          size_t startidx, size_t& idx)
{
    size_t i;
    double realmaxdiff = 0;
    for( i = 0; i < total; i++ )
    {
        double diff = fabs((double)src1[i] - (double)src2[i]);
        if( realmaxdiff < diff )
        {
            realmaxdiff = diff;
            if( diff > imaxdiff && idx == 0 )
                idx = i + startidx;
        }
    }
    return realmaxdiff;
}


static double
cmpUlpsFlt_(const int* src1, const int* src2, size_t total, int imaxdiff, size_t startidx, size_t& idx)
{
    const int C = 0x7fffffff;
    int realmaxdiff = 0;
    size_t i;
    for( i = 0; i < total; i++ )
    {
        int a = src1[i], b = src2[i];
        if( a < 0 ) a ^= C;
        if( b < 0 ) b ^= C;
        int diff = std::abs(a - b);
        if( realmaxdiff < diff )
        {
            realmaxdiff = diff;
            if( diff > imaxdiff && idx == 0 )
                idx = i + startidx;
        }
    }
    return realmaxdiff;
}


static double
cmpUlpsFlt_(const int64* src1, const int64* src2, size_t total, int imaxdiff, size_t startidx, size_t& idx)
{
    const int64 C = CV_BIG_INT(0x7fffffffffffffff);
    double realmaxdiff = 0;
    size_t i;
    for( i = 0; i < total; i++ )
    {
        int64 a = src1[i], b = src2[i];
        if( a < 0 ) a ^= C;
        if( b < 0 ) b ^= C;
        double diff = fabs((double)a - (double)b);
        if( realmaxdiff < diff )
        {
            realmaxdiff = diff;
            if( diff > imaxdiff && idx == 0 )
                idx = i + startidx;
        }
    }
    return realmaxdiff;
}

bool cmpUlps(const Mat& src1, const Mat& src2, int imaxDiff, double* _realmaxdiff, vector<int>* loc)
{
    CV_Assert( src1.type() == src2.type() && src1.size == src2.size );
    const Mat *arrays[]={&src1, &src2, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].channels();
    size_t i, nplanes = it.nplanes;
    int depth = src1.depth();
    size_t startidx = 1, idx = 0;
    if(_realmaxdiff)
        *_realmaxdiff = 0;

    for( i = 0; i < nplanes; i++, ++it, startidx += total )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();
        double realmaxdiff = 0;

        switch( depth )
        {
        case CV_8U:
            realmaxdiff = cmpUlpsInt_((const uchar*)sptr1, (const uchar*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_8S:
            realmaxdiff = cmpUlpsInt_((const schar*)sptr1, (const schar*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_16U:
            realmaxdiff = cmpUlpsInt_((const ushort*)sptr1, (const ushort*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_16S:
            realmaxdiff = cmpUlpsInt_((const short*)sptr1, (const short*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_32S:
            realmaxdiff = cmpUlpsInt_((const int*)sptr1, (const int*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_32F:
            realmaxdiff = cmpUlpsFlt_((const int*)sptr1, (const int*)sptr2, total, imaxDiff, startidx, idx);
            break;
        case CV_64F:
            realmaxdiff = cmpUlpsFlt_((const int64*)sptr1, (const int64*)sptr2, total, imaxDiff, startidx, idx);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        }

        if(_realmaxdiff)
            *_realmaxdiff = std::max(*_realmaxdiff, realmaxdiff);
    }
    if(idx > 0 && loc)
        setpos(src1, *loc, idx);
    return idx == 0;
}


template<typename _Tp> static void
checkInt_(const _Tp* a, size_t total, int imin, int imax, size_t startidx, size_t& idx)
{
    for( size_t i = 0; i < total; i++ )
    {
        int val = a[i];
        if( val < imin || val > imax )
        {
            idx = i + startidx;
            break;
        }
    }
}


template<typename _Tp> static void
checkFlt_(const _Tp* a, size_t total, double fmin, double fmax, size_t startidx, size_t& idx)
{
    for( size_t i = 0; i < total; i++ )
    {
        double val = a[i];
        if( cvIsNaN(val) || cvIsInf(val) || val < fmin || val > fmax )
        {
            idx = i + startidx;
            break;
        }
    }
}


// checks that the array does not have NaNs and/or Infs and all the elements are
// within [min_val,max_val). idx is the index of the first "bad" element.
int check( const Mat& a, double fmin, double fmax, vector<int>* _idx )
{
    const Mat *arrays[]={&a, 0};
    Mat plane;
    NAryMatIterator it(arrays, &plane);
    size_t total = plane.total()*plane.channels();
    size_t i, nplanes = it.nplanes;
    int depth = a.depth();
    size_t startidx = 1, idx = 0;
    int imin = 0, imax = 0;

    if( depth <= CV_32S )
    {
        imin = cvCeil(fmin);
        imax = cvFloor(fmax);
    }

    for( i = 0; i < nplanes; i++, ++it, startidx += total )
    {
        const uchar* aptr = plane.ptr();

        switch( depth )
        {
            case CV_8U:
                checkInt_((const uchar*)aptr, total, imin, imax, startidx, idx);
                break;
            case CV_8S:
                checkInt_((const schar*)aptr, total, imin, imax, startidx, idx);
                break;
            case CV_16U:
                checkInt_((const ushort*)aptr, total, imin, imax, startidx, idx);
                break;
            case CV_16S:
                checkInt_((const short*)aptr, total, imin, imax, startidx, idx);
                break;
            case CV_32S:
                checkInt_((const int*)aptr, total, imin, imax, startidx, idx);
                break;
            case CV_32F:
                checkFlt_((const float*)aptr, total, fmin, fmax, startidx, idx);
                break;
            case CV_64F:
                checkFlt_((const double*)aptr, total, fmin, fmax, startidx, idx);
                break;
            default:
                CV_Error(Error::StsUnsupportedFormat, "");
        }

        if( idx != 0 )
            break;
    }

    if(idx != 0 && _idx)
        setpos(a, *_idx, idx);
    return idx == 0 ? 0 : -1;
}

#define CMP_EPS_OK 0
#define CMP_EPS_BIG_DIFF -1
#define CMP_EPS_INVALID_TEST_DATA -2 // there is NaN or Inf value in test data
#define CMP_EPS_INVALID_REF_DATA -3 // there is NaN or Inf value in reference data

// compares two arrays. max_diff is the maximum actual difference,
// success_err_level is maximum allowed difference, idx is the index of the first
// element for which difference is >success_err_level
// (or index of element with the maximum difference)
int cmpEps( const Mat& arr, const Mat& refarr, double* _realmaxdiff,
            double success_err_level, vector<int>* _idx,
            bool element_wise_relative_error )
{
    CV_Assert( arr.type() == refarr.type() && arr.size == refarr.size );

    int ilevel = refarr.depth() <= CV_32S ? cvFloor(success_err_level) : 0;
    int result = CMP_EPS_OK;

    const Mat *arrays[]={&arr, &refarr, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].channels(), j = total;
    size_t i, nplanes = it.nplanes;
    int depth = arr.depth();
    size_t startidx = 1, idx = 0;
    double realmaxdiff = 0, maxval = 0;

    if(_realmaxdiff)
        *_realmaxdiff = 0;

    if( refarr.depth() >= CV_32F && !element_wise_relative_error )
    {
        maxval = cvtest::norm( refarr, NORM_INF );
        maxval = MAX(maxval, 1.);
    }

    for( i = 0; i < nplanes; i++, ++it, startidx += total )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            realmaxdiff = cmpUlpsInt_((const uchar*)sptr1, (const uchar*)sptr2, total, ilevel, startidx, idx);
            break;
        case CV_8S:
            realmaxdiff = cmpUlpsInt_((const schar*)sptr1, (const schar*)sptr2, total, ilevel, startidx, idx);
            break;
        case CV_16U:
            realmaxdiff = cmpUlpsInt_((const ushort*)sptr1, (const ushort*)sptr2, total, ilevel, startidx, idx);
            break;
        case CV_16S:
            realmaxdiff = cmpUlpsInt_((const short*)sptr1, (const short*)sptr2, total, ilevel, startidx, idx);
            break;
        case CV_32S:
            realmaxdiff = cmpUlpsInt_((const int*)sptr1, (const int*)sptr2, total, ilevel, startidx, idx);
            break;
        case CV_32F:
            for( j = 0; j < total; j++ )
            {
                double a_val = ((float*)sptr1)[j];
                double b_val = ((float*)sptr2)[j];
                double threshold;
                if( ((int*)sptr1)[j] == ((int*)sptr2)[j] )
                    continue;
                if( cvIsNaN(a_val) || cvIsInf(a_val) )
                {
                    result = CMP_EPS_INVALID_TEST_DATA;
                    idx = startidx + j;
                    break;
                }
                if( cvIsNaN(b_val) || cvIsInf(b_val) )
                {
                    result = CMP_EPS_INVALID_REF_DATA;
                    idx = startidx + j;
                    break;
                }
                a_val = fabs(a_val - b_val);
                threshold = element_wise_relative_error ? fabs(b_val) + 1 : maxval;
                if( a_val > threshold*success_err_level )
                {
                    realmaxdiff = a_val/threshold;
                    if( idx == 0 )
                        idx = startidx + j;
                    break;
                }
            }
            break;
        case CV_64F:
            for( j = 0; j < total; j++ )
            {
                double a_val = ((double*)sptr1)[j];
                double b_val = ((double*)sptr2)[j];
                double threshold;
                if( ((int64*)sptr1)[j] == ((int64*)sptr2)[j] )
                    continue;
                if( cvIsNaN(a_val) || cvIsInf(a_val) )
                {
                    result = CMP_EPS_INVALID_TEST_DATA;
                    idx = startidx + j;
                    break;
                }
                if( cvIsNaN(b_val) || cvIsInf(b_val) )
                {
                    result = CMP_EPS_INVALID_REF_DATA;
                    idx = startidx + j;
                    break;
                }
                a_val = fabs(a_val - b_val);
                threshold = element_wise_relative_error ? fabs(b_val) + 1 : maxval;
                if( a_val > threshold*success_err_level )
                {
                    realmaxdiff = a_val/threshold;
                    idx = startidx + j;
                    break;
                }
            }
            break;
        default:
            assert(0);
            return CMP_EPS_BIG_DIFF;
        }
        if(_realmaxdiff)
            *_realmaxdiff = MAX(*_realmaxdiff, realmaxdiff);
        if( idx != 0 )
            break;
    }

    if( result == 0 && idx != 0 )
        result = CMP_EPS_BIG_DIFF;

    if( result < -1 && _realmaxdiff )
        *_realmaxdiff = exp(1000.);
    if(idx > 0 && _idx)
        setpos(arr, *_idx, idx);

    return result;
}


int cmpEps2( TS* ts, const Mat& a, const Mat& b, double success_err_level,
             bool element_wise_relative_error, const char* desc )
{
    char msg[100];
    double diff = 0;
    vector<int> idx;
    int code = cmpEps( a, b, &diff, success_err_level, &idx, element_wise_relative_error );

    switch( code )
    {
    case CMP_EPS_BIG_DIFF:
        sprintf( msg, "%s: Too big difference (=%g)", desc, diff );
        code = TS::FAIL_BAD_ACCURACY;
        break;
    case CMP_EPS_INVALID_TEST_DATA:
        sprintf( msg, "%s: Invalid output", desc );
        code = TS::FAIL_INVALID_OUTPUT;
        break;
    case CMP_EPS_INVALID_REF_DATA:
        sprintf( msg, "%s: Invalid reference output", desc );
        code = TS::FAIL_INVALID_OUTPUT;
        break;
    default:
        ;
    }

    if( code < 0 )
    {
        if( a.total() == 1 )
        {
            ts->printf( TS::LOG, "%s\n", msg );
        }
        else if( a.dims == 2 && (a.rows == 1 || a.cols == 1) )
        {
            ts->printf( TS::LOG, "%s at element %d\n", msg, idx[0] + idx[1] );
        }
        else
        {
            string idxstr = vec2str(", ", &idx[0], idx.size());
            ts->printf( TS::LOG, "%s at (%s)\n", msg, idxstr.c_str() );
        }
    }

    return code;
}


int cmpEps2_64f( TS* ts, const double* val, const double* refval, int len,
             double eps, const char* param_name )
{
    Mat _val(1, len, CV_64F, (void*)val);
    Mat _refval(1, len, CV_64F, (void*)refval);

    return cmpEps2( ts, _val, _refval, eps, true, param_name );
}


template<typename _Tp> static void
GEMM_(const _Tp* a_data0, int a_step, int a_delta,
      const _Tp* b_data0, int b_step, int b_delta,
      const _Tp* c_data0, int c_step, int c_delta,
      _Tp* d_data, int d_step,
      int d_rows, int d_cols, int a_cols, int cn,
      double alpha, double beta)
{
    for( int i = 0; i < d_rows; i++, d_data += d_step, c_data0 += c_step, a_data0 += a_step )
    {
        for( int j = 0; j < d_cols; j++ )
        {
            const _Tp* a_data = a_data0;
            const _Tp* b_data = b_data0 + j*b_delta;
            const _Tp* c_data = c_data0 + j*c_delta;

            if( cn == 1 )
            {
                double s = 0;
                for( int k = 0; k < a_cols; k++ )
                {
                    s += ((double)a_data[0])*b_data[0];
                    a_data += a_delta;
                    b_data += b_step;
                }
                d_data[j] = (_Tp)(s*alpha + (c_data ? c_data[0]*beta : 0));
            }
            else
            {
                double s_re = 0, s_im = 0;

                for( int k = 0; k < a_cols; k++ )
                {
                    s_re += ((double)a_data[0])*b_data[0] - ((double)a_data[1])*b_data[1];
                    s_im += ((double)a_data[0])*b_data[1] + ((double)a_data[1])*b_data[0];
                    a_data += a_delta;
                    b_data += b_step;
                }

                s_re *= alpha;
                s_im *= alpha;

                if( c_data )
                {
                    s_re += c_data[0]*beta;
                    s_im += c_data[1]*beta;
                }

                d_data[j*2] = (_Tp)s_re;
                d_data[j*2+1] = (_Tp)s_im;
            }
        }
    }
}


void gemm( const Mat& _a, const Mat& _b, double alpha,
           const Mat& _c, double beta, Mat& d, int flags )
{
    Mat a = _a, b = _b, c = _c;

    if( a.data == d.data )
        a = a.clone();

    if( b.data == d.data )
        b = b.clone();

    if( !c.empty() && c.data == d.data && (flags & cv::GEMM_3_T) )
        c = c.clone();

    int a_rows = a.rows, a_cols = a.cols, b_rows = b.rows, b_cols = b.cols;
    int cn = a.channels();
    int a_step = (int)a.step1(), a_delta = cn;
    int b_step = (int)b.step1(), b_delta = cn;
    int c_rows = 0, c_cols = 0, c_step = 0, c_delta = 0;

    CV_Assert( a.type() == b.type() && a.dims == 2 && b.dims == 2 && cn <= 2 );

    if( flags & cv::GEMM_1_T )
    {
        std::swap( a_rows, a_cols );
        std::swap( a_step, a_delta );
    }

    if( flags & cv::GEMM_2_T )
    {
        std::swap( b_rows, b_cols );
        std::swap( b_step, b_delta );
    }

    if( !c.empty() )
    {
        c_rows = c.rows;
        c_cols = c.cols;
        c_step = (int)c.step1();
        c_delta = cn;

        if( flags & cv::GEMM_3_T )
        {
            std::swap( c_rows, c_cols );
            std::swap( c_step, c_delta );
        }

        CV_Assert( c.dims == 2 && c.type() == a.type() && c_rows == a_rows && c_cols == b_cols );
    }

    d.create(a_rows, b_cols, a.type());

    if( a.depth() == CV_32F )
        GEMM_(a.ptr<float>(), a_step, a_delta, b.ptr<float>(), b_step, b_delta,
              !c.empty() ? c.ptr<float>() : 0, c_step, c_delta, d.ptr<float>(),
              (int)d.step1(), a_rows, b_cols, a_cols, cn, alpha, beta );
    else
        GEMM_(a.ptr<double>(), a_step, a_delta, b.ptr<double>(), b_step, b_delta,
              !c.empty() ? c.ptr<double>() : 0, c_step, c_delta, d.ptr<double>(),
              (int)d.step1(), a_rows, b_cols, a_cols, cn, alpha, beta );
}


template<typename _Tp> static void
transform_(const _Tp* sptr, _Tp* dptr, size_t total, int scn, int dcn, const double* mat)
{
    for( size_t i = 0; i < total; i++, sptr += scn, dptr += dcn )
    {
        for( int j = 0; j < dcn; j++ )
        {
            double s = mat[j*(scn + 1) + scn];
            for( int k = 0; k < scn; k++ )
                s += mat[j*(scn + 1) + k]*sptr[k];
            dptr[j] = saturate_cast<_Tp>(s);
        }
    }
}


void transform( const Mat& src, Mat& dst, const Mat& transmat, const Mat& _shift )
{
    double mat[20];

    int scn = src.channels();
    int dcn = dst.channels();
    int depth = src.depth();
    int mattype = transmat.depth();
    Mat shift = _shift.reshape(1, 0);
    bool haveShift = !shift.empty();

    CV_Assert( scn <= 4 && dcn <= 4 &&
              (mattype == CV_32F || mattype == CV_64F) &&
              (!haveShift || (shift.type() == mattype && (shift.rows == 1 || shift.cols == 1))) );

    // prepare cn x (cn + 1) transform matrix
    if( mattype == CV_32F )
    {
        for( int i = 0; i < transmat.rows; i++ )
        {
            mat[i*(scn+1)+scn] = 0.;
            for( int j = 0; j < transmat.cols; j++ )
                mat[i*(scn+1)+j] = transmat.at<float>(i,j);
            if( haveShift )
                mat[i*(scn+1)+scn] = shift.at<float>(i);
        }
    }
    else
    {
        for( int i = 0; i < transmat.rows; i++ )
        {
            mat[i*(scn+1)+scn] = 0.;
            for( int j = 0; j < transmat.cols; j++ )
                mat[i*(scn+1)+j] = transmat.at<double>(i,j);
            if( haveShift )
                mat[i*(scn+1)+scn] = shift.at<double>(i);
        }
    }

    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            transform_((const uchar*)sptr, (uchar*)dptr, total, scn, dcn, mat);
            break;
        case CV_8S:
            transform_((const schar*)sptr, (schar*)dptr, total, scn, dcn, mat);
            break;
        case CV_16U:
            transform_((const ushort*)sptr, (ushort*)dptr, total, scn, dcn, mat);
            break;
        case CV_16S:
            transform_((const short*)sptr, (short*)dptr, total, scn, dcn, mat);
            break;
        case CV_32S:
            transform_((const int*)sptr, (int*)dptr, total, scn, dcn, mat);
            break;
        case CV_32F:
            transform_((const float*)sptr, (float*)dptr, total, scn, dcn, mat);
            break;
        case CV_64F:
            transform_((const double*)sptr, (double*)dptr, total, scn, dcn, mat);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        }
    }
}

template<typename _Tp> static void
minmax_(const _Tp* src1, const _Tp* src2, _Tp* dst, size_t total, char op)
{
    if( op == 'M' )
        for( size_t i = 0; i < total; i++ )
            dst[i] = std::max(src1[i], src2[i]);
    else
        for( size_t i = 0; i < total; i++ )
            dst[i] = std::min(src1[i], src2[i]);
}

static void minmax(const Mat& src1, const Mat& src2, Mat& dst, char op)
{
    dst.create(src1.dims, src1.size, src1.type());
    CV_Assert( src1.type() == src2.type() && src1.size == src2.size );
    const Mat *arrays[]={&src1, &src2, &dst, 0};
    Mat planes[3];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].channels();
    size_t i, nplanes = it.nplanes, depth = src1.depth();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();
        uchar* dptr = planes[2].ptr();

        switch( depth )
        {
        case CV_8U:
            minmax_((const uchar*)sptr1, (const uchar*)sptr2, (uchar*)dptr, total, op);
            break;
        case CV_8S:
            minmax_((const schar*)sptr1, (const schar*)sptr2, (schar*)dptr, total, op);
            break;
        case CV_16U:
            minmax_((const ushort*)sptr1, (const ushort*)sptr2, (ushort*)dptr, total, op);
            break;
        case CV_16S:
            minmax_((const short*)sptr1, (const short*)sptr2, (short*)dptr, total, op);
            break;
        case CV_32S:
            minmax_((const int*)sptr1, (const int*)sptr2, (int*)dptr, total, op);
            break;
        case CV_32F:
            minmax_((const float*)sptr1, (const float*)sptr2, (float*)dptr, total, op);
            break;
        case CV_64F:
            minmax_((const double*)sptr1, (const double*)sptr2, (double*)dptr, total, op);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        }
    }
}


void min(const Mat& src1, const Mat& src2, Mat& dst)
{
    minmax( src1, src2, dst, 'm' );
}

void max(const Mat& src1, const Mat& src2, Mat& dst)
{
    minmax( src1, src2, dst, 'M' );
}


template<typename _Tp> static void
minmax_(const _Tp* src1, _Tp val, _Tp* dst, size_t total, char op)
{
    if( op == 'M' )
        for( size_t i = 0; i < total; i++ )
            dst[i] = std::max(src1[i], val);
    else
        for( size_t i = 0; i < total; i++ )
            dst[i] = std::min(src1[i], val);
}

static void minmax(const Mat& src1, double val, Mat& dst, char op)
{
    dst.create(src1.dims, src1.size, src1.type());
    const Mat *arrays[]={&src1, &dst, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].channels();
    size_t i, nplanes = it.nplanes, depth = src1.depth();
    int ival = saturate_cast<int>(val);

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            minmax_((const uchar*)sptr1, saturate_cast<uchar>(ival), (uchar*)dptr, total, op);
            break;
        case CV_8S:
            minmax_((const schar*)sptr1, saturate_cast<schar>(ival), (schar*)dptr, total, op);
            break;
        case CV_16U:
            minmax_((const ushort*)sptr1, saturate_cast<ushort>(ival), (ushort*)dptr, total, op);
            break;
        case CV_16S:
            minmax_((const short*)sptr1, saturate_cast<short>(ival), (short*)dptr, total, op);
            break;
        case CV_32S:
            minmax_((const int*)sptr1, saturate_cast<int>(ival), (int*)dptr, total, op);
            break;
        case CV_32F:
            minmax_((const float*)sptr1, saturate_cast<float>(val), (float*)dptr, total, op);
            break;
        case CV_64F:
            minmax_((const double*)sptr1, saturate_cast<double>(val), (double*)dptr, total, op);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        }
    }
}


void min(const Mat& src1, double val, Mat& dst)
{
    minmax( src1, val, dst, 'm' );
}

void max(const Mat& src1, double val, Mat& dst)
{
    minmax( src1, val, dst, 'M' );
}


template<typename _Tp> static void
muldiv_(const _Tp* src1, const _Tp* src2, _Tp* dst, size_t total, double scale, char op)
{
    if( op == '*' )
        for( size_t i = 0; i < total; i++ )
            dst[i] = saturate_cast<_Tp>((scale*src1[i])*src2[i]);
    else if( src1 )
        for( size_t i = 0; i < total; i++ )
            dst[i] = src2[i] ? saturate_cast<_Tp>((scale*src1[i])/src2[i]) : 0;
    else
        for( size_t i = 0; i < total; i++ )
            dst[i] = src2[i] ? saturate_cast<_Tp>(scale/src2[i]) : 0;
}

static void muldiv(const Mat& src1, const Mat& src2, Mat& dst, double scale, char op)
{
    dst.create(src2.dims, src2.size, src2.type());
    CV_Assert( src1.empty() || (src1.type() == src2.type() && src1.size == src2.size) );
    const Mat *arrays[]={&src1, &src2, &dst, 0};
    Mat planes[3];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[1].total()*planes[1].channels();
    size_t i, nplanes = it.nplanes, depth = src2.depth();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();
        uchar* dptr = planes[2].ptr();

        switch( depth )
        {
        case CV_8U:
            muldiv_((const uchar*)sptr1, (const uchar*)sptr2, (uchar*)dptr, total, scale, op);
            break;
        case CV_8S:
            muldiv_((const schar*)sptr1, (const schar*)sptr2, (schar*)dptr, total, scale, op);
            break;
        case CV_16U:
            muldiv_((const ushort*)sptr1, (const ushort*)sptr2, (ushort*)dptr, total, scale, op);
            break;
        case CV_16S:
            muldiv_((const short*)sptr1, (const short*)sptr2, (short*)dptr, total, scale, op);
            break;
        case CV_32S:
            muldiv_((const int*)sptr1, (const int*)sptr2, (int*)dptr, total, scale, op);
            break;
        case CV_32F:
            muldiv_((const float*)sptr1, (const float*)sptr2, (float*)dptr, total, scale, op);
            break;
        case CV_64F:
            muldiv_((const double*)sptr1, (const double*)sptr2, (double*)dptr, total, scale, op);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        }
    }
}


void multiply(const Mat& src1, const Mat& src2, Mat& dst, double scale)
{
    muldiv( src1, src2, dst, scale, '*' );
}

void divide(const Mat& src1, const Mat& src2, Mat& dst, double scale)
{
    muldiv( src1, src2, dst, scale, '/' );
}


template<typename _Tp> static void
mean_(const _Tp* src, const uchar* mask, size_t total, int cn, Scalar& sum, int& nz)
{
    if( !mask )
    {
        nz += (int)total;
        total *= cn;
        for( size_t i = 0; i < total; i += cn )
        {
            for( int c = 0; c < cn; c++ )
                sum[c] += src[i + c];
        }
    }
    else
    {
        for( size_t i = 0; i < total; i++ )
            if( mask[i] )
            {
                nz++;
                for( int c = 0; c < cn; c++ )
                    sum[c] += src[i*cn + c];
            }
    }
}

Scalar mean(const Mat& src, const Mat& mask)
{
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size == src.size));
    Scalar sum;
    int nz = 0;

    const Mat *arrays[]={&src, &mask, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth(), cn = src.channels();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        const uchar* mptr = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            mean_((const uchar*)sptr, mptr, total, cn, sum, nz);
            break;
        case CV_8S:
            mean_((const schar*)sptr, mptr, total, cn, sum, nz);
            break;
        case CV_16U:
            mean_((const ushort*)sptr, mptr, total, cn, sum, nz);
            break;
        case CV_16S:
            mean_((const short*)sptr, mptr, total, cn, sum, nz);
            break;
        case CV_32S:
            mean_((const int*)sptr, mptr, total, cn, sum, nz);
            break;
        case CV_32F:
            mean_((const float*)sptr, mptr, total, cn, sum, nz);
            break;
        case CV_64F:
            mean_((const double*)sptr, mptr, total, cn, sum, nz);
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "");
        }
    }

    return sum * (1./std::max(nz, 1));
}


void  patchZeros( Mat& mat, double level )
{
    int j, ncols = mat.cols * mat.channels();
    int depth = mat.depth();
    CV_Assert( depth == CV_32F || depth == CV_64F );

    for( int i = 0; i < mat.rows; i++ )
    {
        if( depth == CV_32F )
        {
            float* data = mat.ptr<float>(i);
            for( j = 0; j < ncols; j++ )
                if( fabs(data[j]) < level )
                    data[j] += 1;
        }
        else
        {
            double* data = mat.ptr<double>(i);
            for( j = 0; j < ncols; j++ )
                if( fabs(data[j]) < level )
                    data[j] += 1;
        }
    }
}


static void calcSobelKernel1D( int order, int _aperture_size, int size, vector<int>& kernel )
{
    int i, j, oldval, newval;
    kernel.resize(size + 1);

    if( _aperture_size < 0 )
    {
        static const int scharr[] = { 3, 10, 3, -1, 0, 1 };
        assert( size == 3 );
        for( i = 0; i < size; i++ )
            kernel[i] = scharr[order*3 + i];
        return;
    }

    for( i = 1; i <= size; i++ )
        kernel[i] = 0;
    kernel[0] = 1;

    for( i = 0; i < size - order - 1; i++ )
    {
        oldval = kernel[0];
        for( j = 1; j <= size; j++ )
        {
            newval = kernel[j] + kernel[j-1];
            kernel[j-1] = oldval;
            oldval = newval;
        }
    }

    for( i = 0; i < order; i++ )
    {
        oldval = -kernel[0];
        for( j = 1; j <= size; j++ )
        {
            newval = kernel[j-1] - kernel[j];
            kernel[j-1] = oldval;
            oldval = newval;
        }
    }
}


Mat calcSobelKernel2D( int dx, int dy, int _aperture_size, int origin )
{
    CV_Assert( (_aperture_size == -1 || (_aperture_size >= 1 && _aperture_size % 2 == 1)) &&
              dx >= 0 && dy >= 0 && dx + dy <= 3 );
    Size ksize = _aperture_size == -1 ? Size(3,3) : _aperture_size > 1 ?
        Size(_aperture_size, _aperture_size) : dx > 0 ? Size(3, 1) : Size(1, 3);

    Mat kernel(ksize, CV_32F);
    vector<int> kx, ky;

    calcSobelKernel1D( dx, _aperture_size, ksize.width, kx );
    calcSobelKernel1D( dy, _aperture_size, ksize.height, ky );

    for( int i = 0; i < kernel.rows; i++ )
    {
        float ay = (float)ky[i]*(origin && (dy & 1) ? -1 : 1);
        for( int j = 0; j < kernel.cols; j++ )
            kernel.at<float>(i, j) = kx[j]*ay;
    }

    return kernel;
}


Mat calcLaplaceKernel2D( int aperture_size )
{
    int ksize = aperture_size == 1 ? 3 : aperture_size;
    Mat kernel(ksize, ksize, CV_32F);

    vector<int> kx, ky;

    calcSobelKernel1D( 2, aperture_size, ksize, kx );
    if( aperture_size > 1 )
        calcSobelKernel1D( 0, aperture_size, ksize, ky );
    else
    {
        ky.resize(3);
        ky[0] = ky[2] = 0; ky[1] = 1;
    }

    for( int i = 0; i < ksize; i++ )
        for( int j = 0; j < ksize; j++ )
            kernel.at<float>(i, j) = (float)(kx[j]*ky[i] + kx[i]*ky[j]);

    return kernel;
}


void initUndistortMap( const Mat& _a0, const Mat& _k0, Size sz, Mat& _mapx, Mat& _mapy )
{
    _mapx.create(sz, CV_32F);
    _mapy.create(sz, CV_32F);

    double a[9], k[5]={0,0,0,0,0};
    Mat _a(3, 3, CV_64F, a);
    Mat _k(_k0.rows,_k0.cols, CV_MAKETYPE(CV_64F,_k0.channels()),k);
    double fx, fy, cx, cy, ifx, ify, cxn, cyn;

    _a0.convertTo(_a, CV_64F);
    _k0.convertTo(_k, CV_64F);
    fx = a[0]; fy = a[4]; cx = a[2]; cy = a[5];
    ifx = 1./fx; ify = 1./fy;
    cxn = cx;
    cyn = cy;

    for( int v = 0; v < sz.height; v++ )
    {
        for( int u = 0; u < sz.width; u++ )
        {
            double x = (u - cxn)*ifx;
            double y = (v - cyn)*ify;
            double x2 = x*x, y2 = y*y;
            double r2 = x2 + y2;
            double cdist = 1 + (k[0] + (k[1] + k[4]*r2)*r2)*r2;
            double x1 = x*cdist + k[2]*2*x*y + k[3]*(r2 + 2*x2);
            double y1 = y*cdist + k[3]*2*x*y + k[2]*(r2 + 2*y2);

            _mapy.at<float>(v, u) = (float)(y1*fy + cy);
            _mapx.at<float>(v, u) = (float)(x1*fx + cx);
        }
    }
}


std::ostream& operator << (std::ostream& out, const MatInfo& m)
{
    if( !m.m || m.m->empty() )
        out << "<Empty>";
    else
    {
        static const char* depthstr[] = {"8u", "8s", "16u", "16s", "32s", "32f", "64f", "?"};
        out << depthstr[m.m->depth()] << "C" << m.m->channels() << " " << m.m->dims << "-dim (";
        for( int i = 0; i < m.m->dims; i++ )
            out << m.m->size[i] << (i < m.m->dims-1 ? " x " : ")");
    }
    return out;
}


static Mat getSubArray(const Mat& m, int border, vector<int>& ofs0, vector<int>& ofs)
{
    ofs.resize(ofs0.size());
    if( border < 0 )
    {
        std::copy(ofs0.begin(), ofs0.end(), ofs.begin());
        return m;
    }
    int i, d = m.dims;
    CV_Assert(d == (int)ofs.size());
    vector<Range> r(d);
    for( i = 0; i < d; i++ )
    {
        r[i].start = std::max(0, ofs0[i] - border);
        r[i].end = std::min(ofs0[i] + 1 + border, m.size[i]);
        ofs[i] = std::min(ofs0[i], border);
    }
    return m(&r[0]);
}

template<typename _Tp, typename _WTp> static void
writeElems(std::ostream& out, const void* data, int nelems, int starpos)
{
    for(int i = 0; i < nelems; i++)
    {
        if( i == starpos )
            out << "*";
        out << (_WTp)((_Tp*)data)[i];
        if( i == starpos )
            out << "*";
        out << (i+1 < nelems ? ", " : "");
    }
}


static void writeElems(std::ostream& out, const void* data, int nelems, int depth, int starpos)
{
    if(depth == CV_8U)
        writeElems<uchar, int>(out, data, nelems, starpos);
    else if(depth == CV_8S)
        writeElems<schar, int>(out, data, nelems, starpos);
    else if(depth == CV_16U)
        writeElems<ushort, int>(out, data, nelems, starpos);
    else if(depth == CV_16S)
        writeElems<short, int>(out, data, nelems, starpos);
    else if(depth == CV_32S)
        writeElems<int, int>(out, data, nelems, starpos);
    else if(depth == CV_32F)
    {
        std::streamsize pp = out.precision();
        out.precision(8);
        writeElems<float, float>(out, data, nelems, starpos);
        out.precision(pp);
    }
    else if(depth == CV_64F)
    {
        std::streamsize pp = out.precision();
        out.precision(16);
        writeElems<double, double>(out, data, nelems, starpos);
        out.precision(pp);
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");
}


struct MatPart
{
    MatPart(const Mat& _m, const vector<int>* _loc)
    : m(&_m), loc(_loc) {}
    const Mat* m;
    const vector<int>* loc;
};

static std::ostream& operator << (std::ostream& out, const MatPart& m)
{
    CV_Assert( !m.loc || ((int)m.loc->size() == m.m->dims && m.m->dims <= 2) );
    if( !m.loc )
        out << *m.m;
    else
    {
        int i, depth = m.m->depth(), cn = m.m->channels(), width = m.m->cols*cn;
        for( i = 0; i < m.m->rows; i++ )
        {
            writeElems(out, m.m->ptr(i), width, depth, i == (*m.loc)[0] ? (*m.loc)[1] : -1);
            out << (i < m.m->rows-1 ? ";\n" : "");
        }
    }
    return out;
}

MatComparator::MatComparator(double _maxdiff, int _context)
    : maxdiff(_maxdiff), realmaxdiff(DBL_MAX), context(_context) {}

::testing::AssertionResult
MatComparator::operator()(const char* expr1, const char* expr2,
                          const Mat& m1, const Mat& m2)
{
    if( m1.type() != m2.type() || m1.size != m2.size )
        return ::testing::AssertionFailure()
        << "The reference and the actual output arrays have different type or size:\n"
        << expr1 << " ~ " << MatInfo(m1) << "\n"
        << expr2 << " ~ " << MatInfo(m2) << "\n";

    //bool ok = cvtest::cmpUlps(m1, m2, maxdiff, &realmaxdiff, &loc0);
    int code = cmpEps( m1, m2, &realmaxdiff, maxdiff, &loc0, true);

    if(code >= 0)
        return ::testing::AssertionSuccess();

    Mat m[] = {m1.reshape(1,0), m2.reshape(1,0)};
    int dims = m[0].dims;
    vector<int> loc;
    int border = dims <= 2 ? context : 0;

    Mat m1part, m2part;
    if( border == 0 )
    {
        loc = loc0;
        m1part = Mat(1, 1, m[0].depth(), m[0].ptr(&loc[0]));
        m2part = Mat(1, 1, m[1].depth(), m[1].ptr(&loc[0]));
    }
    else
    {
        m1part = getSubArray(m[0], border, loc0, loc);
        m2part = getSubArray(m[1], border, loc0, loc);
    }

    return ::testing::AssertionFailure()
    << "too big relative difference (" << realmaxdiff << " > "
    << maxdiff << ") between "
    << MatInfo(m1) << " '" << expr1 << "' and '" << expr2 << "' at " << Mat(loc0).t() << ".\n"
    << "- " << expr1 << ":\n" << MatPart(m1part, border > 0 ? &loc : 0) << ".\n"
    << "- " << expr2 << ":\n" << MatPart(m2part, border > 0 ? &loc : 0) << ".\n";
}

void printVersionInfo(bool useStdOut)
{
    // Tell CTest not to discard any output
    if(useStdOut) std::cout << "CTEST_FULL_OUTPUT" << std::endl;

    ::testing::Test::RecordProperty("cv_version", CV_VERSION);
    if(useStdOut) std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    std::string buildInfo( cv::getBuildInformation() );

    size_t pos1 = buildInfo.find("Version control");
    size_t pos2 = buildInfo.find('\n', pos1);
    if(pos1 != std::string::npos && pos2 != std::string::npos)
    {
        size_t value_start = buildInfo.rfind(' ', pos2) + 1;
        std::string ver( buildInfo.substr(value_start, pos2 - value_start) );
        ::testing::Test::RecordProperty("cv_vcs_version", ver);
        if (useStdOut) std::cout << "OpenCV VCS version: " << ver << std::endl;
    }

    pos1 = buildInfo.find("inner version");
    pos2 = buildInfo.find('\n', pos1);
    if(pos1 != std::string::npos && pos2 != std::string::npos)
    {
        size_t value_start = buildInfo.rfind(' ', pos2) + 1;
        std::string ver( buildInfo.substr(value_start, pos2 - value_start) );
        ::testing::Test::RecordProperty("cv_inner_vcs_version", ver);
        if(useStdOut) std::cout << "Inner VCS version: " << ver << std::endl;
    }

    const char * build_type =
#ifdef _DEBUG
        "debug";
#else
        "release";
#endif

    ::testing::Test::RecordProperty("cv_build_type", build_type);
    if (useStdOut) std::cout << "Build type: " << build_type << std::endl;

    const char* parallel_framework = currentParallelFramework();

    if (parallel_framework) {
        ::testing::Test::RecordProperty("cv_parallel_framework", parallel_framework);
        if (useStdOut) std::cout << "Parallel framework: " << parallel_framework << std::endl;
    }

    std::string cpu_features;

#if CV_POPCNT
    if (checkHardwareSupport(CV_CPU_POPCNT)) cpu_features += " popcnt";
#endif
#if CV_MMX
    if (checkHardwareSupport(CV_CPU_MMX)) cpu_features += " mmx";
#endif
#if CV_SSE
    if (checkHardwareSupport(CV_CPU_SSE)) cpu_features += " sse";
#endif
#if CV_SSE2
    if (checkHardwareSupport(CV_CPU_SSE2)) cpu_features += " sse2";
#endif
#if CV_SSE3
    if (checkHardwareSupport(CV_CPU_SSE3)) cpu_features += " sse3";
#endif
#if CV_SSSE3
    if (checkHardwareSupport(CV_CPU_SSSE3)) cpu_features += " ssse3";
#endif
#if CV_SSE4_1
    if (checkHardwareSupport(CV_CPU_SSE4_1)) cpu_features += " sse4.1";
#endif
#if CV_SSE4_2
    if (checkHardwareSupport(CV_CPU_SSE4_2)) cpu_features += " sse4.2";
#endif
#if CV_AVX
    if (checkHardwareSupport(CV_CPU_AVX)) cpu_features += " avx";
#endif
#if CV_AVX2
    if (checkHardwareSupport(CV_CPU_AVX2)) cpu_features += " avx2";
#endif
#if CV_FMA3
    if (checkHardwareSupport(CV_CPU_FMA3)) cpu_features += " fma3";
#endif
#if CV_AVX_512F
    if (checkHardwareSupport(CV_CPU_AVX_512F)) cpu_features += " avx-512f";
#endif
#if CV_AVX_512BW
    if (checkHardwareSupport(CV_CPU_AVX_512BW)) cpu_features += " avx-512bw";
#endif
#if CV_AVX_512CD
    if (checkHardwareSupport(CV_CPU_AVX_512CD)) cpu_features += " avx-512cd";
#endif
#if CV_AVX_512DQ
    if (checkHardwareSupport(CV_CPU_AVX_512DQ)) cpu_features += " avx-512dq";
#endif
#if CV_AVX_512ER
    if (checkHardwareSupport(CV_CPU_AVX_512ER)) cpu_features += " avx-512er";
#endif
#if CV_AVX_512IFMA512
    if (checkHardwareSupport(CV_CPU_AVX_512IFMA512)) cpu_features += " avx-512ifma512";
#endif
#if CV_AVX_512PF
    if (checkHardwareSupport(CV_CPU_AVX_512PF)) cpu_features += " avx-512pf";
#endif
#if CV_AVX_512VBMI
    if (checkHardwareSupport(CV_CPU_AVX_512VBMI)) cpu_features += " avx-512vbmi";
#endif
#if CV_AVX_512VL
    if (checkHardwareSupport(CV_CPU_AVX_512VL)) cpu_features += " avx-512vl";
#endif
#if CV_NEON
    if (checkHardwareSupport(CV_CPU_NEON)) cpu_features += " neon";
#endif
#if CV_FP16
    if (checkHardwareSupport(CV_CPU_FP16)) cpu_features += " fp16";
#endif

    cpu_features.erase(0, 1); // erase initial space

    ::testing::Test::RecordProperty("cv_cpu_features", cpu_features);
    if (useStdOut) std::cout << "CPU features: " << cpu_features << std::endl;

#ifdef HAVE_TEGRA_OPTIMIZATION
    const char * tegra_optimization = tegra::useTegra() && tegra::isDeviceSupported() ? "enabled" : "disabled";
    ::testing::Test::RecordProperty("cv_tegra_optimization", tegra_optimization);
    if (useStdOut) std::cout << "Tegra optimization: " << tegra_optimization << std::endl;
#endif
}



void threshold( const Mat& _src, Mat& _dst,
                            double thresh, double maxval, int thresh_type )
{
    int i, j;
    int depth = _src.depth(), cn = _src.channels();
    int width_n = _src.cols*cn, height = _src.rows;
    int ithresh = cvFloor(thresh);
    int imaxval, ithresh2;

    if( depth == CV_8U )
    {
        ithresh2 = saturate_cast<uchar>(ithresh);
        imaxval = saturate_cast<uchar>(maxval);
    }
    else if( depth == CV_16S )
    {
        ithresh2 = saturate_cast<short>(ithresh);
        imaxval = saturate_cast<short>(maxval);
    }
    else
    {
        ithresh2 = cvRound(ithresh);
        imaxval = cvRound(maxval);
    }

    assert( depth == CV_8U || depth == CV_16S || depth == CV_32F );

    switch( thresh_type )
    {
    case CV_THRESH_BINARY:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? imaxval : 0);
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (short)(src[j] > ithresh ? imaxval : 0);
            }
            else
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (float)((double)src[j] > thresh ? maxval : 0.f);
            }
        }
        break;
    case CV_THRESH_BINARY_INV:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? 0 : imaxval);
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (short)(src[j] > ithresh ? 0 : imaxval);
            }
            else
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (float)((double)src[j] > thresh ? 0.f : maxval);
            }
        }
        break;
    case CV_THRESH_TRUNC:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? ithresh2 : s);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? ithresh2 : s);
                }
            }
            else
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    double s = src[j];
                    dst[j] = (float)(s > thresh ? thresh : s);
                }
            }
        }
        break;
    case CV_THRESH_TOZERO:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? s : 0);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? s : 0);
                }
            }
            else
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = s > thresh ? s : 0.f;
                }
            }
        }
        break;
    case CV_THRESH_TOZERO_INV:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? 0 : s);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? 0 : s);
                }
            }
            else
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = s > thresh ? 0.f : s;
                }
            }
        }
        break;
    default:
        assert(0);
    }
}


static void
_minMaxIdx( const float* src, const uchar* mask, double* _minVal, double* _maxVal,
            size_t* _minIdx, size_t* _maxIdx, int len, size_t startIdx )
{
    double minVal = FLT_MAX, maxVal = -FLT_MAX;
    size_t minIdx = 0, maxIdx = 0;

    if( !mask )
    {
        for( int i = 0; i < len; i++ )
        {
            float val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }
    else
    {
        for( int i = 0; i < len; i++ )
        {
            float val = src[i];
            if( mask[i] && val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( mask[i] && val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }

    if (_minIdx)
        *_minIdx = minIdx;
    if (_maxIdx)
        *_maxIdx = maxIdx;
    if (_minVal)
        *_minVal = minVal;
    if (_maxVal)
        *_maxVal = maxVal;
}


void minMaxIdx( InputArray _img, double* minVal, double* maxVal,
                    Point* minLoc, Point* maxLoc, InputArray _mask )
{
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();
    CV_Assert(img.dims <= 2);

    _minMaxIdx((const float*)img.data, mask.data, minVal, maxVal, (size_t*)minLoc, (size_t*)maxLoc, (int)img.total(),1);
    if( minLoc )
        std::swap(minLoc->x, minLoc->y);
    if( maxLoc )
        std::swap(maxLoc->x, maxLoc->y);
}

}
