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

using namespace cv;
using namespace cv::gpu;

////////////////////////////////////////////////////////////////////////
//////////////////////////////// GpuMat ////////////////////////////////
////////////////////////////////////////////////////////////////////////

cv::gpu::GpuMat::GpuMat(Size size_, int type_) : 
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

cv::gpu::GpuMat::GpuMat(int rows_, int cols_, int type_, const Scalar& s_) : 
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if (rows_ > 0 && cols_ > 0)
    {
        create(rows_, cols_, type_);
        *this = s_;
    }
}

cv::gpu::GpuMat::GpuMat(Size size_, int type_, const Scalar& s_) : 
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if (size_.height > 0 && size_.width > 0)
    {
        create(size_.height, size_.width, type_);
        *this = s_;
    }
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m) : 
    flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend)
{
    if (refcount)
        CV_XADD(refcount, 1);
}

cv::gpu::GpuMat::GpuMat(int rows_, int cols_, int type_, void* data_, size_t step_) : 
    flags(Mat::MAGIC_VAL + (type_ & TYPE_MASK)), rows(rows_), cols(cols_), step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();
    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1) step = minstep;
        CV_DbgAssert( step >= minstep );
        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(Size size_, int type_, void* data_, size_t step_) : 
    flags(Mat::MAGIC_VAL + (type_ & TYPE_MASK)), rows(size_.height), cols(size_.width),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();
    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1) step = minstep;
        CV_DbgAssert( step >= minstep );
        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, const Range& rowRange, const Range& colRange)
{
    flags = m.flags;
    step = m.step; refcount = m.refcount;
    data = m.data; datastart = m.datastart; dataend = m.dataend;

    if (rowRange == Range::all())
        rows = m.rows;
    else
    {
        CV_Assert( 0 <= rowRange.start && rowRange.start <= rowRange.end && rowRange.end <= m.rows );
        rows = rowRange.size();
        data += step*rowRange.start;
    }

    if (colRange == Range::all())
        cols = m.cols;
    else
    {
        CV_Assert( 0 <= colRange.start && colRange.start <= colRange.end && colRange.end <= m.cols );
        cols = colRange.size();
        data += colRange.start*elemSize();
        flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    }

    if( rows == 1 )
        flags |= Mat::CONTINUOUS_FLAG;

    if( refcount )
        CV_XADD(refcount, 1);
    if( rows <= 0 || cols <= 0 )
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, const Rect& roi) : 
    flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    data += roi.x*elemSize();
    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols &&
        0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );
    if( refcount )
        CV_XADD(refcount, 1);
    if( rows <= 0 || cols <= 0 )
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const Mat& m) : 
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0) 
{ 
    upload(m); 
}

GpuMat& cv::gpu::GpuMat::operator = (const GpuMat& m)
{
    if( this != &m )
    {
        if( m.refcount )
            CV_XADD(m.refcount, 1);
        release();
        flags = m.flags;
        rows = m.rows; cols = m.cols;
        step = m.step; data = m.data;
        datastart = m.datastart; dataend = m.dataend;
        refcount = m.refcount;
    }
    return *this;
}

GpuMat& cv::gpu::GpuMat::operator = (const Mat& m) 
{ 
    upload(m); return *this; 
}

cv::gpu::GpuMat::operator Mat() const
{
    Mat m;
    download(m);
    return m;
}

GpuMat cv::gpu::GpuMat::row(int y) const 
{ 
    return GpuMat(*this, Range(y, y+1), Range::all()); 
}

GpuMat cv::gpu::GpuMat::col(int x) const 
{ 
    return GpuMat(*this, Range::all(), Range(x, x+1)); 
}

GpuMat cv::gpu::GpuMat::rowRange(int startrow, int endrow) const 
{ 
    return GpuMat(*this, Range(startrow, endrow), Range::all()); 
}

GpuMat cv::gpu::GpuMat::rowRange(const Range& r) const 
{ 
    return GpuMat(*this, r, Range::all()); 
}

GpuMat cv::gpu::GpuMat::colRange(int startcol, int endcol) const 
{ 
    return GpuMat(*this, Range::all(), Range(startcol, endcol)); 
}

GpuMat cv::gpu::GpuMat::colRange(const Range& r) const 
{ 
    return GpuMat(*this, Range::all(), r); 
}

void cv::gpu::GpuMat::create(Size size_, int type_) 
{ 
    create(size_.height, size_.width, type_); 
}

void cv::gpu::GpuMat::swap(GpuMat& b)
{
    std::swap( flags, b.flags );
    std::swap( rows, b.rows ); 
    std::swap( cols, b.cols );
    std::swap( step, b.step ); 
    std::swap( data, b.data );
    std::swap( datastart, b.datastart );
    std::swap( dataend, b.dataend );
    std::swap( refcount, b.refcount );
}

void cv::gpu::GpuMat::locateROI(Size& wholeSize, Point& ofs) const
{
    size_t esz = elemSize(), minstep;
    ptrdiff_t delta1 = data - datastart, delta2 = dataend - datastart;
    CV_DbgAssert( step > 0 );
    if( delta1 == 0 )
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = (int)(delta1/step);
        ofs.x = (int)((delta1 - step*ofs.y)/esz);
        CV_DbgAssert( data == datastart + ofs.y*step + ofs.x*esz );
    }
    minstep = (ofs.x + cols)*esz;
    wholeSize.height = (int)((delta2 - minstep)/step + 1);
    wholeSize.height = std::max(wholeSize.height, ofs.y + rows);
    wholeSize.width = (int)((delta2 - step*(wholeSize.height-1))/esz);
    wholeSize.width = std::max(wholeSize.width, ofs.x + cols);
}

GpuMat& cv::gpu::GpuMat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    Size wholeSize; Point ofs;
    size_t esz = elemSize();
    locateROI( wholeSize, ofs );
    int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
    int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
    data += (row1 - ofs.y)*step + (col1 - ofs.x)*esz;
    rows = row2 - row1; cols = col2 - col1;
    if( esz*cols == step || rows == 1 )
        flags |= Mat::CONTINUOUS_FLAG;
    else
        flags &= ~Mat::CONTINUOUS_FLAG;
    return *this;
}

cv::gpu::GpuMat GpuMat::operator()(Range rowRange, Range colRange) const 
{ 
    return GpuMat(*this, rowRange, colRange); 
}

cv::gpu::GpuMat GpuMat::operator()(const Rect& roi) const 
{ 
    return GpuMat(*this, roi); 
}

bool cv::gpu::GpuMat::isContinuous() const 
{ 
    return (flags & Mat::CONTINUOUS_FLAG) != 0; 
}

size_t cv::gpu::GpuMat::elemSize() const 
{ 
    return CV_ELEM_SIZE(flags); 
}

size_t cv::gpu::GpuMat::elemSize1() const 
{ 
    return CV_ELEM_SIZE1(flags); 
}

int cv::gpu::GpuMat::type() const 
{ 
    return CV_MAT_TYPE(flags); 
}

int cv::gpu::GpuMat::depth() const 
{ 
    return CV_MAT_DEPTH(flags); 
}

int cv::gpu::GpuMat::channels() const 
{ 
    return CV_MAT_CN(flags); 
}

Size cv::gpu::GpuMat::size() const 
{ 
    return Size(cols, rows); 
}

unsigned char* cv::gpu::GpuMat::ptr(int y)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step*y;
}

const unsigned char* cv::gpu::GpuMat::ptr(int y) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step*y;
}

GpuMat cv::gpu::GpuMat::t() const
{
    GpuMat tmp;
    transpose(*this, tmp);
    return tmp;
}

GpuMat cv::gpu::createContinuous(int rows, int cols, int type)
{
    GpuMat m;
    createContinuous(rows, cols, type, m);
    return m;
}

void cv::gpu::createContinuous(Size size, int type, GpuMat& m)
{
    createContinuous(size.height, size.width, type, m);
}

GpuMat cv::gpu::createContinuous(Size size, int type)
{
    GpuMat m;
    createContinuous(size, type, m);
    return m;
}

void cv::gpu::ensureSizeIsEnough(Size size, int type, GpuMat& m)
{
    ensureSizeIsEnough(size.height, size.width, type, m);
}

#if !defined (HAVE_CUDA)

void cv::gpu::GpuMat::upload(const Mat&) { throw_nogpu(); }
void cv::gpu::GpuMat::download(cv::Mat&) const { throw_nogpu(); }
void cv::gpu::GpuMat::copyTo(GpuMat&) const { throw_nogpu(); }
void cv::gpu::GpuMat::copyTo(GpuMat&, const GpuMat&) const { throw_nogpu(); }
void cv::gpu::GpuMat::convertTo(GpuMat&, int, double, double) const { throw_nogpu(); }
GpuMat& cv::gpu::GpuMat::operator = (const Scalar&) { throw_nogpu(); return *this; }
GpuMat& cv::gpu::GpuMat::setTo(const Scalar&, const GpuMat&) { throw_nogpu(); return *this; }
GpuMat cv::gpu::GpuMat::reshape(int, int) const { throw_nogpu(); return GpuMat(); }
void cv::gpu::GpuMat::create(int, int, int) { throw_nogpu(); }
void cv::gpu::GpuMat::release() {}
void cv::gpu::createContinuous(int, int, int, GpuMat&) { throw_nogpu(); }
void cv::gpu::ensureSizeIsEnough(int, int, int, GpuMat&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace matrix_operations
{
    void copy_to_with_mask(const DevMem2D& src, DevMem2D dst, int depth, const DevMem2D& mask, int channels, const cudaStream_t & stream = 0);

    template <typename T>
    void set_to_gpu(const DevMem2D& mat, const T* scalar, int channels, cudaStream_t stream);
    template <typename T>
    void set_to_gpu(const DevMem2D& mat, const T* scalar, const DevMem2D& mask, int channels, cudaStream_t stream);

    void convert_gpu(const DevMem2D& src, int sdepth, const DevMem2D& dst, int ddepth, double alpha, double beta, cudaStream_t stream = 0);
}}}


void cv::gpu::GpuMat::upload(const Mat& m)
{
    CV_DbgAssert(!m.empty());
    create(m.size(), m.type());
    cudaSafeCall( cudaMemcpy2D(data, step, m.data, m.step, cols * elemSize(), rows, cudaMemcpyHostToDevice) );
}

void cv::gpu::GpuMat::upload(const CudaMem& m, Stream& stream)
{
    CV_DbgAssert(!m.empty());
    stream.enqueueUpload(m, *this);
}

void cv::gpu::GpuMat::download(cv::Mat& m) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    cudaSafeCall( cudaMemcpy2D(m.data, m.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToHost) );
}

void cv::gpu::GpuMat::download(CudaMem& m, Stream& stream) const
{
    CV_DbgAssert(!m.empty());
    stream.enqueueDownload(*this, m);
}

void cv::gpu::GpuMat::copyTo(GpuMat& m) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    cudaSafeCall( cudaMemcpy2D(m.data, m.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice) );
    cudaSafeCall( cudaDeviceSynchronize() );
}

void cv::gpu::GpuMat::copyTo(GpuMat& mat, const GpuMat& mask) const
{
    if (mask.empty())
    {
        copyTo(mat);
    }
    else
    {
        mat.create(size(), type());
        cv::gpu::matrix_operations::copy_to_with_mask(*this, mat, depth(), mask, channels());
    }
}

namespace
{
    template<int n> struct NPPTypeTraits;
    template<> struct NPPTypeTraits<CV_8U>  { typedef Npp8u npp_type; };
    template<> struct NPPTypeTraits<CV_16U> { typedef Npp16u npp_type; };
    template<> struct NPPTypeTraits<CV_16S> { typedef Npp16s npp_type; };
    template<> struct NPPTypeTraits<CV_32S> { typedef Npp32s npp_type; };
    template<> struct NPPTypeTraits<CV_32F> { typedef Npp32f npp_type; };

    template<int SDEPTH, int DDEPTH> struct NppConvertFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI);
    };
    template<int DDEPTH> struct NppConvertFunc<CV_32F, DDEPTH>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);
    };

    template<int SDEPTH, int DDEPTH, typename NppConvertFunc<SDEPTH, DDEPTH>::func_ptr func> struct NppCvt
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void cvt(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int DDEPTH, typename NppConvertFunc<CV_32F, DDEPTH>::func_ptr func> struct NppCvt<CV_32F, DDEPTH, func>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void cvt(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            nppSafeCall( func(src.ptr<Npp32f>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz, NPP_RND_NEAR) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    void convertToKernelCaller(const GpuMat& src, GpuMat& dst)
    {
        matrix_operations::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), 1.0, 0.0);
    }
}

void cv::gpu::GpuMat::convertTo( GpuMat& dst, int rtype, double alpha, double beta ) const
{
    CV_Assert((depth() != CV_64F && CV_MAT_DEPTH(rtype) != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    bool noScale = fabs(alpha-1) < std::numeric_limits<double>::epsilon() && fabs(beta) < std::numeric_limits<double>::epsilon();

    if( rtype < 0 )
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    int scn = channels();
    int sdepth = depth(), ddepth = CV_MAT_DEPTH(rtype);
    if( sdepth == ddepth && noScale )
    {
        copyTo(dst);
        return;
    }

    GpuMat temp;
    const GpuMat* psrc = this;
    if( sdepth != ddepth && psrc == &dst )
        psrc = &(temp = *this);

    dst.create( size(), rtype );

    if (!noScale)
        matrix_operations::convert_gpu(psrc->reshape(1), sdepth, dst.reshape(1), ddepth, alpha, beta);
    else
    {
        typedef void (*convert_caller_t)(const GpuMat& src, GpuMat& dst);
        static const convert_caller_t convert_callers[8][8][4] =
        {
            {
                {0,0,0,0},
                {convertToKernelCaller, convertToKernelCaller, convertToKernelCaller, convertToKernelCaller},
                {NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C4R>::cvt},
                {NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C4R>::cvt},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {NppCvt<CV_8U, CV_32F, nppiConvert_8u32f_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0}
            },
            {
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0}
            },
            {
                {NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C4R>::cvt},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {NppCvt<CV_16U, CV_32S, nppiConvert_16u32s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {NppCvt<CV_16U, CV_32F, nppiConvert_16u32f_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0}
            },
            {
                {NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C4R>::cvt},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0},
                {NppCvt<CV_16S, CV_32S, nppiConvert_16s32s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {NppCvt<CV_16S, CV_32F, nppiConvert_16s32f_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0}
            },
            {
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0}
            },
            {
                {NppCvt<CV_32F, CV_8U, nppiConvert_32f8u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {NppCvt<CV_32F, CV_16U, nppiConvert_32f16u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {NppCvt<CV_32F, CV_16S, nppiConvert_32f16s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0}
            },
            {
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                {0,0,0,0},
                {0,0,0,0}
            },
            {
                {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}
            }
        };

        convert_callers[sdepth][ddepth][scn-1](*psrc, dst);
    }
}

GpuMat& GpuMat::operator = (const Scalar& s)
{
    setTo(s);
    return *this;
}

namespace
{
    template<int SDEPTH, int SCN> struct NppSetFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<int SDEPTH> struct NppSetFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };

    template<int SDEPTH, int SCN, typename NppSetFunc<SDEPTH, SCN>::func_ptr func> struct NppSet
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, const Scalar& s)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            Scalar_<src_t> nppS = s;
            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetFunc<SDEPTH, 1>::func_ptr func> struct NppSet<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, const Scalar& s)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            Scalar_<src_t> nppS = s;
            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template <typename T>
    void kernelSet(GpuMat& src, const Scalar& s)
    {
        Scalar_<T> sf = s;
        matrix_operations::set_to_gpu(src, sf.val, src.channels(), 0);
    }

    template<int SDEPTH, int SCN> struct NppSetMaskFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };
    template<int SDEPTH> struct NppSetMaskFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };

    template<int SDEPTH, int SCN, typename NppSetMaskFunc<SDEPTH, SCN>::func_ptr func> struct NppSetMask
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, const Scalar& s, const GpuMat& mask)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            Scalar_<src_t> nppS = s;
            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetMaskFunc<SDEPTH, 1>::func_ptr func> struct NppSetMask<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, const Scalar& s, const GpuMat& mask)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            Scalar_<src_t> nppS = s;
            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template <typename T>
    void kernelSetMask(GpuMat& src, const Scalar& s, const GpuMat& mask)
    {
        Scalar_<T> sf = s;
        matrix_operations::set_to_gpu(src, sf.val, mask, src.channels(), 0);
    }
}

GpuMat& GpuMat::setTo(const Scalar& s, const GpuMat& mask)
{
    CV_Assert(mask.type() == CV_8UC1);

    CV_Assert((depth() != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    CV_DbgAssert(!this->empty());

    NppiSize sz;
    sz.width  = cols;
    sz.height = rows;

    if (mask.empty())
    {
        if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
        {
            cudaSafeCall( cudaMemset2D(data, step, 0, cols * elemSize(), rows) );
            return *this;
        }
        if (depth() == CV_8U)
        {
            int cn = channels();

            if (cn == 1 || (cn == 2 && s[0] == s[1]) || (cn == 3 && s[0] == s[1] && s[0] == s[2]) || (cn == 4 && s[0] == s[1] && s[0] == s[2] && s[0] == s[3]))
            {
                int val = saturate_cast<uchar>(s[0]);
                cudaSafeCall( cudaMemset2D(data, step, val, cols * elemSize(), rows) );
                return *this;
            }
        }
        typedef void (*set_caller_t)(GpuMat& src, const Scalar& s);
        static const set_caller_t set_callers[8][4] =
        {
            {NppSet<CV_8U, 1, nppiSet_8u_C1R>::set,kernelSet<uchar>,kernelSet<uchar>,NppSet<CV_8U, 4, nppiSet_8u_C4R>::set},
            {kernelSet<schar>,kernelSet<schar>,kernelSet<schar>,kernelSet<schar>},
            {NppSet<CV_16U, 1, nppiSet_16u_C1R>::set,NppSet<CV_16U, 2, nppiSet_16u_C2R>::set,kernelSet<ushort>,NppSet<CV_16U, 4, nppiSet_16u_C4R>::set},
            {NppSet<CV_16S, 1, nppiSet_16s_C1R>::set,NppSet<CV_16S, 2, nppiSet_16s_C2R>::set,kernelSet<short>,NppSet<CV_16S, 4, nppiSet_16s_C4R>::set},
            {NppSet<CV_32S, 1, nppiSet_32s_C1R>::set,kernelSet<int>,kernelSet<int>,NppSet<CV_32S, 4, nppiSet_32s_C4R>::set},
            {NppSet<CV_32F, 1, nppiSet_32f_C1R>::set,kernelSet<float>,kernelSet<float>,NppSet<CV_32F, 4, nppiSet_32f_C4R>::set},
            {kernelSet<double>,kernelSet<double>,kernelSet<double>,kernelSet<double>},
            {0,0,0,0}
        };
        set_callers[depth()][channels()-1](*this, s);
    }
    else
    {
        typedef void (*set_caller_t)(GpuMat& src, const Scalar& s, const GpuMat& mask);
        static const set_caller_t set_callers[8][4] =
        {
            {NppSetMask<CV_8U, 1, nppiSet_8u_C1MR>::set,kernelSetMask<uchar>,kernelSetMask<uchar>,NppSetMask<CV_8U, 4, nppiSet_8u_C4MR>::set},
            {kernelSetMask<schar>,kernelSetMask<schar>,kernelSetMask<schar>,kernelSetMask<schar>},
            {NppSetMask<CV_16U, 1, nppiSet_16u_C1MR>::set,kernelSetMask<ushort>,kernelSetMask<ushort>,NppSetMask<CV_16U, 4, nppiSet_16u_C4MR>::set},
            {NppSetMask<CV_16S, 1, nppiSet_16s_C1MR>::set,kernelSetMask<short>,kernelSetMask<short>,NppSetMask<CV_16S, 4, nppiSet_16s_C4MR>::set},
            {NppSetMask<CV_32S, 1, nppiSet_32s_C1MR>::set,kernelSetMask<int>,kernelSetMask<int>,NppSetMask<CV_32S, 4, nppiSet_32s_C4MR>::set},
            {NppSetMask<CV_32F, 1, nppiSet_32f_C1MR>::set,kernelSetMask<float>,kernelSetMask<float>,NppSetMask<CV_32F, 4, nppiSet_32f_C4MR>::set},
            {kernelSetMask<double>,kernelSetMask<double>,kernelSetMask<double>,kernelSetMask<double>},
            {0,0,0,0}
        };
        set_callers[depth()][channels()-1](*this, s, mask);
    }

    return *this;
}


GpuMat cv::gpu::GpuMat::reshape(int new_cn, int new_rows) const
{
    GpuMat hdr = *this;

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
            CV_Error( CV_BadStep, "The matrix is not continuous, thus its number of rows can not be changed" );

        if( (unsigned)new_rows > (unsigned)total_size )
            CV_Error( CV_StsOutOfRange, "Bad new number of rows" );

        total_width = total_size / new_rows;

        if( total_width * new_rows != total_size )
            CV_Error( CV_StsBadArg, "The total number of matrix elements is not divisible by the new number of rows" );

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if( new_width * new_cn != total_width )
        CV_Error( CV_BadNumChannels, "The total width is not divisible by the new number of channels" );

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
    return hdr;
}

void cv::gpu::GpuMat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    if( rows == _rows && cols == _cols && type() == _type && data )
        return;
    if( data )
        release();
    CV_DbgAssert( _rows >= 0 && _cols >= 0 );
    if( _rows > 0 && _cols > 0 )
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        size_t esz = elemSize();

        void *dev_ptr;
        cudaSafeCall( cudaMallocPitch(&dev_ptr, &step, esz * cols, rows) );

        // Single row must be continuous
        if (rows == 1)
            step = esz * cols;

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = (int64)step*rows;
        size_t nettosize = (size_t)_nettosize;

        datastart = data = (uchar*)dev_ptr;
        dataend = data + nettosize;

        refcount = (int*)fastMalloc(sizeof(*refcount));
        *refcount = 1;
    }
}

void cv::gpu::GpuMat::release()
{
    if( refcount && CV_XADD(refcount, -1) == 1 )
    {
        fastFree(refcount);
        cudaSafeCall( cudaFree(datastart) );
    }
    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}

void cv::gpu::createContinuous(int rows, int cols, int type, GpuMat& m)
{
    int area = rows * cols;
    if (!m.isContinuous() || m.type() != type || m.size().area() != area)
        m.create(1, area, type);
    m = m.reshape(0, rows);
}

void cv::gpu::ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m)
{
    if (m.type() == type && m.rows >= rows && m.cols >= cols)
        m = m(Rect(0, 0, cols, rows));
    else
        m.create(rows, cols, type);
}

#endif /* !defined (HAVE_CUDA) */
