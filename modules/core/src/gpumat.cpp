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
#include "opencv2/core/gpumat.hpp"

#include <iostream>

#ifdef HAVE_CUDA
    #include <cuda_runtime.h>
    #include <npp.h>
#endif

#ifdef HAVE_OPENGL
    #include <GL/gl.h>

    #ifdef HAVE_CUDA
        #include <cuda_gl_interop.h>
    #endif
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

////////////////////////////////////////////////////////////////////////
// GpuMat

cv::gpu::GpuMat::GpuMat(const GpuMat& m) 
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend)
{
    if (refcount)
        CV_XADD(refcount, 1);
}

cv::gpu::GpuMat::GpuMat(int rows_, int cols_, int type_, void* data_, size_t step_) : 
    flags(Mat::MAGIC_VAL + (type_ & TYPE_MASK)), rows(rows_), cols(cols_), 
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
        if (rows == 1) 
            step = minstep;

        CV_DbgAssert(step >= minstep);

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
        if (rows == 1) 
            step = minstep;

        CV_DbgAssert(step >= minstep);

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Range rowRange, Range colRange)
{
    flags = m.flags;
    step = m.step; refcount = m.refcount;
    data = m.data; datastart = m.datastart; dataend = m.dataend;

    if (rowRange == Range::all())
        rows = m.rows;
    else
    {
        CV_Assert(0 <= rowRange.start && rowRange.start <= rowRange.end && rowRange.end <= m.rows);

        rows = rowRange.size();
        data += step*rowRange.start;
    }

    if (colRange == Range::all())
        cols = m.cols;
    else
    {
        CV_Assert(0 <= colRange.start && colRange.start <= colRange.end && colRange.end <= m.cols);

        cols = colRange.size();
        data += colRange.start*elemSize();
        flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    }

    if (rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Rect roi) : 
    flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    data += roi.x * elemSize();

    CV_Assert(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows);

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const Mat& m) : 
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0) 
{ 
    upload(m); 
}

GpuMat& cv::gpu::GpuMat::operator = (const GpuMat& m)
{
    if (this != &m)
    {
        GpuMat temp(m);
        swap(temp);
    }

    return *this;
}

void cv::gpu::GpuMat::swap(GpuMat& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows); 
    std::swap(cols, b.cols);
    std::swap(step, b.step); 
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
}

void cv::gpu::GpuMat::locateROI(Size& wholeSize, Point& ofs) const
{
    size_t esz = elemSize();
    ptrdiff_t delta1 = data - datastart;
    ptrdiff_t delta2 = dataend - datastart;

    CV_DbgAssert(step > 0);

    if (delta1 == 0)
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = static_cast<int>(delta1 / step);
        ofs.x = static_cast<int>((delta1 - step * ofs.y) / esz);

        CV_DbgAssert(data == datastart + ofs.y * step + ofs.x * esz);
    }

    size_t minstep = (ofs.x + cols) * esz;

    wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / step + 1), ofs.y + rows);
    wholeSize.width = std::max(static_cast<int>((delta2 - step * (wholeSize.height - 1)) / esz), ofs.x + cols);
}

GpuMat& cv::gpu::GpuMat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    Size wholeSize; 
    Point ofs;
    locateROI(wholeSize, ofs);

    size_t esz = elemSize();

    int row1 = std::max(ofs.y - dtop, 0); 
    int row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);

    int col1 = std::max(ofs.x - dleft, 0);
    int col2 = std::min(ofs.x + cols + dright, wholeSize.width);

    data += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
    rows = row2 - row1; 
    cols = col2 - col1;

    if (esz * cols == step || rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;
    else
        flags &= ~Mat::CONTINUOUS_FLAG;

    return *this;
}

GpuMat cv::gpu::GpuMat::reshape(int new_cn, int new_rows) const
{
    GpuMat hdr = *this;

    int cn = channels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;

    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(CV_BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(CV_StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;

        if (total_width * new_rows != total_size)
            CV_Error(CV_StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if (new_width * new_cn != total_width)
        CV_Error(CV_BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);

    return hdr;
}

cv::Mat::Mat(const GpuMat& m) : flags(0), dims(0), rows(0), cols(0), data(0), refcount(0), datastart(0), dataend(0), datalimit(0), allocator(0), size(&rows)
{
    m.download(*this);
}

namespace
{
    class CV_EXPORTS GpuFuncTable
    {
    public:
        virtual ~GpuFuncTable() {}

        virtual void copy(const Mat& src, GpuMat& dst) const = 0;
        virtual void copy(const GpuMat& src, Mat& dst) const = 0;
        virtual void copy(const GpuMat& src, GpuMat& dst) const = 0;

        virtual void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask) const = 0;

        virtual void convert(const GpuMat& src, GpuMat& dst) const = 0;
        virtual void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta) const = 0;

        virtual void setTo(GpuMat& m, Scalar s, const GpuMat& mask) const = 0;

        virtual void mallocPitch(void** devPtr, size_t* step, size_t width, size_t height) const = 0;
        virtual void free(void* devPtr) const = 0;
    };
}

#ifndef HAVE_CUDA

namespace
{
    void throw_nogpu() 
    { 
        CV_Error(CV_GpuNotSupported, "The library is compiled without GPU support"); 
    }

    class EmptyFuncTable : public GpuFuncTable
    {
    public:
        void copy(const Mat&, GpuMat&) const { throw_nogpu(); }
        void copy(const GpuMat&, Mat&) const { throw_nogpu(); }
        void copy(const GpuMat&, GpuMat&) const { throw_nogpu(); }

        void copyWithMask(const GpuMat&, GpuMat&, const GpuMat&) const { throw_nogpu(); }

        void convert(const GpuMat&, GpuMat&) const { throw_nogpu(); }
        void convert(const GpuMat&, GpuMat&, double, double) const { throw_nogpu(); }

        void setTo(GpuMat&, Scalar, const GpuMat&) const { throw_nogpu(); }

        void mallocPitch(void**, size_t*, size_t, size_t) const { throw_nogpu(); }
        void free(void*) const {}
    };

    const GpuFuncTable* gpuFuncTable()
    {
        static EmptyFuncTable empty;
        return &empty;
    }
}

#else // HAVE_CUDA

namespace cv { namespace gpu { namespace device 
{
    void copy_to_with_mask(DevMem2Db src, DevMem2Db dst, int depth, DevMem2Db mask, int channels, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(DevMem2Db mat, const T* scalar, int channels, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(DevMem2Db mat, const T* scalar, DevMem2Db mask, int channels, cudaStream_t stream);

    void convert_gpu(DevMem2Db src, int sdepth, DevMem2Db dst, int ddepth, double alpha, double beta, cudaStream_t stream);
}}}

namespace
{
#if defined(__GNUC__)
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
    #define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)
    #define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__)
#endif

    inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
    {
        if (cudaSuccess != err)
            cv::gpu::error(cudaGetErrorString(err), file, line, func);
    }

    inline void ___nppSafeCall(int err, const char *file, const int line, const char *func = "")
    {
        if (err < 0)
        {
            std::ostringstream msg;
            msg << "NPP API Call Error: " << err;
            cv::gpu::error(msg.str().c_str(), file, line, func);
        }
    }
}

namespace
{
    template <typename T> void kernelSetCaller(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        ::cv::gpu::device::set_to_gpu(src, sf.val, src.channels(), stream);
    }

    template <typename T> void kernelSetCaller(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        ::cv::gpu::device::set_to_gpu(src, sf.val, mask, src.channels(), stream);
    }
}

namespace cv { namespace gpu
{
    CV_EXPORTS void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream = 0) 
    { 
        ::cv::gpu::device::copy_to_with_mask(src, dst, src.depth(), mask, src.channels(), stream);
    }

    CV_EXPORTS void convertTo(const GpuMat& src, GpuMat& dst)
    {
        ::cv::gpu::device::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), 1.0, 0.0, 0);
    }  

    CV_EXPORTS void convertTo(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream = 0)
    {
        ::cv::gpu::device::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), alpha, beta, stream);
    }

    CV_EXPORTS void setTo(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        typedef void (*caller_t)(GpuMat& src, Scalar s, cudaStream_t stream);

        static const caller_t callers[] = 
        {
            kernelSetCaller<uchar>, kernelSetCaller<schar>, kernelSetCaller<ushort>, kernelSetCaller<short>, kernelSetCaller<int>,
            kernelSetCaller<float>, kernelSetCaller<double>
        };

        callers[src.depth()](src, s, stream);
    }

    CV_EXPORTS void setTo(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
    {
        typedef void (*caller_t)(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream);

        static const caller_t callers[] = 
        {
            kernelSetCaller<uchar>, kernelSetCaller<schar>, kernelSetCaller<ushort>, kernelSetCaller<short>, kernelSetCaller<int>,
            kernelSetCaller<float>, kernelSetCaller<double>
        };

        callers[src.depth()](src, s, mask, stream);
    }

    CV_EXPORTS void setTo(GpuMat& src, Scalar s)
    {
        setTo(src, s, 0);
    }

    CV_EXPORTS void setTo(GpuMat& src, Scalar s, const GpuMat& mask)
    {
        setTo(src, s, mask, 0);
    }
}}

namespace
{
    //////////////////////////////////////////////////////////////////////////
    // Convert

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

    //////////////////////////////////////////////////////////////////////////
    // Set
    
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

        static void set(GpuMat& src, Scalar s)
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

        static void set(GpuMat& src, Scalar s)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };    

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

        static void set(GpuMat& src, Scalar s, const GpuMat& mask)
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

        static void set(GpuMat& src, Scalar s, const GpuMat& mask)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };    

    class CudaFuncTable : public GpuFuncTable
    {
    public:
        void copy(const Mat& src, GpuMat& dst) const 
        { 
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyHostToDevice) );
        }
        void copy(const GpuMat& src, Mat& dst) const
        { 
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToHost) );
        }
        void copy(const GpuMat& src, GpuMat& dst) const
        { 
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToDevice) );
        }

        void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask) const 
        { 
            ::cv::gpu::copyWithMask(src, dst, mask);
        }

        void convert(const GpuMat& src, GpuMat& dst) const 
        { 
            typedef void (*caller_t)(const GpuMat& src, GpuMat& dst);
            static const caller_t callers[7][7][7] =
            {
                {                
                    /*  8U ->  8U */ {0, 0, 0, 0},
                    /*  8U ->  8S */ {::cv::gpu::convertTo, ::cv::gpu::convertTo, ::cv::gpu::convertTo, ::cv::gpu::convertTo},
                    /*  8U -> 16U */ {NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C4R>::cvt},
                    /*  8U -> 16S */ {NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C4R>::cvt},
                    /*  8U -> 32S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /*  8U -> 32F */ {NppCvt<CV_8U, CV_32F, nppiConvert_8u32f_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /*  8U -> 64F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo}
                },
                {
                    /*  8S ->  8U */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /*  8S ->  8S */ {0,0,0,0},
                    /*  8S -> 16U */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /*  8S -> 16S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /*  8S -> 32S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /*  8S -> 32F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /*  8S -> 64F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo}
                },
                {
                    /* 16U ->  8U */ {NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C4R>::cvt},
                    /* 16U ->  8S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16U -> 16U */ {0,0,0,0},
                    /* 16U -> 16S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16U -> 32S */ {NppCvt<CV_16U, CV_32S, nppiConvert_16u32s_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16U -> 32F */ {NppCvt<CV_16U, CV_32F, nppiConvert_16u32f_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16U -> 64F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo}
                },
                {
                    /* 16S ->  8U */ {NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C4R>::cvt},
                    /* 16S ->  8S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16S -> 16U */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16S -> 16S */ {0,0,0,0},
                    /* 16S -> 32S */ {NppCvt<CV_16S, CV_32S, nppiConvert_16s32s_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16S -> 32F */ {NppCvt<CV_16S, CV_32F, nppiConvert_16s32f_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 16S -> 64F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo}
                },
                {
                    /* 32S ->  8U */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32S ->  8S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32S -> 16U */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32S -> 16S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32S -> 32S */ {0,0,0,0},
                    /* 32S -> 32F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32S -> 64F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo}
                },
                {
                    /* 32F ->  8U */ {NppCvt<CV_32F, CV_8U, nppiConvert_32f8u_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32F ->  8S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32F -> 16U */ {NppCvt<CV_32F, CV_16U, nppiConvert_32f16u_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32F -> 16S */ {NppCvt<CV_32F, CV_16S, nppiConvert_32f16s_C1R>::cvt,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32F -> 32S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 32F -> 32F */ {0,0,0,0},
                    /* 32F -> 64F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo}
                },
                {
                    /* 64F ->  8U */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 64F ->  8S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 64F -> 16U */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 64F -> 16S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 64F -> 32S */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 64F -> 32F */ {::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo,::cv::gpu::convertTo},
                    /* 64F -> 64F */ {0,0,0,0}
                }
            };

            caller_t func = callers[src.depth()][dst.depth()][src.channels() - 1];
            CV_DbgAssert(func != 0);

            func(src, dst);
        }

        void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta) const 
        { 
            ::cv::gpu::convertTo(src, dst, alpha, beta);
        }

        void setTo(GpuMat& m, Scalar s, const GpuMat& mask) const
        {
            NppiSize sz;
            sz.width  = m.cols;
            sz.height = m.rows;

            if (mask.empty())
            {
                if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
                {
                    cudaSafeCall( cudaMemset2D(m.data, m.step, 0, m.cols * m.elemSize(), m.rows) );
                    return;
                }

                if (m.depth() == CV_8U)
                {
                    int cn = m.channels();

                    if (cn == 1 || (cn == 2 && s[0] == s[1]) || (cn == 3 && s[0] == s[1] && s[0] == s[2]) || (cn == 4 && s[0] == s[1] && s[0] == s[2] && s[0] == s[3]))
                    {
                        int val = saturate_cast<uchar>(s[0]);
                        cudaSafeCall( cudaMemset2D(m.data, m.step, val, m.cols * m.elemSize(), m.rows) );
                        return;
                    }
                }

                typedef void (*caller_t)(GpuMat& src, Scalar s);
                static const caller_t callers[7][4] =
                {
                    {NppSet<CV_8U, 1, nppiSet_8u_C1R>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSet<CV_8U, 4, nppiSet_8u_C4R>::set},
                    {::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo},
                    {NppSet<CV_16U, 1, nppiSet_16u_C1R>::set, NppSet<CV_16U, 2, nppiSet_16u_C2R>::set, ::cv::gpu::setTo, NppSet<CV_16U, 4, nppiSet_16u_C4R>::set},
                    {NppSet<CV_16S, 1, nppiSet_16s_C1R>::set, NppSet<CV_16S, 2, nppiSet_16s_C2R>::set, ::cv::gpu::setTo, NppSet<CV_16S, 4, nppiSet_16s_C4R>::set},
                    {NppSet<CV_32S, 1, nppiSet_32s_C1R>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSet<CV_32S, 4, nppiSet_32s_C4R>::set},
                    {NppSet<CV_32F, 1, nppiSet_32f_C1R>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSet<CV_32F, 4, nppiSet_32f_C4R>::set},
                    {::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo}
                };

                callers[m.depth()][m.channels() - 1](m, s);
            }
            else
            {
                typedef void (*caller_t)(GpuMat& src, Scalar s, const GpuMat& mask);

                static const caller_t callers[7][4] =
                {
                    {NppSetMask<CV_8U, 1, nppiSet_8u_C1MR>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSetMask<CV_8U, 4, nppiSet_8u_C4MR>::set},
                    {::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo},
                    {NppSetMask<CV_16U, 1, nppiSet_16u_C1MR>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSetMask<CV_16U, 4, nppiSet_16u_C4MR>::set},
                    {NppSetMask<CV_16S, 1, nppiSet_16s_C1MR>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSetMask<CV_16S, 4, nppiSet_16s_C4MR>::set},
                    {NppSetMask<CV_32S, 1, nppiSet_32s_C1MR>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSetMask<CV_32S, 4, nppiSet_32s_C4MR>::set},
                    {NppSetMask<CV_32F, 1, nppiSet_32f_C1MR>::set, ::cv::gpu::setTo, ::cv::gpu::setTo, NppSetMask<CV_32F, 4, nppiSet_32f_C4MR>::set},
                    {::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo, ::cv::gpu::setTo}
                };

                callers[m.depth()][m.channels() - 1](m, s, mask);
            }
        }

        void mallocPitch(void** devPtr, size_t* step, size_t width, size_t height) const
        {
            cudaSafeCall( cudaMallocPitch(devPtr, step, width, height) );
        }

        void free(void* devPtr) const
        {
            cudaFree(devPtr);
        }
    };
    
    const GpuFuncTable* gpuFuncTable()
    {
        static CudaFuncTable funcTable;
        return &funcTable;
    }
}

#endif // HAVE_CUDA

void cv::gpu::GpuMat::upload(const Mat& m)
{
    CV_DbgAssert(!m.empty());

    create(m.size(), m.type());

    gpuFuncTable()->copy(m, *this);
}

void cv::gpu::GpuMat::download(Mat& m) const
{
    CV_DbgAssert(!empty());

    m.create(size(), type());

    gpuFuncTable()->copy(*this, m);
}

void cv::gpu::GpuMat::copyTo(GpuMat& m) const
{
    CV_DbgAssert(!empty());

    m.create(size(), type());

    gpuFuncTable()->copy(*this, m);
}

void cv::gpu::GpuMat::copyTo(GpuMat& mat, const GpuMat& mask) const
{
    if (mask.empty())
        copyTo(mat);
    else
    {
        mat.create(size(), type());

        gpuFuncTable()->copyWithMask(*this, mat, mask);
    }
}

void cv::gpu::GpuMat::convertTo(GpuMat& dst, int rtype, double alpha, double beta) const
{
    bool noScale = fabs(alpha - 1) < numeric_limits<double>::epsilon() && fabs(beta) < numeric_limits<double>::epsilon();

    if (rtype < 0)
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    int sdepth = depth();
    int ddepth = CV_MAT_DEPTH(rtype);
    if (sdepth == ddepth && noScale)
    {
        copyTo(dst);
        return;
    }

    GpuMat temp;
    const GpuMat* psrc = this;
    if (sdepth != ddepth && psrc == &dst)
    {
        temp = *this;
        psrc = &temp;
    }

    dst.create(size(), rtype);

    if (noScale)
        gpuFuncTable()->convert(*psrc, dst);
    else
        gpuFuncTable()->convert(*psrc, dst, alpha, beta);
}

GpuMat& cv::gpu::GpuMat::setTo(Scalar s, const GpuMat& mask)
{
    CV_Assert(mask.empty() || mask.type() == CV_8UC1);
    CV_DbgAssert(!empty());

    gpuFuncTable()->setTo(*this, s, mask);    

    return *this;
}

void cv::gpu::GpuMat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

    if (data)
        release();

    CV_DbgAssert(_rows >= 0 && _cols >= 0);

    if (_rows > 0 && _cols > 0)
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        size_t esz = elemSize();

        void* devPtr;
        gpuFuncTable()->mallocPitch(&devPtr, &step, esz * cols, rows);

        // Single row must be continuous
        if (rows == 1)
            step = esz * cols;

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = static_cast<int64>(step) * rows;
        size_t nettosize = static_cast<size_t>(_nettosize);

        datastart = data = static_cast<uchar*>(devPtr);
        dataend = data + nettosize;

        refcount = static_cast<int*>(fastMalloc(sizeof(*refcount)));
        *refcount = 1;
    }
}

void cv::gpu::GpuMat::release()
{
    if (refcount && CV_XADD(refcount, -1) == 1)
    {
        fastFree(refcount);

        gpuFuncTable()->free(datastart);
    }

    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}

////////////////////////////////////////////////////////////////////////
// OpenGL

namespace
{
    void throw_nogl() 
    {
    #ifndef HAVE_OPENGL
        CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support"); 
    #else
        CV_Error(CV_OpenGlNotSupported, "OpenGL context doesn't exist"); 
    #endif
    }

    class EmptyGlFuncTab : public GlFuncTab
    {
    public:
        void genBuffers(int, unsigned int*) const { throw_nogl(); }
        void deleteBuffers(int, const unsigned int*) const { throw_nogl(); }

        void bufferData(unsigned int, ptrdiff_t, const void*, unsigned int) const { throw_nogl(); }
        void bufferSubData(unsigned int, ptrdiff_t, ptrdiff_t, const void*) const { throw_nogl(); }

        void bindBuffer(unsigned int, unsigned int) const { throw_nogl(); }

        void* mapBuffer(unsigned int, unsigned int) const { throw_nogl(); return 0; }
        void unmapBuffer(unsigned int) const { throw_nogl(); }

        bool isGlContextInitialized() const { return false; }
    };

    const GlFuncTab* g_glFuncTab = 0;

    const GlFuncTab* glFuncTab()
    {
        static EmptyGlFuncTab empty;
        return g_glFuncTab ? g_glFuncTab : &empty;
    }
}

void cv::gpu::setGlFuncTab(const GlFuncTab* tab)
{
    g_glFuncTab = tab;
}

#ifdef HAVE_OPENGL
    #ifndef GL_DYNAMIC_DRAW
        #define GL_DYNAMIC_DRAW 0x88E8
    #endif

    #ifndef GL_READ_WRITE
        #define GL_READ_WRITE 0x88BA
    #endif

    #ifndef GL_BGR
        #define GL_BGR 0x80E0
    #endif

    #ifndef GL_BGRA
        #define GL_BGRA 0x80E1
    #endif

    namespace
    {    
        const GLenum gl_types[] = {GL_UNSIGNED_BYTE, GL_BYTE, GL_UNSIGNED_SHORT, GL_SHORT, GL_INT, GL_FLOAT, GL_DOUBLE};

    #ifdef HAVE_CUDA
        bool g_isCudaGlDeviceInitialized = false;
    #endif
    }
#endif // HAVE_OPENGL



void cv::gpu::setGlDevice(int device)
{
#ifndef HAVE_CUDA
    throw_nogpu();
#else
    #ifndef HAVE_OPENGL
        throw_nogl();
    #else
        if (!glFuncTab()->isGlContextInitialized())
            throw_nogl();

        cudaSafeCall( cudaGLSetGLDevice(device) ); 

        g_isCudaGlDeviceInitialized = true;
    #endif
#endif
}

////////////////////////////////////////////////////////////////////////
// CudaGlInterop

#if defined HAVE_CUDA && defined HAVE_OPENGL
namespace
{
    class CudaGlInterop
    {
    public:
        CudaGlInterop() : resource_(0) 
        {
        }

        ~CudaGlInterop() 
        { 
            if (resource_)
            {
                cudaGraphicsUnregisterResource(resource_);
                resource_ = 0;
            } 
        }

        void registerBuffer(unsigned int buffer)
        {
            if (!g_isCudaGlDeviceInitialized)
                cvError(CV_GpuApiCallError, "registerBuffer", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

            cudaGraphicsResource_t resource;
            cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone) );

            resource_ = resource;
        }

        void copyFrom(const GpuMat& mat, cudaStream_t stream = 0)
        {
            CV_Assert(resource_ != 0);

            cudaSafeCall( cudaGraphicsMapResources(1, &resource_, stream) );

            void* dst_ptr;
            size_t num_bytes;
            cudaSafeCall( cudaGraphicsResourceGetMappedPointer(&dst_ptr, &num_bytes, resource_) );
            
            const void* src_ptr = mat.ptr();
            size_t widthBytes = mat.cols * mat.elemSize();

            CV_Assert(widthBytes * mat.rows <= num_bytes);

            if (stream == 0)
                cudaSafeCall( cudaMemcpy2D(dst_ptr, widthBytes, src_ptr, mat.step, widthBytes, mat.rows, cudaMemcpyDeviceToDevice) );
            else
                cudaSafeCall( cudaMemcpy2DAsync(dst_ptr, widthBytes, src_ptr, mat.step, widthBytes, mat.rows, cudaMemcpyDeviceToDevice, stream) );

            cudaGraphicsUnmapResources(1, &resource_, stream);
        }

        GpuMat map(int rows, int cols, int type, cudaStream_t stream = 0)
        {
            CV_Assert(resource_ != 0);

            cudaSafeCall( cudaGraphicsMapResources(1, &resource_, stream) );

            void* ptr;
            size_t num_bytes;
            cudaSafeCall( cudaGraphicsResourceGetMappedPointer(&ptr, &num_bytes, resource_) );

            CV_Assert( static_cast<size_t>(cols) * CV_ELEM_SIZE(type) * rows <= num_bytes );

            return GpuMat(rows, cols, type, ptr);
        }

        void unmap(cudaStream_t stream = 0)
        {
            cudaGraphicsUnmapResources(1, &resource_, stream);
        }

    private:
        cudaGraphicsResource_t resource_;
    };
}
#endif // HAVE_CUDA && HAVE_OPENGL

////////////////////////////////////////////////////////////////////////
// GlBuffer

#ifndef HAVE_OPENGL

class cv::gpu::GlBuffer::Impl
{
};

#else

class cv::gpu::GlBuffer::Impl
{
public:
    explicit Impl(unsigned int target) : rows_(0), cols_(0), type_(0), target_(target), buffer_(0) 
    {
    }

    Impl(int rows, int cols, int type, unsigned int target);

    Impl(const Mat& m, unsigned int target);

    ~Impl();

    void copyFrom(const Mat& m);

#ifdef HAVE_CUDA
    void copyFrom(const GpuMat& mat, cudaStream_t stream = 0);
#endif

    void bind() const;
    void unbind() const;

    Mat mapHost();
    void unmapHost();

#ifdef HAVE_CUDA
    GpuMat mapDevice(cudaStream_t stream = 0);
    void unmapDevice(cudaStream_t stream = 0);
#endif

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int type() const { return type_; }    
    int target() const { return target_; }

private:
    int rows_;
    int cols_;
    int type_;

    unsigned int target_;

    unsigned int buffer_;

#ifdef HAVE_CUDA
    CudaGlInterop cudaGlInterop_;
#endif
};

cv::gpu::GlBuffer::Impl::Impl(int rows, int cols, int type, unsigned int target) : rows_(0), cols_(0), type_(0), target_(target), buffer_(0) 
{ 
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl();

    CV_DbgAssert(rows > 0 && cols > 0);
    CV_DbgAssert(CV_MAT_DEPTH(type) >= 0 && CV_MAT_DEPTH(type) <= CV_64F);

    unsigned int buffer;    
    glFuncTab()->genBuffers(1, &buffer);
    CV_CheckGlError();

    size_t size = rows * cols * CV_ELEM_SIZE(type);

    glFuncTab()->bindBuffer(target_, buffer);
    CV_CheckGlError();

    glFuncTab()->bufferData(target_, size, 0, GL_DYNAMIC_DRAW);
    CV_CheckGlError();

    glFuncTab()->bindBuffer(target_, 0);
    
#ifdef HAVE_CUDA
    if (g_isCudaGlDeviceInitialized)
        cudaGlInterop_.registerBuffer(buffer);
#endif

    rows_ = rows;
    cols_ = cols;
    type_ = type;
    buffer_ = buffer;
}

cv::gpu::GlBuffer::Impl::Impl(const Mat& m, unsigned int target) : rows_(0), cols_(0), type_(0), target_(target), buffer_(0) 
{ 
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl();

    CV_DbgAssert(m.rows > 0 && m.cols > 0);
    CV_DbgAssert(m.depth() >= 0 && m.depth() <= CV_64F);
    CV_Assert(m.isContinuous());

    unsigned int buffer;  
    glFuncTab()->genBuffers(1, &buffer);
    CV_CheckGlError();

    size_t size = m.rows * m.cols * m.elemSize();

    glFuncTab()->bindBuffer(target_, buffer);
    CV_CheckGlError();

    glFuncTab()->bufferData(target_, size, m.data, GL_DYNAMIC_DRAW);
    CV_CheckGlError();

    glFuncTab()->bindBuffer(target_, 0);
    
#ifdef HAVE_CUDA
    if (g_isCudaGlDeviceInitialized)
        cudaGlInterop_.registerBuffer(buffer);
#endif

    rows_ = m.rows;
    cols_ = m.cols;
    type_ = m.type();
    buffer_ = buffer;
}

cv::gpu::GlBuffer::Impl::~Impl() 
{ 
    try
    {
        if (buffer_)
            glFuncTab()->deleteBuffers(1, &buffer_);
    }
#ifdef _DEBUG
    catch(const exception& e)
    {
        cerr << e.what() << endl;
    }
#endif
    catch(...)
    {
    }
}

void cv::gpu::GlBuffer::Impl::copyFrom(const Mat& m)
{
    CV_Assert(buffer_ != 0);

    CV_DbgAssert(rows_ == m.rows && cols_ == m.cols && type_ == m.type());
    CV_Assert(m.isContinuous());

    bind();

    size_t size = m.rows * m.cols * m.elemSize();

    glFuncTab()->bufferSubData(target_, 0, size, m.data);
    CV_CheckGlError();

    unbind();
}

#ifdef HAVE_CUDA

void cv::gpu::GlBuffer::Impl::copyFrom(const GpuMat& mat, cudaStream_t stream) 
{
    if (!g_isCudaGlDeviceInitialized)
        cvError(CV_GpuApiCallError, "copyFrom", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

    CV_Assert(buffer_ != 0);

    CV_DbgAssert(rows_ == mat.rows && cols_ == mat.cols && type_ == mat.type());

    cudaGlInterop_.copyFrom(mat, stream);
}

#endif // HAVE_CUDA

void cv::gpu::GlBuffer::Impl::bind() const 
{
    CV_Assert(buffer_ != 0);

    glFuncTab()->bindBuffer(target_, buffer_); 
    CV_CheckGlError();
}

void cv::gpu::GlBuffer::Impl::unbind() const
{ 
    glFuncTab()->bindBuffer(target_, 0);
}

Mat cv::gpu::GlBuffer::Impl::mapHost()
{
    void* ptr = glFuncTab()->mapBuffer(target_, GL_READ_WRITE);
    CV_CheckGlError();

    return Mat(rows_, cols_, type_, ptr);
}

void cv::gpu::GlBuffer::Impl::unmapHost()
{
    glFuncTab()->unmapBuffer(target_);
}

#ifdef HAVE_CUDA

GpuMat cv::gpu::GlBuffer::Impl::mapDevice(cudaStream_t stream)
{
    if (!g_isCudaGlDeviceInitialized)
        cvError(CV_GpuApiCallError, "copyFrom", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

    CV_Assert(buffer_ != 0);

    return cudaGlInterop_.map(rows_, cols_, type_, stream);
}

void cv::gpu::GlBuffer::Impl::unmapDevice(cudaStream_t stream)
{
    if (!g_isCudaGlDeviceInitialized)
        cvError(CV_GpuApiCallError, "copyFrom", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

    cudaGlInterop_.unmap(stream);
}

#endif // HAVE_CUDA

#endif // HAVE_OPENGL

cv::gpu::GlBuffer::GlBuffer(Usage usage)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(usage);
#endif
}

cv::gpu::GlBuffer::GlBuffer(int rows, int cols, int type, Usage usage)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(rows, cols, type, usage);
#endif
}

cv::gpu::GlBuffer::GlBuffer(Size size, int type, Usage usage)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(size.height, size.width, type, usage);
#endif
}

cv::gpu::GlBuffer::GlBuffer(const Mat& mat, Usage usage)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(mat, usage);
#endif
}

cv::gpu::GlBuffer::GlBuffer(const GpuMat& d_mat, Usage usage)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    #ifndef HAVE_CUDA
        throw_nogpu();
    #else
        impl_ = new Impl(d_mat.rows, d_mat.cols, d_mat.type(), usage);
        impl_->copyFrom(d_mat);
    #endif
#endif
}

cv::gpu::GlBuffer::~GlBuffer()
{
}

void cv::gpu::GlBuffer::create(int rows_, int cols_, int type_, Usage usage_)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    if (rows_ != rows() || cols_ != cols() || type_ != type() || usage_ != usage())
        impl_ = new Impl(rows_, cols_, type_, usage_);
#endif
}

void cv::gpu::GlBuffer::release()
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(usage());
#endif
}

void cv::gpu::GlBuffer::copyFrom(const Mat& mat)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    create(mat.rows, mat.cols, mat.type());
    impl_->copyFrom(mat);
#endif
}

void cv::gpu::GlBuffer::copyFrom(const GpuMat& d_mat)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    #ifndef HAVE_CUDA
        throw_nogpu();
    #else
        create(d_mat.rows, d_mat.cols, d_mat.type());
        impl_->copyFrom(d_mat);
    #endif
#endif
}

void cv::gpu::GlBuffer::bind() const
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_->bind();
#endif
}

void cv::gpu::GlBuffer::unbind() const
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_->unbind();
#endif
}

Mat cv::gpu::GlBuffer::mapHost()
{
#ifndef HAVE_OPENGL
    throw_nogl();
    return Mat();
#else
    return impl_->mapHost();
#endif
}

void cv::gpu::GlBuffer::unmapHost()
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_->unmapHost();
#endif
}

GpuMat cv::gpu::GlBuffer::mapDevice()
{
#ifndef HAVE_OPENGL
    throw_nogl();
    return GpuMat();
#else
    #ifndef HAVE_CUDA
        throw_nogpu();
        return GpuMat();
    #else
        return impl_->mapDevice();
    #endif
#endif
}

void cv::gpu::GlBuffer::unmapDevice()
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    #ifndef HAVE_CUDA
        throw_nogpu();
    #else
        impl_->unmapDevice();
    #endif
#endif
}

int cv::gpu::GlBuffer::rows() const
{
#ifndef HAVE_OPENGL
    return 0;
#else
    return impl_->rows();
#endif
}

int cv::gpu::GlBuffer::cols() const
{
#ifndef HAVE_OPENGL
    return 0;
#else
    return impl_->cols();
#endif
}

Size cv::gpu::GlBuffer::size() const
{
    return Size(cols(), rows());
}

bool cv::gpu::GlBuffer::empty() const
{
    return rows() == 0 || cols() == 0;
}

int cv::gpu::GlBuffer::type() const
{
#ifndef HAVE_OPENGL
    return 0;
#else
    return impl_->type();
#endif
}

int cv::gpu::GlBuffer::depth() const 
{ 
    return CV_MAT_DEPTH(type()); 
}

int cv::gpu::GlBuffer::channels() const 
{ 
    return CV_MAT_CN(type()); 
}

int cv::gpu::GlBuffer::elemSize() const 
{ 
    return CV_ELEM_SIZE(type()); 
}

int cv::gpu::GlBuffer::elemSize1() const 
{ 
    return CV_ELEM_SIZE1(type()); 
}

GlBuffer::Usage cv::gpu::GlBuffer::usage() const
{
#ifndef HAVE_OPENGL
    return ARRAY_BUFFER;
#else
    return static_cast<Usage>(impl_->target());
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////
// GlTexture

#ifndef HAVE_OPENGL

class cv::gpu::GlTexture::Impl
{
};

#else

class cv::gpu::GlTexture::Impl
{
public:
    Impl();

    Impl(int rows, int cols, int type);

    Impl(const Mat& mat, bool bgra);
    Impl(const GlBuffer& buf, bool bgra);

    ~Impl();

    void copyFrom(const Mat& mat, bool bgra);
    void copyFrom(const GlBuffer& buf, bool bgra);

    void bind() const;
    void unbind() const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int type() const { return type_; }

private:    
    int rows_;
    int cols_;
    int type_;
    unsigned int tex_;
};

cv::gpu::GlTexture::Impl::Impl() : rows_(0), cols_(0), type_(0), tex_(0)
{
}

cv::gpu::GlTexture::Impl::Impl(int rows, int cols, int type) : rows_(0), cols_(0), type_(0), tex_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl();

    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);

    CV_DbgAssert(rows > 0 && cols > 0);
    CV_Assert(cn == 1 || cn == 3 || cn == 4);
    CV_Assert(depth >= 0 && depth <= CV_32F);

    GLuint tex;
    glGenTextures(1, &tex);
    CV_CheckGlError();

    glBindTexture(GL_TEXTURE_2D, tex);
    CV_CheckGlError();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    CV_CheckGlError();

    GLenum format = cn == 1 ? GL_LUMINANCE : cn == 3 ? GL_BGR : GL_BGRA;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    glTexImage2D(GL_TEXTURE_2D, 0, cn, cols, rows, 0, format, gl_types[depth], 0);
    CV_CheckGlError();

    rows_ = rows;
    cols_ = cols;
    type_ = type;
    tex_ = tex;
}

cv::gpu::GlTexture::Impl::Impl(const Mat& mat, bool bgra) : rows_(0), cols_(0), type_(0), tex_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl();

    int depth = mat.depth();
    int cn = mat.channels();

    CV_DbgAssert(mat.rows > 0 && mat.cols > 0);
    CV_Assert(cn == 1 || cn == 3 || cn == 4);
    CV_Assert(depth >= 0 && depth <= CV_32F);
    CV_Assert(mat.isContinuous());

    GLuint tex;
    glGenTextures(1, &tex);
    CV_CheckGlError();

    glBindTexture(GL_TEXTURE_2D, tex);
    CV_CheckGlError();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    CV_CheckGlError();

    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    glTexImage2D(GL_TEXTURE_2D, 0, cn, mat.cols, mat.rows, 0, format, gl_types[depth], mat.data);
    CV_CheckGlError();

    rows_ = mat.rows;
    cols_ = mat.cols;
    type_ = mat.type();
    tex_ = tex;
}

cv::gpu::GlTexture::Impl::Impl(const GlBuffer& buf, bool bgra) : rows_(0), cols_(0), type_(0), tex_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl();

    int depth = buf.depth();
    int cn = buf.channels();

    CV_DbgAssert(buf.rows() > 0 && buf.cols() > 0);
    CV_Assert(cn == 1 || cn == 3 || cn == 4);
    CV_Assert(depth >= 0 && depth <= CV_32F);
    CV_Assert(buf.usage() == GlBuffer::TEXTURE_BUFFER);

    GLuint tex;
    glGenTextures(1, &tex);
    CV_CheckGlError();

    glBindTexture(GL_TEXTURE_2D, tex);
    CV_CheckGlError();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    CV_CheckGlError();

    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    buf.bind();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    glTexImage2D(GL_TEXTURE_2D, 0, cn, buf.cols(), buf.rows(), 0, format, gl_types[depth], 0);
    CV_CheckGlError();

    buf.unbind();

    rows_ = buf.rows();
    cols_ = buf.cols();
    type_ = buf.type();
    tex_ = tex;
}

cv::gpu::GlTexture::Impl::~Impl()
{
    if (tex_)
        glDeleteTextures(1, &tex_);
}

void cv::gpu::GlTexture::Impl::copyFrom(const Mat& mat, bool bgra)
{
    CV_Assert(tex_ != 0);
    CV_DbgAssert(mat.cols == cols_ && mat.rows == rows_ && mat.type() == type_);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    int cn = mat.channels();
    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cols_, rows_, format, gl_types[mat.depth()], mat.data);
    CV_CheckGlError();
}

void cv::gpu::GlTexture::Impl::copyFrom(const GlBuffer& buf, bool bgra)
{
    CV_Assert(tex_ != 0);
    CV_DbgAssert(buf.cols() == cols_ && buf.rows() == rows_ && buf.type() == type_);
    CV_Assert(buf.usage() == GlBuffer::TEXTURE_BUFFER);

    buf.bind();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    int cn = buf.channels();
    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cols_, rows_, format, gl_types[buf.depth()], 0);
    CV_CheckGlError();

    buf.unbind();
}

void cv::gpu::GlTexture::Impl::bind() const
{
    CV_Assert(tex_ != 0);

    glEnable(GL_TEXTURE_2D);
    CV_CheckGlError();

    glBindTexture(GL_TEXTURE_2D, tex_);
    CV_CheckGlError();
}

void cv::gpu::GlTexture::Impl::unbind() const
{
    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_TEXTURE_2D);
}

#endif // HAVE_OPENGL

cv::gpu::GlTexture::GlTexture()
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl;
#endif
}

cv::gpu::GlTexture::GlTexture(int rows, int cols, int type)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(rows, cols, type);
#endif
}

cv::gpu::GlTexture::GlTexture(Size size, int type)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(size.height, size.width, type);
#endif
}

cv::gpu::GlTexture::GlTexture(const Mat& mat, bool bgra)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(mat, bgra);
#endif
}

cv::gpu::GlTexture::GlTexture(const GlBuffer& buf, bool bgra)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl(buf, bgra);
#endif
}

cv::gpu::GlTexture::~GlTexture()
{
}

void cv::gpu::GlTexture::create(int rows_, int cols_, int type_)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    if (rows_ != rows() || cols_ != cols() || type_ != type())
        impl_ = new Impl(rows_, cols_, type_);
#endif
}

void cv::gpu::GlTexture::release()
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_ = new Impl;
#endif
}

void cv::gpu::GlTexture::copyFrom(const Mat& mat, bool bgra)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    create(mat.rows, mat.cols, mat.type());
    impl_->copyFrom(mat, bgra);
#endif
}

void cv::gpu::GlTexture::copyFrom(const GlBuffer& buf, bool bgra)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    create(buf.rows(), buf.cols(), buf.type());
    impl_->copyFrom(buf, bgra);
#endif
}

void cv::gpu::GlTexture::bind() const
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_->bind();
#endif
}

void cv::gpu::GlTexture::unbind() const
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    impl_->unbind();
#endif
}

int cv::gpu::GlTexture::rows() const
{
#ifndef HAVE_OPENGL
    return 0;
#else
    return impl_->rows();
#endif
}

int cv::gpu::GlTexture::cols() const
{
#ifndef HAVE_OPENGL
    return 0;
#else
    return impl_->cols();
#endif
}

Size cv::gpu::GlTexture::size() const
{
    return Size(cols(), rows());
}

bool cv::gpu::GlTexture::empty() const
{
    return rows() == 0 || cols() == 0;
}

int cv::gpu::GlTexture::type() const
{
#ifndef HAVE_OPENGL
    return 0;
#else
    return impl_->type();
#endif
}

int cv::gpu::GlTexture::depth() const 
{ 
    return CV_MAT_DEPTH(type()); 
}

int cv::gpu::GlTexture::channels() const 
{ 
    return CV_MAT_CN(type()); 
}

int cv::gpu::GlTexture::elemSize() const 
{ 
    return CV_ELEM_SIZE(type()); 
}

int cv::gpu::GlTexture::elemSize1() const 
{ 
    return CV_ELEM_SIZE1(type()); 
}

////////////////////////////////////////////////////////////////////////
// Rendering

void cv::gpu::render(const GlTexture& tex)
{
#ifndef HAVE_OPENGL
    throw_nogl();
#else
    if (!tex.empty())
    {
        tex.bind();

        glDisable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 1, 1, 0, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        glBegin(GL_QUADS);
            glVertex2d(0.0, 0.0);
            glTexCoord2d(1.0, 0.0);	 

            glVertex2d(1.0, 0.0);
            glTexCoord2d(1.0, 1.0);

            glVertex2d(1.0, 1.0);
            glTexCoord2d(0.0, 1.0);	 

            glVertex2d(0.0, 1.0);
            glTexCoord2d(0.0, 0.0);	
        glEnd();

        CV_CheckGlError();

        tex.unbind();
    }
#endif
}

////////////////////////////////////////////////////////////////////////
// Error handling

void cv::gpu::error(const char *error_string, const char *file, const int line, const char *func)
{
    int code = CV_GpuApiCallError;

    if (uncaught_exception())
    {
        const char* errorStr = cvErrorStr(code);            
        const char* function = func ? func : "unknown function";    

        cerr << "OpenCV Error: " << errorStr << "(" << error_string << ") in " << function << ", file " << file << ", line " << line;
        cerr.flush();            
    }
    else    
        cv::error( cv::Exception(code, error_string, func, file, line) );
}

bool cv::gpu::checkGlError(const char* file, const int line, const char* func)
{
#ifndef HAVE_OPENGL
    return true;
#else
    GLenum err = glGetError();

    if (err != GL_NO_ERROR)
    {
        const char* msg;

        switch (err)
        {
        case GL_INVALID_ENUM:
            msg = "An unacceptable value is specified for an enumerated argument";
            break;
        case GL_INVALID_VALUE:
            msg = "A numeric argument is out of range";
            break;
        case GL_INVALID_OPERATION:
            msg = "The specified operation is not allowed in the current state";
            break;
        case GL_STACK_OVERFLOW:
            msg = "This command would cause a stack overflow";
            break;
        case GL_STACK_UNDERFLOW:
            msg = "This command would cause a stack underflow";
            break;
        case GL_OUT_OF_MEMORY:
            msg = "There is not enough memory left to execute the command";
            break;
        default:
            msg = "Unknown error";
        };

        cvError(CV_OpenGlApiCallError, func, msg, file, line);

        return false;
    }

    return true;
#endif
}
