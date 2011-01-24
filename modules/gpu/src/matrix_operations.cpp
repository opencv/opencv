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


#if !defined (HAVE_CUDA)

namespace cv
{
    namespace gpu
    {
        void GpuMat::upload(const Mat& /*m*/) { throw_nogpu(); }
        void GpuMat::download(cv::Mat& /*m*/) const { throw_nogpu(); }
        void GpuMat::copyTo( GpuMat& /*m*/ ) const { throw_nogpu(); }
        void GpuMat::copyTo( GpuMat& /*m*/, const GpuMat&/* mask */) const { throw_nogpu(); }
        void GpuMat::convertTo( GpuMat& /*m*/, int /*rtype*/, double /*alpha*/, double /*beta*/ ) const { throw_nogpu(); }
        GpuMat& GpuMat::operator = (const Scalar& /*s*/) { throw_nogpu(); return *this; }
        GpuMat& GpuMat::setTo(const Scalar& /*s*/, const GpuMat& /*mask*/) { throw_nogpu(); return *this; }
        GpuMat GpuMat::reshape(int /*new_cn*/, int /*new_rows*/) const { throw_nogpu(); return GpuMat(); }
        void GpuMat::create(int /*_rows*/, int /*_cols*/, int /*_type*/) { throw_nogpu(); }
        void GpuMat::release() { throw_nogpu(); }

        void createContinuous(int /*rows*/, int /*cols*/, int /*type*/, GpuMat& /*m*/) { throw_nogpu(); }

        void CudaMem::create(int /*_rows*/, int /*_cols*/, int /*_type*/, int /*type_alloc*/) { throw_nogpu(); }
        bool CudaMem::canMapHostMemory() { throw_nogpu(); return false; }
        void CudaMem::release() { throw_nogpu(); }
        GpuMat CudaMem::createGpuMatHeader () const { throw_nogpu(); return GpuMat(); }
    }

}

#else /* !defined (HAVE_CUDA) */

namespace cv
{
    namespace gpu
    {
        namespace matrix_operations
        {
            void copy_to_with_mask(const DevMem2D& src, DevMem2D dst, int depth, const DevMem2D& mask, int channels, const cudaStream_t & stream = 0);

            void set_to_without_mask (DevMem2D dst, int depth, const double *scalar, int channels, const cudaStream_t & stream = 0);
            void set_to_with_mask    (DevMem2D dst, int depth, const double *scalar, const DevMem2D& mask, int channels, const cudaStream_t & stream = 0);

            void convert_gpu(const DevMem2D& src, int sdepth, const DevMem2D& dst, int ddepth, double alpha, double beta, cudaStream_t stream = 0);
        }
    }
}

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

void cv::gpu::GpuMat::copyTo( GpuMat& m ) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    cudaSafeCall( cudaMemcpy2D(m.data, m.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice) );
    cudaSafeCall( cudaThreadSynchronize() );
}

void cv::gpu::GpuMat::copyTo( GpuMat& mat, const GpuMat& mask ) const
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
            nppSafeCall( func(src.ptr<src_t>(), src.step, dst.ptr<dst_t>(), dst.step, sz) );
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
            nppSafeCall( func(src.ptr<Npp32f>(), src.step, dst.ptr<dst_t>(), dst.step, sz, NPP_RND_NEAR) );
        }
    };

    void convertToKernelCaller(const GpuMat& src, GpuMat& dst)
    {
        matrix_operations::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), 1.0, 0.0);
    }
}

void cv::gpu::GpuMat::convertTo( GpuMat& dst, int rtype, double alpha, double beta ) const
{
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
            nppSafeCall( func(nppS.val, src.ptr<src_t>(), src.step, sz) );
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
            nppSafeCall( func(nppS[0], src.ptr<src_t>(), src.step, sz) );
        }
    };

    void kernelSet(GpuMat& src, const Scalar& s)
    {
        matrix_operations::set_to_without_mask(src, src.depth(), s.val, src.channels());
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
            nppSafeCall( func(nppS.val, src.ptr<src_t>(), src.step, sz, mask.ptr<Npp8u>(), mask.step) );
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
            nppSafeCall( func(nppS[0], src.ptr<src_t>(), src.step, sz, mask.ptr<Npp8u>(), mask.step) );
        }
    };

    void kernelSetMask(GpuMat& src, const Scalar& s, const GpuMat& mask)
    {
        matrix_operations::set_to_with_mask(src, src.depth(), s.val, mask, src.channels());
    }
}

GpuMat& GpuMat::setTo(const Scalar& s, const GpuMat& mask)
{
    CV_Assert(mask.type() == CV_8UC1);

    CV_DbgAssert(!this->empty());

    NppiSize sz;
    sz.width  = cols;
    sz.height = rows;

    if (mask.empty())
    {
        typedef void (*set_caller_t)(GpuMat& src, const Scalar& s);
        static const set_caller_t set_callers[8][4] =
        {
            {NppSet<CV_8U, 1, nppiSet_8u_C1R>::set,kernelSet,kernelSet,NppSet<CV_8U, 4, nppiSet_8u_C4R>::set},
            {kernelSet,kernelSet,kernelSet,kernelSet},
            {NppSet<CV_16U, 1, nppiSet_16u_C1R>::set,kernelSet,kernelSet,NppSet<CV_16U, 4, nppiSet_16u_C4R>::set},
            {NppSet<CV_16S, 1, nppiSet_16s_C1R>::set,kernelSet,kernelSet,NppSet<CV_16S, 4, nppiSet_16s_C4R>::set},
            {NppSet<CV_32S, 1, nppiSet_32s_C1R>::set,kernelSet,kernelSet,NppSet<CV_32S, 4, nppiSet_32s_C4R>::set},
            {NppSet<CV_32F, 1, nppiSet_32f_C1R>::set,kernelSet,kernelSet,NppSet<CV_32F, 4, nppiSet_32f_C4R>::set},
            {kernelSet,kernelSet,kernelSet,kernelSet},
            {0,0,0,0}
        };
        set_callers[depth()][channels()-1](*this, s);
    }
    else
    {
        typedef void (*set_caller_t)(GpuMat& src, const Scalar& s, const GpuMat& mask);
        static const set_caller_t set_callers[8][4] =
        {
            {NppSetMask<CV_8U, 1, nppiSet_8u_C1MR>::set,kernelSetMask,kernelSetMask,NppSetMask<CV_8U, 4, nppiSet_8u_C4MR>::set},
            {kernelSetMask,kernelSetMask,kernelSetMask,kernelSetMask},
            {NppSetMask<CV_16U, 1, nppiSet_16u_C1MR>::set,kernelSetMask,kernelSetMask,NppSetMask<CV_16U, 4, nppiSet_16u_C4MR>::set},
            {NppSetMask<CV_16S, 1, nppiSet_16s_C1MR>::set,kernelSetMask,kernelSetMask,NppSetMask<CV_16S, 4, nppiSet_16s_C4MR>::set},
            {NppSetMask<CV_32S, 1, nppiSet_32s_C1MR>::set,kernelSetMask,kernelSetMask,NppSetMask<CV_32S, 4, nppiSet_32s_C4MR>::set},
            {NppSetMask<CV_32F, 1, nppiSet_32f_C1MR>::set,kernelSetMask,kernelSetMask,NppSetMask<CV_32F, 4, nppiSet_32f_C4MR>::set},
            {kernelSetMask,kernelSetMask,kernelSetMask,kernelSetMask},
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
        return;
    m.create(rows, cols, type);
}


///////////////////////////////////////////////////////////////////////
//////////////////////////////// CudaMem //////////////////////////////
///////////////////////////////////////////////////////////////////////

bool cv::gpu::CudaMem::canMapHostMemory()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return (prop.canMapHostMemory != 0) ? true : false;
}

void cv::gpu::CudaMem::create(int _rows, int _cols, int _type, int _alloc_type)
{
    if (_alloc_type == ALLOC_ZEROCOPY && !canMapHostMemory())
            cv::gpu::error("ZeroCopy is not supported by current device", __FILE__, __LINE__);

    _type &= TYPE_MASK;
    if( rows == _rows && cols == _cols && type() == _type && data )
        return;
    if( data )
        release();
    CV_DbgAssert( _rows >= 0 && _cols >= 0 );
    if( _rows > 0 && _cols > 0 )
    {
        flags = Mat::MAGIC_VAL + Mat::CONTINUOUS_FLAG + _type;
        rows = _rows;
        cols = _cols;
        step = elemSize()*cols;
        int64 _nettosize = (int64)step*rows;
        size_t nettosize = (size_t)_nettosize;
        if( _nettosize != (int64)nettosize )
            CV_Error(CV_StsNoMem, "Too big buffer is allocated");
        size_t datasize = alignSize(nettosize, (int)sizeof(*refcount));

        //datastart = data = (uchar*)fastMalloc(datasize + sizeof(*refcount));
        alloc_type = _alloc_type;
        void *ptr;

        switch (alloc_type)
        {
            case ALLOC_PAGE_LOCKED:    cudaSafeCall( cudaHostAlloc( &ptr, datasize, cudaHostAllocDefault) ); break;
            case ALLOC_ZEROCOPY:       cudaSafeCall( cudaHostAlloc( &ptr, datasize, cudaHostAllocMapped) );  break;
            case ALLOC_WRITE_COMBINED: cudaSafeCall( cudaHostAlloc( &ptr, datasize, cudaHostAllocWriteCombined) ); break;
            default: cv::gpu::error("Invalid alloc type", __FILE__, __LINE__);
        }

        datastart = data =  (uchar*)ptr;
        dataend = data + nettosize;

        refcount = (int*)cv::fastMalloc(sizeof(*refcount));
        *refcount = 1;
    }
}

GpuMat cv::gpu::CudaMem::createGpuMatHeader () const
{
    GpuMat res;
    if (alloc_type == ALLOC_ZEROCOPY)
    {
        void *pdev;
        cudaSafeCall( cudaHostGetDevicePointer( &pdev, data, 0 ) );
        res = GpuMat(rows, cols, type(), pdev, step);
    }
    else
        cv::gpu::error("Zero-copy is not supported or memory was allocated without zero-copy flag", __FILE__, __LINE__);

    return res;
}

void cv::gpu::CudaMem::release()
{
    if( refcount && CV_XADD(refcount, -1) == 1 )
    {
        cudaSafeCall( cudaFreeHost(datastart ) );
        fastFree(refcount);
    }
    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}

#endif /* !defined (HAVE_CUDA) */
