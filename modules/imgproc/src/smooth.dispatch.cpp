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
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include <opencv2/core/utils/logger.hpp>

#include <opencv2/core/utils/configuration.private.hpp>

#include <vector>
#include <iostream>

#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

#include "filter.hpp"

#include "opencv2/core/softfloat.hpp"

namespace cv {
#include "fixedpoint.inl.hpp"
}

#include "smooth.simd.hpp"
#include "smooth.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

/****************************************************************************************\
                                     Gaussian Blur
\****************************************************************************************/

/**
 * Bit-exact in terms of softfloat computations
 *
 * returns sum of kernel values. Should be equal to 1.0
 */
static
softdouble getGaussianKernelBitExact(std::vector<softdouble>& result, int n, double sigma)
{
    CV_Assert(n > 0);
    //TODO: incorrect SURF implementation requests kernel with n = 20 (PATCH_SZ): https://github.com/opencv/opencv/issues/15856
    //CV_Assert((n & 1) == 1);  // odd

    if (sigma <= 0)
    {
        if (n == 1)
        {
            result = std::vector<softdouble>(1, softdouble::one());
            return softdouble::one();
        }
        else if (n == 3)
        {
            softdouble v3[] = {
                softdouble::fromRaw(0x3fd0000000000000),  // 0.25
                softdouble::fromRaw(0x3fe0000000000000),  // 0.5
                softdouble::fromRaw(0x3fd0000000000000)   // 0.25
            };
            result.assign(v3, v3 + 3);
            return softdouble::one();
        }
        else if (n == 5)
        {
            softdouble v5[] = {
                softdouble::fromRaw(0x3fb0000000000000),  // 0.0625
                softdouble::fromRaw(0x3fd0000000000000),  // 0.25
                softdouble::fromRaw(0x3fd8000000000000),  // 0.375
                softdouble::fromRaw(0x3fd0000000000000),  // 0.25
                softdouble::fromRaw(0x3fb0000000000000)   // 0.0625
            };
            result.assign(v5, v5 + 5);
            return softdouble::one();
        }
        else if (n == 7)
        {
            softdouble v7[] = {
                softdouble::fromRaw(0x3fa0000000000000),  // 0.03125
                softdouble::fromRaw(0x3fbc000000000000),  // 0.109375
                softdouble::fromRaw(0x3fcc000000000000),  // 0.21875
                softdouble::fromRaw(0x3fd2000000000000),  // 0.28125
                softdouble::fromRaw(0x3fcc000000000000),  // 0.21875
                softdouble::fromRaw(0x3fbc000000000000),  // 0.109375
                softdouble::fromRaw(0x3fa0000000000000)   // 0.03125
            };
            result.assign(v7, v7 + 7);
            return softdouble::one();
        }
        else if (n == 9)
        {
            softdouble v9[] = {
                softdouble::fromRaw(0x3f90000000000000),  // 4  / 256
                softdouble::fromRaw(0x3faa000000000000),  // 13 / 256
                softdouble::fromRaw(0x3fbe000000000000),  // 30 / 256
                softdouble::fromRaw(0x3fc9800000000000),  // 51 / 256
                softdouble::fromRaw(0x3fce000000000000),  // 60 / 256
                softdouble::fromRaw(0x3fc9800000000000),  // 51 / 256
                softdouble::fromRaw(0x3fbe000000000000),  // 30 / 256
                softdouble::fromRaw(0x3faa000000000000),  // 13 / 256
                softdouble::fromRaw(0x3f90000000000000)   // 4  / 256
            };
            result.assign(v9, v9 + 9);
            return softdouble::one();
        }
    }

    softdouble sd_0_15 = softdouble::fromRaw(0x3fc3333333333333);  // 0.15
    softdouble sd_0_35 = softdouble::fromRaw(0x3fd6666666666666);  // 0.35
    softdouble sd_minus_0_125 = softdouble::fromRaw(0xbfc0000000000000);  // -0.5*0.25

    softdouble sigmaX = sigma > 0 ? softdouble(sigma) : mulAdd(softdouble(n), sd_0_15, sd_0_35);// softdouble(((n-1)*0.5 - 1)*0.3 + 0.8)
    softdouble scale2X = sd_minus_0_125/(sigmaX*sigmaX);

    int n2_ = (n - 1) / 2;
    cv::AutoBuffer<softdouble> values(n2_ + 1);
    softdouble sum = softdouble::zero();
    for (int i = 0, x = 1 - n; i < n2_; i++, x+=2)
    {
        // x = i - (n - 1)*0.5
        // t = std::exp(scale2X*x*x)
        softdouble t = exp(softdouble(x*x)*scale2X);
        values[i] = t;
        sum += t;
    }
    sum *= softdouble(2);
    //values[n2_] = softdouble::one(); // x=0 in exp(softdouble(x*x)*scale2X);
    sum += softdouble::one();
    if ((n & 1) == 0)
    {
        //values[n2_ + 1] = softdouble::one();
        sum += softdouble::one();
    }

    // normalize: sum(k[i]) = 1
    softdouble mul1 = softdouble::one()/sum;

    result.resize(n);

    softdouble sum2 = softdouble::zero();
    for (int i = 0; i < n2_; i++ )
    {
        softdouble t = values[i] * mul1;
        result[i] = t;
        result[n - 1 - i] = t;
        sum2 += t;
    }
    sum2 *= softdouble(2);
    result[n2_] = /*values[n2_]*/ softdouble::one() * mul1;
    sum2 += result[n2_];
    if ((n & 1) == 0)
    {
        result[n2_ + 1] = result[n2_];
        sum2 += result[n2_];
    }

    return sum2;
}

Mat getGaussianKernel(int n, double sigma, int ktype)
{
    CV_CheckDepth(ktype, ktype == CV_32F || ktype == CV_64F, "");
    Mat kernel(n, 1, ktype);

    std::vector<softdouble> kernel_bitexact;
    getGaussianKernelBitExact(kernel_bitexact, n, sigma);

    if (ktype == CV_32F)
    {
        for (int i = 0; i < n; i++)
            kernel.at<float>(i) = (float)kernel_bitexact[i];
    }
    else
    {
        CV_DbgAssert(ktype == CV_64F);
        for (int i = 0; i < n; i++)
            kernel.at<double>(i) = kernel_bitexact[i];
    }

    return kernel;
}

static
softdouble getGaussianKernelFixedPoint_ED(CV_OUT std::vector<int64_t>& result, const std::vector<softdouble> kernel_bitexact, int fractionBits)
{
    const int n = (int)kernel_bitexact.size();
    CV_Assert((n & 1) == 1);  // odd

    CV_CheckGT(fractionBits, 0, "");
    CV_CheckLE(fractionBits, 32, "");

    int64_t fractionMultiplier = CV_BIG_INT(1) << fractionBits;
    softdouble fractionMultiplier_sd(fractionMultiplier);

    result.resize(n);

    int n2_ = n / 2;  // n is odd
    softdouble err = softdouble::zero();
    int64_t sum = 0;
    for (int i = 0; i < n2_; i++)
    {
        //softdouble err0 = err;
        softdouble adj_v = kernel_bitexact[i] * fractionMultiplier_sd + err;
        int64_t v0 = cvRound(adj_v);  // cvFloor() provides bad results
        err = adj_v - softdouble(v0);
        //printf("%3d: adj_v=%8.3f(%8.3f+%8.3f)  v0=%d   ed_err=%8.3f\n", i, (double)adj_v, (double)(kernel_bitexact[i] * fractionMultiplier_sd), (double)err0, (int)v0, (double)err);

        result[i] = v0;
        result[n - 1 - i] = v0;
        sum += v0;
    }
    sum *= 2;
    softdouble adj_v_center = kernel_bitexact[n2_] * fractionMultiplier_sd + err;
    int64_t v_center = fractionMultiplier - sum;
    result[n2_] = v_center;
    //printf("center = %g ===> %g  ===> %g\n", (double)(kernel_bitexact[n2_] * fractionMultiplier), (double)adj_v_center, (double)v_center);
    return (adj_v_center - softdouble(v_center));
}

static void getGaussianKernel(int n, double sigma, int ktype, Mat& res) { res = getGaussianKernel(n, sigma, ktype); }
template <typename FT> static void getGaussianKernel(int n, double sigma, int, std::vector<FT>& res)
{
    std::vector<softdouble> res_sd;
    softdouble s0 = getGaussianKernelBitExact(res_sd, n, sigma);
    CV_UNUSED(s0);

    std::vector<int64_t> fixed_256;
    softdouble approx_err = getGaussianKernelFixedPoint_ED(fixed_256, res_sd, FT::fixedShift);
    CV_UNUSED(approx_err);

    res.resize(n);
    for (int i = 0; i < n; i++)
    {
        res[i] = FT::fromRaw((typename FT::raw_t)fixed_256[i]);
        //printf("%03d: %d\n", i, res[i].raw());
    }
}

template <typename T>
static void createGaussianKernels( T & kx, T & ky, int type, Size &ksize,
                                   double sigma1, double sigma2 )
{
    int depth = CV_MAT_DEPTH(type);
    if( sigma2 <= 0 )
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    if( ksize.height <= 0 && sigma2 > 0 )
        ksize.height = cvRound(sigma2*(depth == CV_8U ? 3 : 4)*2 + 1)|1;

    CV_Assert( ksize.width  > 0 && ksize.width  % 2 == 1 &&
               ksize.height > 0 && ksize.height % 2 == 1 );

    sigma1 = std::max( sigma1, 0. );
    sigma2 = std::max( sigma2, 0. );

    getGaussianKernel( ksize.width, sigma1, std::max(depth, CV_32F), kx );
    if( ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON )
        ky = kx;
    else
        getGaussianKernel( ksize.height, sigma2, std::max(depth, CV_32F), ky );
}

Ptr<FilterEngine> createGaussianFilter( int type, Size ksize,
                                        double sigma1, double sigma2,
                                        int borderType )
{
    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);

    return createSeparableLinearFilter( type, type, kx, ky, Point(-1,-1), 0, borderType );
}

#ifdef HAVE_OPENCL

static bool ocl_GaussianBlur_8UC1(InputArray _src, OutputArray _dst, Size ksize, int ddepth,
                                  InputArray _kernelX, InputArray _kernelY, int borderType)
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if ( !(dev.isIntel() && (type == CV_8UC1) &&
         (_src.offset() == 0) && (_src.step() % 4 == 0) &&
         ((ksize.width == 5 && (_src.cols() % 4 == 0)) ||
         (ksize.width == 3 && (_src.cols() % 16 == 0) && (_src.rows() % 2 == 0)))) )
        return false;

    Mat kernelX = _kernelX.getMat().reshape(1, 1);
    if (kernelX.cols % 2 != 1)
        return false;
    Mat kernelY = _kernelY.getMat().reshape(1, 1);
    if (kernelY.cols % 2 != 1)
        return false;

    if (ddepth < 0)
        ddepth = sdepth;

    Size size = _src.size();
    size_t globalsize[2] = { 0, 0 };
    size_t localsize[2] = { 0, 0 };

    if (ksize.width == 3)
    {
        globalsize[0] = size.width / 16;
        globalsize[1] = size.height / 2;
    }
    else if (ksize.width == 5)
    {
        globalsize[0] = size.width / 4;
        globalsize[1] = size.height / 1;
    }

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", 0, "BORDER_REFLECT_101" };
    char build_opts[1024];
    snprintf(build_opts, sizeof(build_opts), "-D %s %s%s", borderMap[borderType & ~BORDER_ISOLATED],
            ocl::kernelToStr(kernelX, CV_32F, "KERNEL_MATRIX_X").c_str(),
            ocl::kernelToStr(kernelY, CV_32F, "KERNEL_MATRIX_Y").c_str());

    ocl::Kernel kernel;

    if (ksize.width == 3)
        kernel.create("gaussianBlur3x3_8UC1_cols16_rows2", cv::ocl::imgproc::gaussianBlur3x3_oclsrc, build_opts);
    else if (ksize.width == 5)
        kernel.create("gaussianBlur5x5_8UC1_cols4", cv::ocl::imgproc::gaussianBlur5x5_oclsrc, build_opts);

    if (kernel.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, CV_MAKETYPE(ddepth, cn));
    if (!(_dst.offset() == 0 && _dst.step() % 4 == 0))
        return false;
    UMat dst = _dst.getUMat();

    int idxArg = kernel.set(0, ocl::KernelArg::PtrReadOnly(src));
    idxArg = kernel.set(idxArg, (int)src.step);
    idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
    idxArg = kernel.set(idxArg, (int)dst.step);
    idxArg = kernel.set(idxArg, (int)dst.rows);
    idxArg = kernel.set(idxArg, (int)dst.cols);

    return kernel.run(2, globalsize, (localsize[0] == 0) ? NULL : localsize, false);
}

#endif

#ifdef HAVE_OPENVX

namespace ovx {
    template <> inline bool skipSmallImages<VX_KERNEL_GAUSSIAN_3x3>(int w, int h) { return w*h < 320 * 240; }
}
static bool openvx_gaussianBlur(InputArray _src, OutputArray _dst, Size ksize,
                                double sigma1, double sigma2, int borderType)
{
    if (sigma2 <= 0)
        sigma2 = sigma1;
    // automatic detection of kernel size from sigma
    if (ksize.width <= 0 && sigma1 > 0)
        ksize.width = cvRound(sigma1*6 + 1) | 1;
    if (ksize.height <= 0 && sigma2 > 0)
        ksize.height = cvRound(sigma2*6 + 1) | 1;

    if (_src.type() != CV_8UC1 ||
        _src.cols() < 3 || _src.rows() < 3 ||
        ksize.width != 3 || ksize.height != 3)
        return false;

    sigma1 = std::max(sigma1, 0.);
    sigma2 = std::max(sigma2, 0.);

    if (!(sigma1 == 0.0 || (sigma1 - 0.8) < DBL_EPSILON) || !(sigma2 == 0.0 || (sigma2 - 0.8) < DBL_EPSILON) ||
        ovx::skipSmallImages<VX_KERNEL_GAUSSIAN_3x3>(_src.cols(), _src.rows()))
        return false;

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    if ((borderType & BORDER_ISOLATED) == 0 && src.isSubmatrix())
        return false; //Process isolated borders only
    vx_enum border;
    switch (borderType & ~BORDER_ISOLATED)
    {
    case BORDER_CONSTANT:
        border = VX_BORDER_CONSTANT;
        break;
    case BORDER_REPLICATE:
        border = VX_BORDER_REPLICATE;
        break;
    default:
        return false;
    }

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();

        Mat a;
        if (dst.data != src.data)
            a = src;
        else
            src.copyTo(a);

        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(a.cols, a.rows, 1, (vx_int32)(a.step)), a.data),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(dst.cols, dst.rows, 1, (vx_int32)(dst.step)), dst.data);

        //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
        //since OpenVX standard says nothing about thread-safety for now
        ivx::border_t prevBorder = ctx.immediateBorder();
        ctx.setImmediateBorder(border, (vx_uint8)(0));
        ivx::IVX_CHECK_STATUS(vxuGaussian3x3(ctx, ia, ib));
        ctx.setImmediateBorder(prevBorder);
    }
    catch (const ivx::RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (const ivx::WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }
    return true;
}

#endif

#ifdef ENABLE_IPP_GAUSSIAN_BLUR  // see CMake's OPENCV_IPP_GAUSSIAN_BLUR option

#define IPP_DISABLE_GAUSSIAN_BLUR_LARGE_KERNELS_1TH 1
#define IPP_DISABLE_GAUSSIAN_BLUR_16SC4_1TH 1
#define IPP_DISABLE_GAUSSIAN_BLUR_32FC4_1TH 1

// IW 2017u2 has bug which doesn't allow use of partial inMem with tiling
#if IPP_VERSION_X100 < 201900
#define IPP_GAUSSIANBLUR_PARALLEL 0
#else
#define IPP_GAUSSIANBLUR_PARALLEL 1
#endif

#ifdef HAVE_IPP_IW

class ipp_gaussianBlurParallel: public ParallelLoopBody
{
public:
    ipp_gaussianBlurParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, int kernelSize, float sigma, ::ipp::IwiBorderType &border, bool *pOk):
        m_src(src), m_dst(dst), m_kernelSize(kernelSize), m_sigma(sigma), m_border(border), m_pOk(pOk) {
        *m_pOk = true;
    }
    ~ipp_gaussianBlurParallel()
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION_IPP();

        if(!*m_pOk)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterGaussian, m_src, m_dst, m_kernelSize, m_sigma, ::ipp::IwDefault(), m_border, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *m_pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    int m_kernelSize;
    float m_sigma;
    ::ipp::IwiBorderType &m_border;

    volatile bool *m_pOk;
    const ipp_gaussianBlurParallel& operator= (const ipp_gaussianBlurParallel&);
};

#endif

static bool ipp_GaussianBlur(cv::Mat& src, cv::Mat& dst, Size ksize,
                   double sigma1, double sigma2, int borderType )
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 < 201800 && ((defined _MSC_VER && defined _M_IX86) || (defined __GNUC__ && defined __i386__))
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(ksize); CV_UNUSED(sigma1); CV_UNUSED(sigma2); CV_UNUSED(borderType);
    return false; // bug on ia32
#else
    if(sigma1 != sigma2)
        return false;

    if(sigma1 < FLT_EPSILON)
        return false;

    if(ksize.width != ksize.height)
        return false;

    // Acquire data and begin processing
    try
    {
        ::ipp::IwiImage       iwSrc      = ippiGetImage(src);
        ::ipp::IwiImage       iwDst      = ippiGetImage(dst);
        ::ipp::IwiBorderSize  borderSize = ::ipp::iwiSizeToBorderSize(ippiGetSize(ksize));
        ::ipp::IwiBorderType  ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        const int threads = ippiSuggestThreadsNum(iwDst, 2);

        if (IPP_DISABLE_GAUSSIAN_BLUR_LARGE_KERNELS_1TH && (threads == 1 && ksize.width > 25))
            return false;
        if (IPP_DISABLE_GAUSSIAN_BLUR_16SC4_1TH && (threads == 1 && src.type() == CV_16SC4))
            return false;
        if (IPP_DISABLE_GAUSSIAN_BLUR_32FC4_1TH && (threads == 1 && src.type() == CV_32FC4))
            return false;

        if(IPP_GAUSSIANBLUR_PARALLEL && threads > 1 && iwSrc.m_size.height/(threads * 4) >= ksize.height/2) {
            bool ok;
            ipp_gaussianBlurParallel invoker(iwSrc, iwDst, ksize.width, (float) sigma1, ippBorder, &ok);

            if(!ok)
                return false;
            const Range range(0, (int) iwDst.m_size.height);
            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return false;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterGaussian, iwSrc, iwDst, ksize.width, sigma1, ::ipp::IwDefault(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return false;
    }

    return true;
#endif
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(ksize); CV_UNUSED(sigma1); CV_UNUSED(sigma2); CV_UNUSED(borderType);
    return false;
#endif
}
#endif

template<typename T>
static bool validateGaussianBlurKernel(std::vector<T>& kernel)
{
    softdouble validation_sum = softdouble::zero();
    for (size_t i = 0; i < kernel.size(); i++)
    {
        validation_sum += softdouble((double)kernel[i]);
    }

    bool isValid = validation_sum == softdouble::one();
    return isValid;
}

void GaussianBlur(InputArray _src, OutputArray _dst, Size ksize,
                  double sigma1, double sigma2,
                  int borderType, AlgorithmHint hint)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_src.empty());

    int type = _src.type();
    Size size = _src.size();
    _dst.create( size, type );

    if( (borderType & ~BORDER_ISOLATED) != BORDER_CONSTANT &&
        ((borderType & BORDER_ISOLATED) != 0 || !_src.getMat().isSubmatrix()) )
    {
        if( size.height == 1 )
            ksize.height = 1;
        if( size.width == 1 )
            ksize.width = 1;
    }

    if( ksize.width == 1 && ksize.height == 1 )
    {
        _src.copyTo(_dst);
        return;
    }

    if (sigma2 <= 0)
        sigma2 = sigma1;

    bool useOpenCL = ocl::isOpenCLActivated() && _dst.isUMat() && _src.dims() <= 2 &&
               _src.rows() >= ksize.height && _src.cols() >= ksize.width &&
               ksize.width > 1 && ksize.height > 1;
    CV_UNUSED(useOpenCL);

    int sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);

    CV_OCL_RUN(useOpenCL && sdepth == CV_8U &&
            ((ksize.width == 3 && ksize.height == 3) ||
            (ksize.width == 5 && ksize.height == 5)),
            ocl_GaussianBlur_8UC1(_src, _dst, ksize, CV_MAT_DEPTH(type), kx, ky, borderType)
    );

    if(sdepth == CV_8U && ((borderType & BORDER_ISOLATED) || !_src.isSubmatrix()))
    {
        std::vector<ufixedpoint16> fkx, fky;
        createGaussianKernels(fkx, fky, type, ksize, sigma1, sigma2);

        static bool param_check_gaussian_blur_bitexact_kernels = utils::getConfigurationParameterBool("OPENCV_GAUSSIANBLUR_CHECK_BITEXACT_KERNELS", false);
        if (param_check_gaussian_blur_bitexact_kernels && !validateGaussianBlurKernel(fkx))
        {
            CV_LOG_INFO(NULL, "GaussianBlur: bit-exact fx kernel can't be applied: ksize=" << ksize << " sigma=" << Size2d(sigma1, sigma2));
        }
        else if (param_check_gaussian_blur_bitexact_kernels && !validateGaussianBlurKernel(fky))
        {
            CV_LOG_INFO(NULL, "GaussianBlur: bit-exact fy kernel can't be applied: ksize=" << ksize << " sigma=" << Size2d(sigma1, sigma2));
        }
        else
        {
            CV_OCL_RUN(useOpenCL,
                    ocl_sepFilter2D_BitExact(_src, _dst, sdepth,
                            ksize,
                            (const uint16_t*)&fkx[0], (const uint16_t*)&fky[0],
                            Point(-1, -1), 0, borderType,
                            8/*shift_bits*/)
            );

            Mat src = _src.getMat();
            Mat dst = _dst.getMat();

            if (src.data == dst.data)
                src = src.clone();

            if ((sigma1 == 0.0) && (sigma2 == 0.0) && (ksize.height == ksize.width))
            {
                Point ofs;
                Size wsz(src.cols, src.rows);
                Mat src2 = src;
                if(!(borderType & BORDER_ISOLATED))
                    src2.locateROI( wsz, ofs );

                CALL_HAL(gaussianBlurBinomial, cv_hal_gaussianBlurBinomial, src2.ptr(), src2.step, dst.ptr(), dst.step, src2.cols, src2.rows, sdepth, cn,
                         ofs.x, ofs.y, wsz.width - src2.cols - ofs.x,  wsz.height - src2.rows - ofs.y, ksize.width,
                         borderType & ~BORDER_ISOLATED);
            }

            if (hint == ALGO_ALLOW_APPROXIMATION)
            {
                Point ofs;
                Size wsz(src.cols, src.rows);
                if(!(borderType & BORDER_ISOLATED))
                    src.locateROI( wsz, ofs );

                CALL_HAL(gaussianBlur, cv_hal_gaussianBlur, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, cn,
                        ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, ksize.width, ksize.height,
                        sigma1, sigma2, borderType & ~BORDER_ISOLATED);

#ifdef ENABLE_IPP_GAUSSIAN_BLUR
                // IPP is not bit-exact to OpenCV implementation
                CV_IPP_RUN_FAST(ipp_GaussianBlur(src, dst, ksize, sigma1, sigma2, borderType));
#endif
                CV_OVX_RUN(true,
                        openvx_gaussianBlur(src, dst, ksize, sigma1, sigma2, borderType))
            }

            CV_CPU_DISPATCH(GaussianBlurFixedPoint, (src, dst, (const uint16_t*)&fkx[0], (int)fkx.size(), (const uint16_t*)&fky[0], (int)fky.size(), borderType),
                CV_CPU_DISPATCH_MODES_ALL);

            return;
        }
    }
    if(sdepth == CV_16U && ((borderType & BORDER_ISOLATED) || !_src.isSubmatrix()))
    {
        CV_LOG_INFO(NULL, "GaussianBlur: running bit-exact version...");

        std::vector<ufixedpoint32> fkx, fky;
        createGaussianKernels(fkx, fky, type, ksize, sigma1, sigma2);

        static bool param_check_gaussian_blur_bitexact_kernels = utils::getConfigurationParameterBool("OPENCV_GAUSSIANBLUR_CHECK_BITEXACT_KERNELS", false);
        if (param_check_gaussian_blur_bitexact_kernels && !validateGaussianBlurKernel(fkx))
        {
            CV_LOG_INFO(NULL, "GaussianBlur: bit-exact fx kernel can't be applied: ksize=" << ksize << " sigma=" << Size2d(sigma1, sigma2));
        }
        else if (param_check_gaussian_blur_bitexact_kernels && !validateGaussianBlurKernel(fky))
        {
            CV_LOG_INFO(NULL, "GaussianBlur: bit-exact fy kernel can't be applied: ksize=" << ksize << " sigma=" << Size2d(sigma1, sigma2));
        }
        else
        {
            // TODO: implement ocl_sepFilter2D_BitExact -- how to deal with bdepth?
            // CV_OCL_RUN(useOpenCL,
            //         ocl_sepFilter2D_BitExact(_src, _dst, sdepth,
            //                 ksize,
            //                 (const uint32_t*)&fkx[0], (const uint32_t*)&fky[0],
            //                 Point(-1, -1), 0, borderType,
            //                 16/*shift_bits*/)
            // );

            Mat src = _src.getMat();
            Mat dst = _dst.getMat();

            if (src.data == dst.data)
                src = src.clone();

            if ((sigma1 == 0.0) && (sigma2 == 0.0) && (ksize.height == ksize.width))
            {
                Point ofs;
                Size wsz(src.cols, src.rows);
                Mat src2 = src;
                if(!(borderType & BORDER_ISOLATED))
                    src2.locateROI( wsz, ofs );

                CALL_HAL(gaussianBlurBinomial, cv_hal_gaussianBlurBinomial, src2.ptr(), src2.step, dst.ptr(), dst.step, src2.cols, src2.rows, sdepth, cn,
                         ofs.x, ofs.y, wsz.width - src2.cols - ofs.x,  wsz.height - src2.rows - ofs.y, ksize.width, borderType&~BORDER_ISOLATED);
            }

            if (hint == ALGO_ALLOW_APPROXIMATION)
            {
                Point ofs;
                Size wsz(src.cols, src.rows);
                if(!(borderType & BORDER_ISOLATED))
                    src.locateROI( wsz, ofs );

                CALL_HAL(gaussianBlur, cv_hal_gaussianBlur, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, cn,
                        ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, ksize.width, ksize.height,
                        sigma1, sigma2, borderType & ~BORDER_ISOLATED);

#ifdef ENABLE_IPP_GAUSSIAN_BLUR
                // IPP is not bit-exact to OpenCV implementation
                CV_IPP_RUN_FAST(ipp_GaussianBlur(src, dst, ksize, sigma1, sigma2, borderType));
#endif
                CV_OVX_RUN(true,
                        openvx_gaussianBlur(src, dst, ksize, sigma1, sigma2, borderType))
            }

            CV_CPU_DISPATCH(GaussianBlurFixedPoint, (src, dst, (const uint32_t*)&fkx[0], (int)fkx.size(), (const uint32_t*)&fky[0], (int)fky.size(), borderType),
                CV_CPU_DISPATCH_MODES_ALL);

            return;
        }
    }

#ifdef HAVE_OPENCL
    if (useOpenCL)
    {
        sepFilter2D(_src, _dst, sdepth, kx, ky, Point(-1, -1), 0, borderType);
        return;
    }
#endif

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType & BORDER_ISOLATED))
        src.locateROI( wsz, ofs );

    CALL_HAL(gaussianBlur, cv_hal_gaussianBlur, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, cn,
             ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, ksize.width, ksize.height,
             sigma1, sigma2, borderType & ~BORDER_ISOLATED);

    CV_OVX_RUN(true,
               openvx_gaussianBlur(src, dst, ksize, sigma1, sigma2, borderType))

#if defined ENABLE_IPP_GAUSSIAN_BLUR
    // IPP is not bit-exact to OpenCV implementation
    CV_IPP_RUN_FAST(ipp_GaussianBlur(src, dst, ksize, sigma1, sigma2, borderType));
#endif

    sepFilter2D(src, dst, sdepth, kx, ky, Point(-1, -1), 0, borderType);
}

} // namespace

//////////////////////////////////////////////////////////////////////////////////////////

CV_IMPL void
cvSmooth( const void* srcarr, void* dstarr, int smooth_type,
          int param1, int param2, double param3, double param4 )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;

    CV_Assert( dst.size() == src.size() &&
        (smooth_type == CV_BLUR_NO_SCALE || dst.type() == src.type()) );

    if( param2 <= 0 )
        param2 = param1;

    if( smooth_type == CV_BLUR || smooth_type == CV_BLUR_NO_SCALE )
        cv::boxFilter( src, dst, dst.depth(), cv::Size(param1, param2), cv::Point(-1,-1),
            smooth_type == CV_BLUR, cv::BORDER_REPLICATE );
    else if( smooth_type == CV_GAUSSIAN )
        cv::GaussianBlur( src, dst, cv::Size(param1, param2), param3, param4, cv::BORDER_REPLICATE );
    else if( smooth_type == CV_MEDIAN )
        cv::medianBlur( src, dst, param1 );
    else
        cv::bilateralFilter( src, dst, param1, param3, param4, cv::BORDER_REPLICATE );

    if( dst.data != dst0.data )
        CV_Error( cv::Error::StsUnmatchedFormats, "The destination image does not have the proper type" );
}

/* End of file. */
