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
// Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
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

#include <vector>

#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "bilateral_filter.simd.hpp"
#include "bilateral_filter.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

/****************************************************************************************\
                                   Bilateral Filtering
\****************************************************************************************/

namespace cv {

#ifdef HAVE_OPENCL

static bool ocl_bilateralFilter_8u(InputArray _src, OutputArray _dst, int d,
                                   double sigma_color, double sigma_space,
                                   int borderType)
{
    CV_INSTRUMENT_REGION();
#ifdef __ANDROID__
    if (ocl::Device::getDefault().isNVidia())
        return false;
#endif

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int i, j, maxk, radius;

    if (depth != CV_8U || cn > 4)
        return false;

    if (sigma_color <= 0)
        sigma_color = 1;
    if (sigma_space <= 0)
        sigma_space = 1;

    double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    if ( d <= 0 )
        radius = cvRound(sigma_space * 1.5);
    else
        radius = d / 2;
    radius = MAX(radius, 1);
    d = radius * 2 + 1;

    UMat src = _src.getUMat(), dst = _dst.getUMat(), temp;
    if (src.u == dst.u)
        return false;

    copyMakeBorder(src, temp, radius, radius, radius, radius, borderType);
    std::vector<float> _space_weight(d * d);
    std::vector<int> _space_ofs(d * d);
    float * const space_weight = &_space_weight[0];
    int * const space_ofs = &_space_ofs[0];

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i * i + (double)j * j);
            if ( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
            space_ofs[maxk++] = (int)(i * temp.step + j * cn);
        }

    char cvt[3][40];
    String cnstr = cn > 1 ? format("%d", cn) : "";
    String kernelName("bilateral");
    size_t sizeDiv = 1;
    if ((ocl::Device::getDefault().isIntel()) &&
        (ocl::Device::getDefault().type() == ocl::Device::TYPE_GPU))
    {
            //Intel GPU
            if (dst.cols % 4 == 0 && cn == 1) // For single channel x4 sized images.
            {
                kernelName = "bilateral_float4";
                sizeDiv = 4;
            }
     }
     ocl::Kernel k(kernelName.c_str(), ocl::imgproc::bilateral_oclsrc,
            format("-D radius=%d -D maxk=%d -D cn=%d -D int_t=%s -D uint_t=uint%s -D convert_int_t=%s"
            " -D uchar_t=%s -D float_t=%s -D convert_float_t=%s -D convert_uchar_t=%s -D gauss_color_coeff=(float)%f",
            radius, maxk, cn, ocl::typeToStr(CV_32SC(cn)), cnstr.c_str(),
            ocl::convertTypeStr(CV_8U, CV_32S, cn, cvt[0]),
            ocl::typeToStr(type), ocl::typeToStr(CV_32FC(cn)),
            ocl::convertTypeStr(CV_32S, CV_32F, cn, cvt[1]),
            ocl::convertTypeStr(CV_32F, CV_8U, cn, cvt[2]), gauss_color_coeff));
    if (k.empty())
        return false;

    Mat mspace_weight(1, d * d, CV_32FC1, space_weight);
    Mat mspace_ofs(1, d * d, CV_32SC1, space_ofs);
    UMat ucolor_weight, uspace_weight, uspace_ofs;

    mspace_weight.copyTo(uspace_weight);
    mspace_ofs.copyTo(uspace_ofs);

    k.args(ocl::KernelArg::ReadOnlyNoSize(temp), ocl::KernelArg::WriteOnly(dst),
           ocl::KernelArg::PtrReadOnly(uspace_weight),
           ocl::KernelArg::PtrReadOnly(uspace_ofs));

    size_t globalsize[2] = { (size_t)dst.cols / sizeDiv, (size_t)dst.rows };
    return k.run(2, globalsize, NULL, false);
}
#endif


static void
bilateralFilter_8u( const Mat& src, Mat& dst, int d,
    double sigma_color, double sigma_space,
    int borderType )
{
    CV_INSTRUMENT_REGION();

    int cn = src.channels();
    int i, j, maxk, radius;

    CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.data != dst.data );

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;

    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

    std::vector<float> _color_weight(cn*256);
    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* color_weight = &_color_weight[0];
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // initialize color-related bilateral filter coefficients

    for( i = 0; i < 256*cn; i++ )
        color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
    {
        j = -radius;

        for( ; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*temp.step + j*cn);
        }
    }

    CV_CPU_DISPATCH(bilateralFilterInvoker_8u, (dst, temp, radius, maxk, space_ofs, space_weight, color_weight),
        CV_CPU_DISPATCH_MODES_ALL);
}


static void
bilateralFilter_32f( const Mat& src, Mat& dst, int d,
                     double sigma_color, double sigma_space,
                     int borderType )
{
    CV_INSTRUMENT_REGION();

    int cn = src.channels();
    int i, j, maxk, radius;
    double minValSrc=-1, maxValSrc=1;
    const int kExpNumBinsPerChannel = 1 << 12;
    int kExpNumBins = 0;
    float lastExpVal = 1.f;
    float len, scale_index;

    CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.data != dst.data );

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;
    // compute the min/max range for the input image (even if multichannel)

    minMaxLoc( src.reshape(1), &minValSrc, &maxValSrc );
    if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
    {
        src.copyTo(dst);
        return;
    }

    // temporary copy of the image with borders for easy processing
    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

    // allocate lookup tables
    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // assign a length which is slightly more than needed
    len = (float)(maxValSrc - minValSrc) * cn;
    kExpNumBins = kExpNumBinsPerChannel * cn;
    std::vector<float> _expLUT(kExpNumBins+2);
    float* expLUT = &_expLUT[0];

    scale_index = kExpNumBins/len;

    // initialize the exp LUT
    for( i = 0; i < kExpNumBins+2; i++ )
    {
        if( lastExpVal > 0.f )
        {
            double val =  i / scale_index;
            expLUT[i] = (float)std::exp(val * val * gauss_color_coeff);
            lastExpVal = expLUT[i];
        }
        else
            expLUT[i] = 0.f;
    }

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if( r > radius || ( i == 0 && j == 0 ) )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*(temp.step/sizeof(float)) + j*cn);
        }

    // parallel_for usage
    CV_CPU_DISPATCH(bilateralFilterInvoker_32f, (cn, radius, maxk, space_ofs, temp, dst, scale_index, space_weight, expLUT),
        CV_CPU_DISPATCH_MODES_ALL);
}

#ifdef HAVE_IPP
#define IPP_BILATERAL_PARALLEL 1

#ifdef HAVE_IPP_IW
class ipp_bilateralFilterParallel: public ParallelLoopBody
{
public:
    ipp_bilateralFilterParallel(::ipp::IwiImage &_src, ::ipp::IwiImage &_dst, int _radius, Ipp32f _valSquareSigma, Ipp32f _posSquareSigma, ::ipp::IwiBorderType _borderType, bool *_ok):
        src(_src), dst(_dst)
    {
        pOk = _ok;

        radius          = _radius;
        valSquareSigma  = _valSquareSigma;
        posSquareSigma  = _posSquareSigma;
        borderType      = _borderType;

        *pOk = true;
    }
    ~ipp_bilateralFilterParallel() {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        if(*pOk == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, src, dst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &src;
    ::ipp::IwiImage &dst;

    int                  radius;
    Ipp32f               valSquareSigma;
    Ipp32f               posSquareSigma;
    ::ipp::IwiBorderType borderType;

    bool  *pOk;
    const ipp_bilateralFilterParallel& operator= (const ipp_bilateralFilterParallel&);
};
#endif

static bool ipp_bilateralFilter(Mat &src, Mat &dst, int d, double sigmaColor, double sigmaSpace, int borderType)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    int         radius         = IPP_MAX(((d <= 0)?cvRound(sigmaSpace*1.5):d/2), 1);
    Ipp32f      valSquareSigma = (Ipp32f)((sigmaColor <= 0)?1:sigmaColor*sigmaColor);
    Ipp32f      posSquareSigma = (Ipp32f)((sigmaSpace <= 0)?1:sigmaSpace*sigmaSpace);

    // Acquire data and begin processing
    try
    {
        ::ipp::IwiImage      iwSrc = ippiGetImage(src);
        ::ipp::IwiImage      iwDst = ippiGetImage(dst);
        ::ipp::IwiBorderSize borderSize(radius);
        ::ipp::IwiBorderType ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        const int threads = ippiSuggestThreadsNum(iwDst, 2);
        if(IPP_BILATERAL_PARALLEL && threads > 1) {
            bool  ok      = true;
            Range range(0, (int)iwDst.m_size.height);
            ipp_bilateralFilterParallel invoker(iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ippBorder, &ok);
            if(!ok)
                return false;

            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return false;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return false;
    }
    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(d); CV_UNUSED(sigmaColor); CV_UNUSED(sigmaSpace); CV_UNUSED(borderType);
    return false;
#endif
}
#endif

void bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_src.empty());

    _dst.create( _src.size(), _src.type() );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_bilateralFilter_8u(_src, _dst, d, sigmaColor, sigmaSpace, borderType))

    Mat src = _src.getMat(), dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType));

    if( src.depth() == CV_8U )
        bilateralFilter_8u( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else if( src.depth() == CV_32F )
        bilateralFilter_32f( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else
        CV_Error( CV_StsUnsupportedFormat,
        "Bilateral filtering is only implemented for 8u and 32f images" );
}

} // namespace
