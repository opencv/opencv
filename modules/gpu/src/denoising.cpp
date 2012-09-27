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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::bilateralFilter(const GpuMat&, GpuMat&, int, float, float, int, Stream&) { throw_nogpu(); }
void cv::gpu::nonLocalMeans(const GpuMat&, GpuMat&, float, int, int, int, Stream&) { throw_nogpu(); }
void cv::gpu::fastNlMeansDenoising( const GpuMat&, GpuMat&, float, int, int, Stream&) { throw_nogpu(); }

#else

//////////////////////////////////////////////////////////////////////////////////
//// Non Local Means Denosing (brute force)

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template<typename T>
        void bilateral_filter_gpu(const PtrStepSzb& src, PtrStepSzb dst, int kernel_size, float sigma_spatial, float sigma_color, int borderMode, cudaStream_t stream);

        template<typename T>
        void nlm_bruteforce_gpu(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream);
    }
}}}

void cv::gpu::bilateralFilter(const GpuMat& src, GpuMat& dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode, Stream& s)
{
    using cv::gpu::device::imgproc::bilateral_filter_gpu;

    typedef void (*func_t)(const PtrStepSzb& src, PtrStepSzb dst, int kernel_size, float sigma_spatial, float sigma_color, int borderMode, cudaStream_t s);

    static const func_t funcs[6][4] =
    {
        {bilateral_filter_gpu<uchar>      , 0 /*bilateral_filter_gpu<uchar2>*/ , bilateral_filter_gpu<uchar3>      , bilateral_filter_gpu<uchar4>      },
        {0 /*bilateral_filter_gpu<schar>*/, 0 /*bilateral_filter_gpu<schar2>*/ , 0 /*bilateral_filter_gpu<schar3>*/, 0 /*bilateral_filter_gpu<schar4>*/},
        {bilateral_filter_gpu<ushort>     , 0 /*bilateral_filter_gpu<ushort2>*/, bilateral_filter_gpu<ushort3>     , bilateral_filter_gpu<ushort4>     },
        {bilateral_filter_gpu<short>      , 0 /*bilateral_filter_gpu<short2>*/ , bilateral_filter_gpu<short3>      , bilateral_filter_gpu<short4>      },
        {0 /*bilateral_filter_gpu<int>*/  , 0 /*bilateral_filter_gpu<int2>*/   , 0 /*bilateral_filter_gpu<int3>*/  , 0 /*bilateral_filter_gpu<int4>*/  },
        {bilateral_filter_gpu<float>      , 0 /*bilateral_filter_gpu<float2>*/ , bilateral_filter_gpu<float3>      , bilateral_filter_gpu<float4>      }
    };

    sigma_color = (sigma_color <= 0 ) ? 1 : sigma_color;
    sigma_spatial = (sigma_spatial <= 0 ) ? 1 : sigma_spatial;


    int radius = (kernel_size <= 0) ? cvRound(sigma_spatial*1.5) : kernel_size/2;
    kernel_size = std::max(radius, 1)*2 + 1;

    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);
    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert(func != 0);

    CV_Assert(borderMode == BORDER_REFLECT101 || borderMode == BORDER_REPLICATE || borderMode == BORDER_CONSTANT || borderMode == BORDER_REFLECT || borderMode == BORDER_WRAP);

    int gpuBorderType;
    CV_Assert(tryConvertToGpuBorderType(borderMode, gpuBorderType));

    dst.create(src.size(), src.type());
    func(src, dst, kernel_size, sigma_spatial, sigma_color, gpuBorderType, StreamAccessor::getStream(s));
}

void cv::gpu::nonLocalMeans(const GpuMat& src, GpuMat& dst, float h, int search_window_size, int block_size, int borderMode, Stream& s)
{
    using cv::gpu::device::imgproc::nlm_bruteforce_gpu;
    typedef void (*func_t)(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream);

    static const func_t funcs[4] = { nlm_bruteforce_gpu<uchar>, nlm_bruteforce_gpu<uchar2>, nlm_bruteforce_gpu<uchar3>, 0/*nlm_bruteforce_gpu<uchar4>,*/ };

    CV_Assert(src.type() == CV_8U || src.type() == CV_8UC2 || src.type() == CV_8UC3);

    const func_t func = funcs[src.channels() - 1];
    CV_Assert(func != 0);

    int b = borderMode;
    CV_Assert(b == BORDER_REFLECT101 || b == BORDER_REPLICATE || b == BORDER_CONSTANT || b == BORDER_REFLECT || b == BORDER_WRAP);

    int gpuBorderType;
    CV_Assert(tryConvertToGpuBorderType(borderMode, gpuBorderType));

    int search_radius = search_window_size/2;
    int block_radius = block_size/2;

    dst.create(src.size(), src.type());
    func(src, dst, search_radius, block_radius, h, gpuBorderType, StreamAccessor::getStream(s));
}


//////////////////////////////////////////////////////////////////////////////////
//// Non Local Means Denosing (fast approxinate)


namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void nln_fast_get_buffer_size(const PtrStepSzb& src, int search_window, int block_window, int& buffer_cols, int& buffer_rows);

        template<typename T>
        void nlm_fast_gpu(const PtrStepSzb& src, PtrStepSzb dst, PtrStepi buffer,
                          int search_window, int block_window, float h, cudaStream_t stream);

    }
}}}



//class CV_EXPORTS FastNonLocalMeansDenoising
//{
//public:
//    FastNonLocalMeansDenoising(float h, int search_radius, int block_radius, const Size& image_size = Size())
//    {
//        if (size.area() != 0)
//            allocate_buffers(image_size);
//    }

//    void operator()(const GpuMat& src, GpuMat& dst);

//private:
//    void allocate_buffers(const Size& image_size)
//    {
//        col_dist_sums.create(block_window_, search_window_ * search_window_, CV_32S);
//        up_col_dist_sums.create(image_size.width, search_window_ * search_window_, CV_32S);
//    }

//    int search_radius_;
//    int block_radius;
//    GpuMat col_dist_sums_;
//    GpuMat up_col_dist_sums_;
//};

void cv::gpu::fastNlMeansDenoising( const GpuMat& src, GpuMat& dst, float h, int search_radius, int block_radius, Stream& s)
{
    dst.create(src.size(), src.type());
    CV_Assert(src.depth() == CV_8U && src.channels() < 4);

    GpuMat extended_src, src_hdr;
    int border_size = search_radius + block_radius;
    cv::gpu::copyMakeBorder(src, extended_src, border_size, border_size, border_size, border_size, cv::BORDER_DEFAULT, Scalar(), s);
    src_hdr = extended_src(Rect(Point2i(border_size, border_size), src.size()));

    using namespace cv::gpu::device::imgproc;
    typedef void (*nlm_fast_t)(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float, cudaStream_t);
    static const nlm_fast_t funcs[] = { nlm_fast_gpu<uchar>, nlm_fast_gpu<uchar2>, nlm_fast_gpu<uchar3>, 0 };

    int search_window = 2 * search_radius + 1;
    int block_window = 2 * block_radius + 1;
        
    int bcols, brows;
    nln_fast_get_buffer_size(src_hdr, search_window, block_window, bcols, brows);

    //GpuMat col_dist_sums(block_window * gx, search_window * search_window * gy, CV_32S);
    //GpuMat up_col_dist_sums(src.cols, search_window * search_window * gy, CV_32S);
    GpuMat buffer(brows, bcols, CV_32S);

    funcs[src.channels()-1](src_hdr, dst, buffer, search_window, block_window, h, StreamAccessor::getStream(s));
}

//void cv::gpu::fastNlMeansDenoisingColored( const GpuMat& src, GpuMat& dst, float h, float hForColorComponents, int templateWindowSize, int searchWindowSize)
//{
//    Mat src = _src.getMat();
//    _dst.create(src.size(), src.type());
//    Mat dst = _dst.getMat();

//    if (src.type() != CV_8UC3) {
//        CV_Error(CV_StsBadArg, "Type of input image should be CV_8UC3!");
//        return;
//    }

//    Mat src_lab;
//    cvtColor(src, src_lab, CV_LBGR2Lab);

//    Mat l(src.size(), CV_8U);
//    Mat ab(src.size(), CV_8UC2);
//    Mat l_ab[] = { l, ab };
//    int from_to[] = { 0,0, 1,1, 2,2 };
//    mixChannels(&src_lab, 1, l_ab, 2, from_to, 3);

//    fastNlMeansDenoising(l, l, h, templateWindowSize, searchWindowSize);
//    fastNlMeansDenoising(ab, ab, hForColorComponents, templateWindowSize, searchWindowSize);

//    Mat l_ab_denoised[] = { l, ab };
//    Mat dst_lab(src.size(), src.type());
//    mixChannels(l_ab_denoised, 2, &dst_lab, 1, from_to, 3);

//    cvtColor(dst_lab, dst, CV_Lab2LBGR);
//}

//static void fastNlMeansDenoisingMultiCheckPreconditions(
//                               const std::vector<Mat>& srcImgs,
//                               int imgToDenoiseIndex, int temporalWindowSize,
//                               int templateWindowSize, int searchWindowSize)
//{
//    int src_imgs_size = (int)srcImgs.size();
//    if (src_imgs_size == 0) {
//        CV_Error(CV_StsBadArg, "Input images vector should not be empty!");
//    }

//    if (temporalWindowSize % 2 == 0 ||
//        searchWindowSize % 2 == 0 ||
//        templateWindowSize % 2 == 0) {
//        CV_Error(CV_StsBadArg, "All windows sizes should be odd!");
//    }

//    int temporalWindowHalfSize = temporalWindowSize / 2;
//    if (imgToDenoiseIndex - temporalWindowHalfSize < 0 ||
//        imgToDenoiseIndex + temporalWindowHalfSize >= src_imgs_size)
//    {
//        CV_Error(CV_StsBadArg,
//            "imgToDenoiseIndex and temporalWindowSize "
//            "should be choosen corresponding srcImgs size!");
//    }

//    for (int i = 1; i < src_imgs_size; i++) {
//        if (srcImgs[0].size() != srcImgs[i].size() || srcImgs[0].type() != srcImgs[i].type()) {
//            CV_Error(CV_StsBadArg, "Input images should have the same size and type!");
//        }
//    }
//}

//void cv::fastNlMeansDenoisingMulti( InputArrayOfArrays _srcImgs, OutputArray _dst,
//                                    int imgToDenoiseIndex, int temporalWindowSize,
//                                    float h, int templateWindowSize, int searchWindowSize)
//{
//    vector<Mat> srcImgs;
//    _srcImgs.getMatVector(srcImgs);

//    fastNlMeansDenoisingMultiCheckPreconditions(
//        srcImgs, imgToDenoiseIndex,
//        temporalWindowSize, templateWindowSize, searchWindowSize
//    );
//    _dst.create(srcImgs[0].size(), srcImgs[0].type());
//    Mat dst = _dst.getMat();

//    switch (srcImgs[0].type()) {
//        case CV_8U:
//            parallel_for(cv::BlockedRange(0, srcImgs[0].rows),
//                FastNlMeansMultiDenoisingInvoker<uchar>(
//                    srcImgs, imgToDenoiseIndex, temporalWindowSize,
//                    dst, templateWindowSize, searchWindowSize, h));
//            break;
//        case CV_8UC2:
//            parallel_for(cv::BlockedRange(0, srcImgs[0].rows),
//                FastNlMeansMultiDenoisingInvoker<cv::Vec2b>(
//                    srcImgs, imgToDenoiseIndex, temporalWindowSize,
//                    dst, templateWindowSize, searchWindowSize, h));
//            break;
//        case CV_8UC3:
//            parallel_for(cv::BlockedRange(0, srcImgs[0].rows),
//                FastNlMeansMultiDenoisingInvoker<cv::Vec3b>(
//                    srcImgs, imgToDenoiseIndex, temporalWindowSize,
//                    dst, templateWindowSize, searchWindowSize, h));
//            break;
//        default:
//            CV_Error(CV_StsBadArg,
//                "Unsupported matrix format! Only uchar, Vec2b, Vec3b are supported");
//    }
//}

//void cv::fastNlMeansDenoisingColoredMulti( InputArrayOfArrays _srcImgs, OutputArray _dst,
//                                           int imgToDenoiseIndex, int temporalWindowSize,
//                                           float h, float hForColorComponents,
//                                           int templateWindowSize, int searchWindowSize)
//{
//    vector<Mat> srcImgs;
//    _srcImgs.getMatVector(srcImgs);

//    fastNlMeansDenoisingMultiCheckPreconditions(
//        srcImgs, imgToDenoiseIndex,
//        temporalWindowSize, templateWindowSize, searchWindowSize
//    );

//    _dst.create(srcImgs[0].size(), srcImgs[0].type());
//    Mat dst = _dst.getMat();

//    int src_imgs_size = (int)srcImgs.size();

//    if (srcImgs[0].type() != CV_8UC3) {
//        CV_Error(CV_StsBadArg, "Type of input images should be CV_8UC3!");
//        return;
//    }

//    int from_to[] = { 0,0, 1,1, 2,2 };

//    // TODO convert only required images
//    vector<Mat> src_lab(src_imgs_size);
//    vector<Mat> l(src_imgs_size);
//    vector<Mat> ab(src_imgs_size);
//    for (int i = 0; i < src_imgs_size; i++) {
//        src_lab[i] = Mat::zeros(srcImgs[0].size(), CV_8UC3);
//        l[i] = Mat::zeros(srcImgs[0].size(), CV_8UC1);
//        ab[i] = Mat::zeros(srcImgs[0].size(), CV_8UC2);
//        cvtColor(srcImgs[i], src_lab[i], CV_LBGR2Lab);

//        Mat l_ab[] = { l[i], ab[i] };
//        mixChannels(&src_lab[i], 1, l_ab, 2, from_to, 3);
//    }

//    Mat dst_l;
//    Mat dst_ab;

//    fastNlMeansDenoisingMulti(
//        l, dst_l, imgToDenoiseIndex, temporalWindowSize,
//        h, templateWindowSize, searchWindowSize);

//    fastNlMeansDenoisingMulti(
//        ab, dst_ab, imgToDenoiseIndex, temporalWindowSize,
//        hForColorComponents, templateWindowSize, searchWindowSize);

//    Mat l_ab_denoised[] = { dst_l, dst_ab };
//    Mat dst_lab(srcImgs[0].size(), srcImgs[0].type());
//    mixChannels(l_ab_denoised, 2, &dst_lab, 1, from_to, 3);

//    cvtColor(dst_lab, dst, CV_Lab2LBGR);
//}


#endif


