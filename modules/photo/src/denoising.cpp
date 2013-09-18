/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "fast_nlmeans_denoising_invoker.hpp"
#include "fast_nlmeans_multi_denoising_invoker.hpp"

void cv::fastNlMeansDenoising( InputArray _src, OutputArray _dst, float h,
                               int templateWindowSize, int searchWindowSize)
{
    Mat src = _src.getMat();
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
    if(tegra::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize))
        return;
#endif

    switch (src.type()) {
        case CV_8U:
            parallel_for_(cv::Range(0, src.rows),
                FastNlMeansDenoisingInvoker<uchar>(
                    src, dst, templateWindowSize, searchWindowSize, h));
            break;
        case CV_8UC2:
            parallel_for_(cv::Range(0, src.rows),
                FastNlMeansDenoisingInvoker<cv::Vec2b>(
                    src, dst, templateWindowSize, searchWindowSize, h));
            break;
        case CV_8UC3:
            parallel_for_(cv::Range(0, src.rows),
                FastNlMeansDenoisingInvoker<cv::Vec3b>(
                    src, dst, templateWindowSize, searchWindowSize, h));
            break;
        default:
            CV_Error(Error::StsBadArg,
                "Unsupported image format! Only CV_8UC1, CV_8UC2 and CV_8UC3 are supported");
    }
}

void cv::fastNlMeansDenoisingColored( InputArray _src, OutputArray _dst,
                                      float h, float hForColorComponents,
                                      int templateWindowSize, int searchWindowSize)
{
    Mat src = _src.getMat();
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();

    if (src.type() != CV_8UC3) {
        CV_Error(Error::StsBadArg, "Type of input image should be CV_8UC3!");
        return;
    }

    Mat src_lab;
    cvtColor(src, src_lab, COLOR_LBGR2Lab);

    Mat l(src.size(), CV_8U);
    Mat ab(src.size(), CV_8UC2);
    Mat l_ab[] = { l, ab };
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels(&src_lab, 1, l_ab, 2, from_to, 3);

    fastNlMeansDenoising(l, l, h, templateWindowSize, searchWindowSize);
    fastNlMeansDenoising(ab, ab, hForColorComponents, templateWindowSize, searchWindowSize);

    Mat l_ab_denoised[] = { l, ab };
    Mat dst_lab(src.size(), src.type());
    mixChannels(l_ab_denoised, 2, &dst_lab, 1, from_to, 3);

    cvtColor(dst_lab, dst, COLOR_Lab2LBGR);
}

static void fastNlMeansDenoisingMultiCheckPreconditions(
                               const std::vector<Mat>& srcImgs,
                               int imgToDenoiseIndex, int temporalWindowSize,
                               int templateWindowSize, int searchWindowSize)
{
    int src_imgs_size = static_cast<int>(srcImgs.size());
    if (src_imgs_size == 0) {
        CV_Error(Error::StsBadArg, "Input images vector should not be empty!");
    }

    if (temporalWindowSize % 2 == 0 ||
        searchWindowSize % 2 == 0 ||
        templateWindowSize % 2 == 0) {
        CV_Error(Error::StsBadArg, "All windows sizes should be odd!");
    }

    int temporalWindowHalfSize = temporalWindowSize / 2;
    if (imgToDenoiseIndex - temporalWindowHalfSize < 0 ||
        imgToDenoiseIndex + temporalWindowHalfSize >= src_imgs_size)
    {
        CV_Error(Error::StsBadArg,
            "imgToDenoiseIndex and temporalWindowSize "
            "should be choosen corresponding srcImgs size!");
    }

    for (int i = 1; i < src_imgs_size; i++) {
        if (srcImgs[0].size() != srcImgs[i].size() || srcImgs[0].type() != srcImgs[i].type()) {
            CV_Error(Error::StsBadArg, "Input images should have the same size and type!");
        }
    }
}

void cv::fastNlMeansDenoisingMulti( InputArrayOfArrays _srcImgs, OutputArray _dst,
                                    int imgToDenoiseIndex, int temporalWindowSize,
                                    float h, int templateWindowSize, int searchWindowSize)
{
    std::vector<Mat> srcImgs;
    _srcImgs.getMatVector(srcImgs);

    fastNlMeansDenoisingMultiCheckPreconditions(
        srcImgs, imgToDenoiseIndex,
        temporalWindowSize, templateWindowSize, searchWindowSize
    );
    _dst.create(srcImgs[0].size(), srcImgs[0].type());
    Mat dst = _dst.getMat();

    switch (srcImgs[0].type()) {
        case CV_8U:
            parallel_for_(cv::Range(0, srcImgs[0].rows),
                FastNlMeansMultiDenoisingInvoker<uchar>(
                    srcImgs, imgToDenoiseIndex, temporalWindowSize,
                    dst, templateWindowSize, searchWindowSize, h));
            break;
        case CV_8UC2:
            parallel_for_(cv::Range(0, srcImgs[0].rows),
                FastNlMeansMultiDenoisingInvoker<cv::Vec2b>(
                    srcImgs, imgToDenoiseIndex, temporalWindowSize,
                    dst, templateWindowSize, searchWindowSize, h));
            break;
        case CV_8UC3:
            parallel_for_(cv::Range(0, srcImgs[0].rows),
                FastNlMeansMultiDenoisingInvoker<cv::Vec3b>(
                    srcImgs, imgToDenoiseIndex, temporalWindowSize,
                    dst, templateWindowSize, searchWindowSize, h));
            break;
        default:
            CV_Error(Error::StsBadArg,
                "Unsupported matrix format! Only uchar, Vec2b, Vec3b are supported");
    }
}

void cv::fastNlMeansDenoisingColoredMulti( InputArrayOfArrays _srcImgs, OutputArray _dst,
                                           int imgToDenoiseIndex, int temporalWindowSize,
                                           float h, float hForColorComponents,
                                           int templateWindowSize, int searchWindowSize)
{
    std::vector<Mat> srcImgs;
    _srcImgs.getMatVector(srcImgs);

    fastNlMeansDenoisingMultiCheckPreconditions(
        srcImgs, imgToDenoiseIndex,
        temporalWindowSize, templateWindowSize, searchWindowSize
    );

    _dst.create(srcImgs[0].size(), srcImgs[0].type());
    Mat dst = _dst.getMat();

    int src_imgs_size = static_cast<int>(srcImgs.size());

    if (srcImgs[0].type() != CV_8UC3) {
        CV_Error(Error::StsBadArg, "Type of input images should be CV_8UC3!");
        return;
    }

    int from_to[] = { 0,0, 1,1, 2,2 };

    // TODO convert only required images
    std::vector<Mat> src_lab(src_imgs_size);
    std::vector<Mat> l(src_imgs_size);
    std::vector<Mat> ab(src_imgs_size);
    for (int i = 0; i < src_imgs_size; i++) {
        src_lab[i] = Mat::zeros(srcImgs[0].size(), CV_8UC3);
        l[i] = Mat::zeros(srcImgs[0].size(), CV_8UC1);
        ab[i] = Mat::zeros(srcImgs[0].size(), CV_8UC2);
        cvtColor(srcImgs[i], src_lab[i], COLOR_LBGR2Lab);

        Mat l_ab[] = { l[i], ab[i] };
        mixChannels(&src_lab[i], 1, l_ab, 2, from_to, 3);
    }

    Mat dst_l;
    Mat dst_ab;

    fastNlMeansDenoisingMulti(
        l, dst_l, imgToDenoiseIndex, temporalWindowSize,
        h, templateWindowSize, searchWindowSize);

    fastNlMeansDenoisingMulti(
        ab, dst_ab, imgToDenoiseIndex, temporalWindowSize,
        hForColorComponents, templateWindowSize, searchWindowSize);

    Mat l_ab_denoised[] = { dst_l, dst_ab };
    Mat dst_lab(srcImgs[0].size(), srcImgs[0].type());
    mixChannels(l_ab_denoised, 2, &dst_lab, 1, from_to, 3);

    cvtColor(dst_lab, dst, COLOR_Lab2LBGR);
}
