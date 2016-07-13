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

#include "bm3d_denoising_invoker.hpp"
#include "bm3d_denoising_transforms.hpp"

template<typename ST, typename IT, typename UIT, typename D>
static void bm3dDenoising_(
    const Mat& src,
    Mat& dst,
    const float& h,
    const int &templateWindowSize,
    const int &searchWindowSize,
    const int &hBM,
    const int &groupSize)
{
    double granularity = (double)std::max(1., (double)dst.total() / (1 << 16));

    switch (CV_MAT_CN(src.type())) {
    case 1:
        parallel_for_(cv::Range(0, src.rows),
            Bm3dDenoisingInvoker<ST, IT, UIT, D, float, short>(
                src, dst, templateWindowSize, searchWindowSize, h, hBM, groupSize),
            granularity);
        break;
    default:
        CV_Error(Error::StsBadArg,
            "Unsupported number of channels! Only 1 channel is supported at the moment.");
    }
}

void cv::bm3dDenoising(
    InputArray _src,
    OutputArray _dst,
    float h,
    int templateWindowSize,
    int searchWindowSize,
    int blockMatchingThreshold,
    int groupSize,
    int normType,
    int transformType)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(1 == cn);
    CV_Assert(0 == transformType);

    Size srcSize = _src.size();
    Mat src = _src.getMat();
    _dst.create(srcSize, src.type());
    Mat dst = _dst.getMat();

    switch (normType) {
    case cv::NORM_L2:
        switch (depth) {
        case CV_8U:
            bm3dDenoising_<uchar, int, unsigned, DistSquared>(
                src,
                dst,
                h,
                templateWindowSize,
                searchWindowSize,
                blockMatchingThreshold,
                groupSize);
            break;
        default:
            CV_Error(Error::StsBadArg,
                "Unsupported depth! Only CV_8U is supported for NORM_L2");
        }
        break;
    case cv::NORM_L1:
        switch (depth) {
        case CV_8U:
            bm3dDenoising_<uchar, int, unsigned, DistAbs>(
                src,
                dst,
                h,
                templateWindowSize,
                searchWindowSize,
                blockMatchingThreshold,
                groupSize);
            break;
        case CV_16U:
            bm3dDenoising_<ushort, int64, uint64, DistAbs>(
                src,
                dst,
                h,
                templateWindowSize,
                searchWindowSize,
                blockMatchingThreshold,
                groupSize);
            break;
        default:
            CV_Error(Error::StsBadArg,
                "Unsupported depth! Only CV_8U and CV_16U are supported for NORM_L1");
        }
        break;
    default:
        CV_Error(Error::StsBadArg,
            "Unsupported norm type! Only NORM_L2 and NORM_L1 are supported");
    }
}
