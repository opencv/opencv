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
using namespace cv::cuda;

#if !defined HAVE_CUDA || !defined(HAVE_OPENCV_CUDAARITHM) || defined(CUDA_DISABLER)

void cv::cuda::FastOpticalFlowBM::operator ()(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, int, int, Stream&) { throw_no_cuda(); }

#else // HAVE_CUDA

namespace optflowbm_fast
{
    void get_buffer_size(int src_cols, int src_rows, int search_window, int block_window, int& buffer_cols, int& buffer_rows);

    template <typename T>
    void calc(PtrStepSzb I0, PtrStepSzb I1, PtrStepSzf velx, PtrStepSzf vely, PtrStepi buffer, int search_window, int block_window, cudaStream_t stream);
}

void cv::cuda::FastOpticalFlowBM::operator ()(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy, int search_window, int block_window, Stream& stream)
{
    CV_Assert( I0.type() == CV_8UC1 );
    CV_Assert( I1.size() == I0.size() && I1.type() == I0.type() );

    int border_size = search_window / 2 + block_window / 2;
    Size esize = I0.size() + Size(border_size, border_size) * 2;

    ensureSizeIsEnough(esize, I0.type(), extended_I0);
    ensureSizeIsEnough(esize, I0.type(), extended_I1);

    cuda::copyMakeBorder(I0, extended_I0, border_size, border_size, border_size, border_size, cv::BORDER_DEFAULT, Scalar(), stream);
    cuda::copyMakeBorder(I1, extended_I1, border_size, border_size, border_size, border_size, cv::BORDER_DEFAULT, Scalar(), stream);

    GpuMat I0_hdr = extended_I0(Rect(Point2i(border_size, border_size), I0.size()));
    GpuMat I1_hdr = extended_I1(Rect(Point2i(border_size, border_size), I0.size()));

    int bcols, brows;
    optflowbm_fast::get_buffer_size(I0.cols, I0.rows, search_window, block_window, bcols, brows);

    ensureSizeIsEnough(brows, bcols, CV_32SC1, buffer);

    flowx.create(I0.size(), CV_32FC1);
    flowy.create(I0.size(), CV_32FC1);

    optflowbm_fast::calc<uchar>(I0_hdr, I1_hdr, flowx, flowy, buffer, search_window, block_window, StreamAccessor::getStream(stream));
}

#endif // HAVE_CUDA
