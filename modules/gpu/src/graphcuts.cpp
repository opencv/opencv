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

#if !defined (HAVE_CUDA)

void cv::gpu::graphcut(GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

void cv::gpu::graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels, GpuMat& buf)
{
    CV_Assert(leftTransp.type() == CV_32S && rightTransp.type() == CV_32S);
    CV_Assert(terminals.type() == CV_32S && bottom.type() == CV_32S && top.type() == CV_32S);
    CV_Assert(terminals.size() == leftTransp.size());
    CV_Assert(terminals.size() == rightTransp.size());
    CV_Assert(terminals.size() == top.size() && terminals.size() == bottom.size());
    CV_Assert(top.step == bottom.step && top.step == terminals.step && rightTransp.step == leftTransp.step);

    labels.create(terminals.size(), CV_8U);

    NppiSize sznpp;
    sznpp.width = terminals.cols;
    sznpp.height = terminals.rows;

    int bufsz;
    nppSafeCall( nppiGraphcutGetSize(sznpp, &bufsz) );

    if ((size_t)bufsz > buf.cols * buf.rows * buf.elemSize())
        buf.create(1, bufsz, CV_8U);

    nppSafeCall( nppiGraphcut_32s8u(terminals.ptr<Npp32s>(), leftTransp.ptr<Npp32s>(), rightTransp.ptr<Npp32s>(), top.ptr<Npp32s>(), bottom.ptr<Npp32s>(),
        terminals.step, leftTransp.step, sznpp, labels.ptr<Npp8u>(), labels.step, buf.ptr<Npp8u>()) );

    cudaSafeCall( cudaThreadSynchronize() );
}


#endif /* !defined (HAVE_CUDA) */






