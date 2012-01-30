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

void cv::gpu::graphcut(GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

void cv::gpu::graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels, GpuMat& buf, Stream& s)
{
    Size src_size = terminals.size();
    CV_Assert(terminals.type() == CV_32S);
    CV_Assert(leftTransp.size() == Size(src_size.height, src_size.width));
    CV_Assert(leftTransp.type() == CV_32S);
    CV_Assert(rightTransp.size() == Size(src_size.height, src_size.width));
    CV_Assert(rightTransp.type() == CV_32S);
    CV_Assert(top.size() == src_size);
    CV_Assert(top.type() == CV_32S);
    CV_Assert(bottom.size() == src_size);
    CV_Assert(bottom.type() == CV_32S);

    labels.create(src_size, CV_8U);

    NppiSize sznpp;
    sznpp.width = src_size.width;
    sznpp.height = src_size.height;

    int bufsz;
    nppSafeCall( nppiGraphcutGetSize(sznpp, &bufsz) );

    if ((size_t)bufsz > buf.cols * buf.rows * buf.elemSize())
        buf.create(1, bufsz, CV_8U);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

#if CUDART_VERSION > 4000 
    NppiGraphcutState* pState;
    nppSafeCall( nppiGraphcutInitAlloc(sznpp, &pState, buf.ptr<Npp8u>()) );
    
    nppSafeCall( nppiGraphcut_32s8u(terminals.ptr<Npp32s>(), leftTransp.ptr<Npp32s>(), rightTransp.ptr<Npp32s>(), top.ptr<Npp32s>(), bottom.ptr<Npp32s>(),
        static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), pState) );

    nppSafeCall( nppiGraphcutFree(pState) );
#else
    nppSafeCall( nppiGraphcut_32s8u(terminals.ptr<Npp32s>(), leftTransp.ptr<Npp32s>(), rightTransp.ptr<Npp32s>(), top.ptr<Npp32s>(), bottom.ptr<Npp32s>(),
        static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), buf.ptr<Npp8u>()) );
#endif

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}


#endif /* !defined (HAVE_CUDA) */

