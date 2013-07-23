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

#if !defined (HAVE_CUDA) || !defined (HAVE_OPENCV_CUDALEGACY) || defined (CUDA_DISABLER)

void cv::cuda::BroxOpticalFlow::operator ()(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

#else

namespace
{
    size_t getBufSize(const NCVBroxOpticalFlowDescriptor& desc, const NCVMatrix<Ncv32f>& frame0, const NCVMatrix<Ncv32f>& frame1,
                      NCVMatrix<Ncv32f>& u, NCVMatrix<Ncv32f>& v, const cudaDeviceProp& devProp)
    {
        NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));

        ncvSafeCall( NCVBroxOpticalFlow(desc, gpuCounter, frame0, frame1, u, v, 0) );

        return gpuCounter.maxSize();
    }
}

namespace
{
    static void outputHandler(const String &msg) { CV_Error(cv::Error::GpuApiCallError, msg.c_str()); }
}

void cv::cuda::BroxOpticalFlow::operator ()(const GpuMat& frame0, const GpuMat& frame1, GpuMat& u, GpuMat& v, Stream& s)
{
    ncvSetDebugOutputHandler(outputHandler);

    CV_Assert(frame0.type() == CV_32FC1);
    CV_Assert(frame1.size() == frame0.size() && frame1.type() == frame0.type());

    u.create(frame0.size(), CV_32FC1);
    v.create(frame0.size(), CV_32FC1);

    cudaDeviceProp devProp;
    cudaSafeCall( cudaGetDeviceProperties(&devProp, getDevice()) );

    NCVBroxOpticalFlowDescriptor desc;

    desc.alpha = alpha;
    desc.gamma = gamma;
    desc.scale_factor = scale_factor;
    desc.number_of_inner_iterations = inner_iterations;
    desc.number_of_outer_iterations = outer_iterations;
    desc.number_of_solver_iterations = solver_iterations;

    NCVMemSegment frame0MemSeg;
    frame0MemSeg.begin.memtype = NCVMemoryTypeDevice;
    frame0MemSeg.begin.ptr = const_cast<uchar*>(frame0.data);
    frame0MemSeg.size = frame0.step * frame0.rows;

    NCVMemSegment frame1MemSeg;
    frame1MemSeg.begin.memtype = NCVMemoryTypeDevice;
    frame1MemSeg.begin.ptr = const_cast<uchar*>(frame1.data);
    frame1MemSeg.size = frame1.step * frame1.rows;

    NCVMemSegment uMemSeg;
    uMemSeg.begin.memtype = NCVMemoryTypeDevice;
    uMemSeg.begin.ptr = u.ptr();
    uMemSeg.size = u.step * u.rows;

    NCVMemSegment vMemSeg;
    vMemSeg.begin.memtype = NCVMemoryTypeDevice;
    vMemSeg.begin.ptr = v.ptr();
    vMemSeg.size = v.step * v.rows;

    NCVMatrixReuse<Ncv32f> frame0Mat(frame0MemSeg, static_cast<Ncv32u>(devProp.textureAlignment), frame0.cols, frame0.rows, static_cast<Ncv32u>(frame0.step));
    NCVMatrixReuse<Ncv32f> frame1Mat(frame1MemSeg, static_cast<Ncv32u>(devProp.textureAlignment), frame1.cols, frame1.rows, static_cast<Ncv32u>(frame1.step));
    NCVMatrixReuse<Ncv32f> uMat(uMemSeg, static_cast<Ncv32u>(devProp.textureAlignment), u.cols, u.rows, static_cast<Ncv32u>(u.step));
    NCVMatrixReuse<Ncv32f> vMat(vMemSeg, static_cast<Ncv32u>(devProp.textureAlignment), v.cols, v.rows, static_cast<Ncv32u>(v.step));

    cudaStream_t stream = StreamAccessor::getStream(s);

    size_t bufSize = getBufSize(desc, frame0Mat, frame1Mat, uMat, vMat, devProp);

    ensureSizeIsEnough(1, static_cast<int>(bufSize), CV_8UC1, buf);

    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(devProp.textureAlignment), buf.ptr());

    ncvSafeCall( NCVBroxOpticalFlow(desc, gpuAllocator, frame0Mat, frame1Mat, uMat, vMat, stream) );
}

#endif /* HAVE_CUDA */
