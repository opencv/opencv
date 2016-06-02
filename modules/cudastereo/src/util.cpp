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

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::cuda::reprojectImageTo3D(InputArray, OutputArray, InputArray, int, Stream&) { throw_no_cuda(); }
void cv::cuda::drawColorDisp(InputArray, OutputArray, int, Stream&) { throw_no_cuda(); }

#else

////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

namespace cv { namespace cuda { namespace device
{
    template <typename T, typename D>
    void reprojectImageTo3D_gpu(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
}}}

void cv::cuda::reprojectImageTo3D(InputArray _disp, OutputArray _xyz, InputArray _Q, int dst_cn, Stream& stream)
{
    using namespace cv::cuda::device;

    typedef void (*func_t)(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    static const func_t funcs[2][6] =
    {
        {reprojectImageTo3D_gpu<uchar, float3>, 0, 0, reprojectImageTo3D_gpu<short, float3>, reprojectImageTo3D_gpu<int, float3>, reprojectImageTo3D_gpu<float, float3>},
        {reprojectImageTo3D_gpu<uchar, float4>, 0, 0, reprojectImageTo3D_gpu<short, float4>, reprojectImageTo3D_gpu<int, float4>, reprojectImageTo3D_gpu<float, float4>}
    };

    GpuMat disp = _disp.getGpuMat();
    Mat Q = _Q.getMat();

    CV_Assert( disp.type() == CV_8U || disp.type() == CV_16S || disp.type() == CV_32S || disp.type() == CV_32F );
    CV_Assert( Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4 && Q.isContinuous() );
    CV_Assert( dst_cn == 3 || dst_cn == 4 );

    _xyz.create(disp.size(), CV_MAKE_TYPE(CV_32F, dst_cn));
    GpuMat xyz = _xyz.getGpuMat();

    funcs[dst_cn == 4][disp.type()](disp, xyz, Q.ptr<float>(), StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// drawColorDisp

namespace cv { namespace cuda { namespace device
{
    void drawColorDisp_gpu(const PtrStepSzb& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream);
    void drawColorDisp_gpu(const PtrStepSz<short>& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream);
    void drawColorDisp_gpu(const PtrStepSz<int>& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream);
    void drawColorDisp_gpu(const PtrStepSz<float>& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream);
}}}

namespace
{
    template <typename T>
    void drawColorDisp_caller(const GpuMat& src, OutputArray _dst, int ndisp, const cudaStream_t& stream)
    {
        using namespace ::cv::cuda::device;

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        drawColorDisp_gpu((PtrStepSz<T>)src, dst, ndisp, stream);
    }
}

void cv::cuda::drawColorDisp(InputArray _src, OutputArray dst, int ndisp, Stream& stream)
{
    typedef void (*drawColorDisp_caller_t)(const GpuMat& src, OutputArray dst, int ndisp, const cudaStream_t& stream);
    const drawColorDisp_caller_t drawColorDisp_callers[] = {drawColorDisp_caller<unsigned char>, 0, 0, drawColorDisp_caller<short>, drawColorDisp_caller<int>, drawColorDisp_caller<float>, 0, 0};

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8U || src.type() == CV_16S || src.type() == CV_32S || src.type() == CV_32F );

    drawColorDisp_callers[src.type()](src, dst, ndisp, StreamAccessor::getStream(stream));
}

#endif
