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
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

cv::gpu::StereoBM_GPU::StereoBM_GPU() { throw_nogpu(); }
cv::gpu::StereoBM_GPU::StereoBM_GPU(int, int, int) { throw_nogpu(); }

bool cv::gpu::StereoBM_GPU::checkIfGpuCallReasonable() { throw_nogpu(); return false; }
void cv::gpu::StereoBM_GPU::operator() ( const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::StereoBM_GPU::operator() ( const GpuMat&, const GpuMat&, GpuMat&, const CudaStream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu
{
    namespace impl
    {
        //extern "C" void stereoBM_GPU(const DevMem2D& left, const DevMem2D& right, const DevMem2D& disp, int ndisp, int winsz, const DevMem2D_<uint>& minSSD_buf);
        extern "C" void stereoBM_GPU(const DevMem2D& left, const DevMem2D& right, const DevMem2D& disp, int ndisp, int winsz, const DevMem2D_<uint>& minSSD_buf, const cudaStream_t & stream);
        extern "C" void prefilter_xsobel(const DevMem2D& input, const DevMem2D& output, int prefilterCap = 31);
        extern "C" void postfilter_textureness(const DevMem2D& input, int winsz, float avergeTexThreshold, const DevMem2D& disp);
    }
}}

const float defaultAvgTexThreshold = 3;

cv::gpu::StereoBM_GPU::StereoBM_GPU()
    : preset(BASIC_PRESET), ndisp(DEFAULT_NDISP), winSize(DEFAULT_WINSZ), avergeTexThreshold(defaultAvgTexThreshold)  {}

cv::gpu::StereoBM_GPU::StereoBM_GPU(int preset_, int ndisparities_, int winSize_)
    : preset(preset_), ndisp(ndisparities_), winSize(winSize_), avergeTexThreshold(defaultAvgTexThreshold)
{
    const int max_supported_ndisp = 1 << (sizeof(unsigned char) * 8);
    CV_Assert(0 < ndisp && ndisp <= max_supported_ndisp);
    CV_Assert(ndisp % 8 == 0);
    CV_Assert(winSize % 2 == 1);
}

bool cv::gpu::StereoBM_GPU::checkIfGpuCallReasonable()
{
    if (0 == getCudaEnabledDeviceCount())
        return false;

    int device = getDevice();

    int minor, major;
    getComputeCapability(device, &major, &minor);
    int numSM = getNumberOfSMs(device);

    if (major > 1 || numSM > 16)
        return true;

    return false;
}

void stereo_gpu_operator ( GpuMat& minSSD,  GpuMat& leBuf, GpuMat&  riBuf,  int preset, int ndisp, int winSize, float avergeTexThreshold, const GpuMat& left, const GpuMat& right, GpuMat& disparity, const cudaStream_t & stream)
{
    CV_DbgAssert(left.rows == right.rows && left.cols == right.cols);
    CV_DbgAssert(left.type() == CV_8UC1);
    CV_DbgAssert(right.type() == CV_8UC1);

    disparity.create(left.size(), CV_8U);
    minSSD.create(left.size(), CV_32S);

    GpuMat le_for_bm =  left;
    GpuMat ri_for_bm = right;

    if (preset == StereoBM_GPU::PREFILTER_XSOBEL)
    {
        leBuf.create( left.size(),  left.type());
        riBuf.create(right.size(), right.type());

        impl::prefilter_xsobel( left, leBuf);
        impl::prefilter_xsobel(right, riBuf);

        le_for_bm = leBuf;
        ri_for_bm = riBuf;
    }

    impl::stereoBM_GPU(le_for_bm, ri_for_bm, disparity, ndisp, winSize, minSSD, stream);

    if (avergeTexThreshold)
        impl::postfilter_textureness(le_for_bm, winSize, avergeTexThreshold, disparity);
}


void cv::gpu::StereoBM_GPU::operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity)
{
    ::stereo_gpu_operator(minSSD, leBuf, riBuf, preset, ndisp, winSize, avergeTexThreshold, left, right, disparity, 0);
}

void cv::gpu::StereoBM_GPU::operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity, const CudaStream& stream)
{
    ::stereo_gpu_operator(minSSD, leBuf, riBuf, preset, ndisp, winSize, avergeTexThreshold, left, right, disparity, StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
