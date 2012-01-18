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
using namespace std;

#if !defined (HAVE_CUDA)

void cv::gpu::BroxOpticalFlow::operator ()(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::interpolateFrames(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, float, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::createOpticalFlowNeedleMap(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }

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
    static void outputHandler(const std::string &msg) { CV_Error(CV_GpuApiCallError, msg.c_str()); }
}

void cv::gpu::BroxOpticalFlow::operator ()(const GpuMat& frame0, const GpuMat& frame1, GpuMat& u, GpuMat& v, Stream& s) 
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

    NCVMatrixReuse<Ncv32f> frame0Mat(frame0MemSeg, devProp.textureAlignment, frame0.cols, frame0.rows, frame0.step);
    NCVMatrixReuse<Ncv32f> frame1Mat(frame1MemSeg, devProp.textureAlignment, frame1.cols, frame1.rows, frame1.step);
    NCVMatrixReuse<Ncv32f> uMat(uMemSeg, devProp.textureAlignment, u.cols, u.rows, u.step);
    NCVMatrixReuse<Ncv32f> vMat(vMemSeg, devProp.textureAlignment, v.cols, v.rows, v.step);

    cudaStream_t stream = StreamAccessor::getStream(s);

    size_t bufSize = getBufSize(desc, frame0Mat, frame1Mat, uMat, vMat, devProp);

    ensureSizeIsEnough(1, bufSize, CV_8UC1, buf);

    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(devProp.textureAlignment), buf.ptr());
    
    ncvSafeCall( NCVBroxOpticalFlow(desc, gpuAllocator, frame0Mat, frame1Mat, uMat, vMat, stream) );
}

void cv::gpu::interpolateFrames(const GpuMat& frame0, const GpuMat& frame1, const GpuMat& fu, const GpuMat& fv, const GpuMat& bu, const GpuMat& bv, 
                                float pos, GpuMat& newFrame, GpuMat& buf, Stream& s)
{
    CV_Assert(frame0.type() == CV_32FC1);
    CV_Assert(frame1.size() == frame0.size() && frame1.type() == frame0.type());
    CV_Assert(fu.size() == frame0.size() && fu.type() == frame0.type());
    CV_Assert(fv.size() == frame0.size() && fv.type() == frame0.type());
    CV_Assert(bu.size() == frame0.size() && bu.type() == frame0.type());
    CV_Assert(bv.size() == frame0.size() && bv.type() == frame0.type());

    newFrame.create(frame0.size(), frame0.type());

    buf.create(6 * frame0.rows, frame0.cols, CV_32FC1);
    buf.setTo(Scalar::all(0));
    
    // occlusion masks
    GpuMat occ0 = buf.rowRange(0 * frame0.rows, 1 * frame0.rows);
    GpuMat occ1 = buf.rowRange(1 * frame0.rows, 2 * frame0.rows);

    // interpolated forward flow
    GpuMat fui = buf.rowRange(2 * frame0.rows, 3 * frame0.rows);
    GpuMat fvi = buf.rowRange(3 * frame0.rows, 4 * frame0.rows);

    // interpolated backward flow
    GpuMat bui = buf.rowRange(4 * frame0.rows, 5 * frame0.rows);
    GpuMat bvi = buf.rowRange(5 * frame0.rows, 6 * frame0.rows);

    size_t step = frame0.step;

    CV_Assert(frame1.step == step && fu.step == step && fv.step == step && bu.step == step && bv.step == step && newFrame.step == step && buf.step == step);

    cudaStream_t stream = StreamAccessor::getStream(s);
    NppStStreamHandler h(stream);

    NppStInterpolationState state;

    state.size         = NcvSize32u(frame0.cols, frame0.rows);
    state.nStep        = static_cast<Ncv32u>(step);
    state.pSrcFrame0   = const_cast<Ncv32f*>(frame0.ptr<Ncv32f>());
    state.pSrcFrame1   = const_cast<Ncv32f*>(frame1.ptr<Ncv32f>());
    state.pFU          = const_cast<Ncv32f*>(fu.ptr<Ncv32f>());
    state.pFV          = const_cast<Ncv32f*>(fv.ptr<Ncv32f>());
    state.pBU          = const_cast<Ncv32f*>(bu.ptr<Ncv32f>());
    state.pBV          = const_cast<Ncv32f*>(bv.ptr<Ncv32f>());
    state.pos          = pos;
    state.pNewFrame    = newFrame.ptr<Ncv32f>();
    state.ppBuffers[0] = occ0.ptr<Ncv32f>();
    state.ppBuffers[1] = occ1.ptr<Ncv32f>();
    state.ppBuffers[2] = fui.ptr<Ncv32f>();
    state.ppBuffers[3] = fvi.ptr<Ncv32f>();
    state.ppBuffers[4] = bui.ptr<Ncv32f>();
    state.ppBuffers[5] = bvi.ptr<Ncv32f>();

    ncvSafeCall( nppiStInterpolateFrames(&state) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

namespace cv { namespace gpu { namespace device 
{
    namespace optical_flow
    {
        void NeedleMapAverage_gpu(DevMem2Df u, DevMem2Df v, DevMem2Df u_avg, DevMem2Df v_avg);
        void CreateOpticalFlowNeedleMap_gpu(DevMem2Df u_avg, DevMem2Df v_avg, float* vertex_buffer, float* color_data, float max_flow, float xscale, float yscale);
    }
}}}

void cv::gpu::createOpticalFlowNeedleMap(const GpuMat& u, const GpuMat& v, GpuMat& vertex, GpuMat& colors)
{
    using namespace cv::gpu::device::optical_flow;

    CV_Assert(u.type() == CV_32FC1);
    CV_Assert(v.type() == u.type() && v.size() == u.size());

    const int NEEDLE_MAP_SCALE = 16;

	const int x_needles = u.cols / NEEDLE_MAP_SCALE;
	const int y_needles = u.rows / NEEDLE_MAP_SCALE;

    GpuMat u_avg(y_needles, x_needles, CV_32FC1);
    GpuMat v_avg(y_needles, x_needles, CV_32FC1);
    
    NeedleMapAverage_gpu(u, v, u_avg, v_avg);
    
    const int NUM_VERTS_PER_ARROW = 6;
    
    const int num_arrows = x_needles * y_needles * NUM_VERTS_PER_ARROW;

    vertex.create(1, num_arrows, CV_32FC3);
    colors.create(1, num_arrows, CV_32FC3);

    colors.setTo(Scalar::all(1.0));

    double uMax, vMax;
    minMax(u_avg, 0, &uMax);
    minMax(v_avg, 0, &vMax);

    float max_flow = static_cast<float>(sqrt(uMax * uMax + vMax * vMax));

    CreateOpticalFlowNeedleMap_gpu(u_avg, v_avg, vertex.ptr<float>(), colors.ptr<float>(), max_flow, 1.0f / u.cols, 1.0f / u.rows);

    cvtColor(colors, colors, COLOR_HSV2RGB);
}

#endif /* HAVE_CUDA */
