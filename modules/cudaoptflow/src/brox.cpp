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

Ptr<BroxOpticalFlow> cv::cuda::BroxOpticalFlow::create(double, double, double, int, int, int) { throw_no_cuda(); return Ptr<BroxOpticalFlow>(); }

#else

namespace {

    class BroxOpticalFlowImpl : public BroxOpticalFlow
    {
    public:
        BroxOpticalFlowImpl(double alpha, double gamma, double scale_factor,
                            int inner_iterations, int outer_iterations, int solver_iterations) :
            alpha_(alpha), gamma_(gamma), scale_factor_(scale_factor),
            inner_iterations_(inner_iterations), outer_iterations_(outer_iterations),
            solver_iterations_(solver_iterations)
        {
        }

        virtual String getDefaultName() const { return "DenseOpticalFlow.BroxOpticalFlow"; }

        virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow, Stream& stream);

        virtual double getFlowSmoothness() const { return alpha_; }
        virtual void setFlowSmoothness(double alpha) { alpha_ = static_cast<float>(alpha); }

        virtual double getGradientConstancyImportance() const { return gamma_; }
        virtual void setGradientConstancyImportance(double gamma) { gamma_ = static_cast<float>(gamma); }

        virtual double getPyramidScaleFactor() const { return scale_factor_; }
        virtual void setPyramidScaleFactor(double scale_factor) { scale_factor_ = static_cast<float>(scale_factor); }

        //! number of lagged non-linearity iterations (inner loop)
        virtual int getInnerIterations() const { return inner_iterations_; }
        virtual void setInnerIterations(int inner_iterations) { inner_iterations_ = inner_iterations; }

        //! number of warping iterations (number of pyramid levels)
        virtual int getOuterIterations() const { return outer_iterations_; }
        virtual void setOuterIterations(int outer_iterations) { outer_iterations_ = outer_iterations; }

        //! number of linear system solver iterations
        virtual int getSolverIterations() const { return solver_iterations_; }
        virtual void setSolverIterations(int solver_iterations) { solver_iterations_ = solver_iterations; }

    private:
        //! flow smoothness
        float alpha_;

        //! gradient constancy importance
        float gamma_;

        //! pyramid scale factor
        float scale_factor_;

        //! number of lagged non-linearity iterations (inner loop)
        int inner_iterations_;

        //! number of warping iterations (number of pyramid levels)
        int outer_iterations_;

        //! number of linear system solver iterations
        int solver_iterations_;
    };

    static size_t getBufSize(const NCVBroxOpticalFlowDescriptor& desc,
                             const NCVMatrix<Ncv32f>& frame0, const NCVMatrix<Ncv32f>& frame1,
                             NCVMatrix<Ncv32f>& u, NCVMatrix<Ncv32f>& v,
                             size_t textureAlignment)
    {
        NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(textureAlignment));

        ncvSafeCall( NCVBroxOpticalFlow(desc, gpuCounter, frame0, frame1, u, v, 0) );

        return gpuCounter.maxSize();
    }

    static void outputHandler(const String &msg)
    {
        CV_Error(cv::Error::GpuApiCallError, msg.c_str());
    }

    void BroxOpticalFlowImpl::calc(InputArray _I0, InputArray _I1, InputOutputArray _flow, Stream& stream)
    {
        const GpuMat frame0 = _I0.getGpuMat();
        const GpuMat frame1 = _I1.getGpuMat();

        CV_Assert( frame0.type() == CV_32FC1 );
        CV_Assert( frame1.size() == frame0.size() && frame1.type() == frame0.type() );

        ncvSetDebugOutputHandler(outputHandler);

        BufferPool pool(stream);
        GpuMat u = pool.getBuffer(frame0.size(), CV_32FC1);
        GpuMat v = pool.getBuffer(frame0.size(), CV_32FC1);

        NCVBroxOpticalFlowDescriptor desc;
        desc.alpha = alpha_;
        desc.gamma = gamma_;
        desc.scale_factor = scale_factor_;
        desc.number_of_inner_iterations = inner_iterations_;
        desc.number_of_outer_iterations = outer_iterations_;
        desc.number_of_solver_iterations = solver_iterations_;

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

        DeviceInfo devInfo;
        size_t textureAlignment = devInfo.textureAlignment();

        NCVMatrixReuse<Ncv32f> frame0Mat(frame0MemSeg, static_cast<Ncv32u>(textureAlignment), frame0.cols, frame0.rows, static_cast<Ncv32u>(frame0.step));
        NCVMatrixReuse<Ncv32f> frame1Mat(frame1MemSeg, static_cast<Ncv32u>(textureAlignment), frame1.cols, frame1.rows, static_cast<Ncv32u>(frame1.step));
        NCVMatrixReuse<Ncv32f> uMat(uMemSeg, static_cast<Ncv32u>(textureAlignment), u.cols, u.rows, static_cast<Ncv32u>(u.step));
        NCVMatrixReuse<Ncv32f> vMat(vMemSeg, static_cast<Ncv32u>(textureAlignment), v.cols, v.rows, static_cast<Ncv32u>(v.step));

        size_t bufSize = getBufSize(desc, frame0Mat, frame1Mat, uMat, vMat, textureAlignment);
        GpuMat buf = pool.getBuffer(1, static_cast<int>(bufSize), CV_8UC1);

        NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(textureAlignment), buf.ptr());

        ncvSafeCall( NCVBroxOpticalFlow(desc, gpuAllocator, frame0Mat, frame1Mat, uMat, vMat, StreamAccessor::getStream(stream)) );

        GpuMat flows[] = {u, v};
        cuda::merge(flows, 2, _flow, stream);
    }
}

Ptr<BroxOpticalFlow> cv::cuda::BroxOpticalFlow::create(double alpha, double gamma, double scale_factor, int inner_iterations, int outer_iterations, int solver_iterations)
{
    return makePtr<BroxOpticalFlowImpl>(alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations);
}

#endif /* HAVE_CUDA */
