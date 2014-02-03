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

#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::superres;
using namespace cv::cuda;

namespace
{
    class OneFrameSource_CPU : public FrameSource
    {
    public:
        explicit OneFrameSource_CPU(const Mat& frame) : frame_(frame) {}

        void nextFrame(OutputArray frame)
        {
            frame.getMatRef() = frame_;
        }

        void reset()
        {
        }

    private:
        Mat frame_;
    };

    class OneFrameSource_CUDA : public FrameSource
    {
    public:
        explicit OneFrameSource_CUDA(const GpuMat& frame) : frame_(frame) {}

        void nextFrame(OutputArray frame)
        {
            frame.getGpuMatRef() = frame_;
        }

        void reset()
        {
        }

    private:
        GpuMat frame_;
    };

    class ZeroOpticalFlow : public DenseOpticalFlowExt
    {
    public:
        virtual void calc(InputArray frame0, InputArray, OutputArray flow1, OutputArray flow2)
        {
            cv::Size size = frame0.size();

            if (!flow2.needed())
            {
                flow1.create(size, CV_32FC2);
                flow1.setTo(cv::Scalar::all(0));
            }
            else
            {
                flow1.create(size, CV_32FC1);
                flow2.create(size, CV_32FC1);

                flow1.setTo(cv::Scalar::all(0));
                flow2.setTo(cv::Scalar::all(0));
            }
        }

        virtual void collectGarbage()
        {
        }
    };
}

PERF_TEST_P(Size_MatType, SuperResolution_BTVL1,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_8UC1), MatType(CV_8UC3))))
{
    declare.time(5 * 60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    declare.in(frame, WARMUP_RNG);

    const int scale = 2;
    const int iterations = 50;
    const int temporalAreaRadius = 1;
    Ptr<DenseOpticalFlowExt> opticalFlow(new ZeroOpticalFlow);

    if (PERF_RUN_CUDA())
    {
        Ptr<SuperResolution> superRes = createSuperResolution_BTVL1_CUDA();

        superRes->set("scale", scale);
        superRes->set("iterations", iterations);
        superRes->set("temporalAreaRadius", temporalAreaRadius);
        superRes->set("opticalFlow", opticalFlow);

        superRes->setInput(makePtr<OneFrameSource_CUDA>(GpuMat(frame)));

        GpuMat dst;
        superRes->nextFrame(dst);

        TEST_CYCLE_N(10) superRes->nextFrame(dst);

        CUDA_SANITY_CHECK(dst, 2);
    }
    else
    {
        Ptr<SuperResolution> superRes = createSuperResolution_BTVL1();

        superRes->set("scale", scale);
        superRes->set("iterations", iterations);
        superRes->set("temporalAreaRadius", temporalAreaRadius);
        superRes->set("opticalFlow", opticalFlow);

        superRes->setInput(makePtr<OneFrameSource_CPU>(frame));

        Mat dst;
        superRes->nextFrame(dst);

        TEST_CYCLE_N(10) superRes->nextFrame(dst);

        CPU_SANITY_CHECK(dst);
    }
}

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

typedef Size_MatType SuperResolution_BTVL1;

OCL_PERF_TEST_P(SuperResolution_BTVL1 ,BTVL1,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_8UC1), MatType(CV_8UC3))))
{
    Size_MatType_t params = GetParam();
    const Size size = get<0>(params);
    const int type = get<1>(params);

    Mat frame(size, type);
    UMat dst(1, 1, 0);
    declare.in(frame, WARMUP_RNG);

    const int scale = 2;
    const int iterations = 50;
    const int temporalAreaRadius = 1;

    Ptr<DenseOpticalFlowExt> opticalFlow(new ZeroOpticalFlow);
    Ptr<SuperResolution> superRes = createSuperResolution_BTVL1();

    superRes->set("scale", scale);
    superRes->set("iterations", iterations);
    superRes->set("temporalAreaRadius", temporalAreaRadius);
    superRes->set("opticalFlow", opticalFlow);

    superRes->setInput(makePtr<OneFrameSource_CPU>(frame));

    // skip first frame
    superRes->nextFrame(dst);

    OCL_TEST_CYCLE_N(10) superRes->nextFrame(dst);

    SANITY_CHECK_NOTHING();
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
