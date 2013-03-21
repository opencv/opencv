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

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::superres;
using namespace cv::gpu;

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

    class OneFrameSource_GPU : public FrameSource
    {
    public:
        explicit OneFrameSource_GPU(const GpuMat& frame) : frame_(frame) {}

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
        void calc(InputArray frame0, InputArray, OutputArray flow1, OutputArray flow2)
        {
            cv::Size size = frame0.size();

            if (!flow2.needed())
            {
                flow1.create(size, CV_32FC2);

                if (flow1.kind() == cv::_InputArray::GPU_MAT)
                    flow1.getGpuMatRef().setTo(cv::Scalar::all(0));
                else
                    flow1.getMatRef().setTo(cv::Scalar::all(0));
            }
            else
            {
                flow1.create(size, CV_32FC1);
                flow2.create(size, CV_32FC1);

                if (flow1.kind() == cv::_InputArray::GPU_MAT)
                    flow1.getGpuMatRef().setTo(cv::Scalar::all(0));
                else
                    flow1.getMatRef().setTo(cv::Scalar::all(0));

                if (flow2.kind() == cv::_InputArray::GPU_MAT)
                    flow2.getGpuMatRef().setTo(cv::Scalar::all(0));
                else
                    flow2.getMatRef().setTo(cv::Scalar::all(0));
            }
        }

        void collectGarbage()
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

    if (PERF_RUN_GPU())
    {
        Ptr<SuperResolution> superRes = createSuperResolution_BTVL1_GPU();

        superRes->set("scale", scale);
        superRes->set("iterations", iterations);
        superRes->set("temporalAreaRadius", temporalAreaRadius);
        superRes->set("opticalFlow", opticalFlow);

        superRes->setInput(new OneFrameSource_GPU(GpuMat(frame)));

        GpuMat dst;
        superRes->nextFrame(dst);

        TEST_CYCLE_N(10) superRes->nextFrame(dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        Ptr<SuperResolution> superRes = createSuperResolution_BTVL1();

        superRes->set("scale", scale);
        superRes->set("iterations", iterations);
        superRes->set("temporalAreaRadius", temporalAreaRadius);
        superRes->set("opticalFlow", opticalFlow);

        superRes->setInput(new OneFrameSource_CPU(frame));

        Mat dst;
        superRes->nextFrame(dst);

        TEST_CYCLE_N(10) superRes->nextFrame(dst);

        CPU_SANITY_CHECK(dst);
    }
}
