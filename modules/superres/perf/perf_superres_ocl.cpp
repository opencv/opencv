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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
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
#include "opencv2/ocl/ocl.hpp"

#ifdef HAVE_OPENCL

using namespace std;
using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::superres;

namespace
{
    class OneFrameSource_OCL : public FrameSource
    {
    public:
        explicit OneFrameSource_OCL(const ocl::oclMat& frame) : frame_(frame) {}

        void nextFrame(OutputArray)
        {
        }

        void nextFrame(ocl::oclMat& frame)
        {
            frame_.copyTo(frame);
        }
        void reset()
        {
        }

    private:
        ocl::oclMat frame_;
    };


    class ZeroOpticalFlowOCL : public DenseOpticalFlowExt
    {
    public:
        void calc(ocl::oclMat& frame0, ocl::oclMat&, ocl::oclMat& flow1, ocl::oclMat& flow2)
        {
            cv::Size size = frame0.size();

            flow1.create(size, CV_32FC1);
            flow2.create(size, CV_32FC1);

            flow1.setTo(Scalar::all(0));
            flow2.setTo(Scalar::all(0));
        }

        void collectGarbage()
        {
        }
    };
}

PERF_TEST_P(Size_MatType, SuperResolution_BTVL1_OCL,
    Combine(Values(szSmall64, szSmall128),
    Values(MatType(CV_8UC1), MatType(CV_8UC3))))
{
    std::vector<cv::ocl::Info>info;
    cv::ocl::getDevice(info);

    declare.time(5 * 60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    declare.in(frame, WARMUP_RNG);

    ocl::oclMat frame_ocl;
    frame_ocl.upload(frame);


    const int scale = 2;
    const int iterations = 50;
    const int temporalAreaRadius = 1;
    Ptr<DenseOpticalFlowExt> opticalFlowOcl(new ZeroOpticalFlowOCL);

    Ptr<SuperResolution> superRes_ocl = createSuperResolution_BTVL1_OCL();

    superRes_ocl->set("scale", scale);
    superRes_ocl->set("iterations", iterations);
    superRes_ocl->set("temporalAreaRadius", temporalAreaRadius);
    superRes_ocl->set("opticalFlow", opticalFlowOcl);

    superRes_ocl->setInput(new OneFrameSource_OCL(frame_ocl));

    ocl::oclMat dst_ocl;
    superRes_ocl->nextFrame(dst_ocl);

    TEST_CYCLE_N(10) superRes_ocl->nextFrame(dst_ocl);
    frame_ocl.release();
    CPU_SANITY_CHECK(dst_ocl);
}
#endif