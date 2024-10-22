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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/ts/ocl_test.hpp>

namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

typedef testing::TestWithParam<Target> Reproducibility_GoogLeNet;
TEST_P(Reproducibility_GoogLeNet, Batching)
{
    const int targetId = GetParam();
    if (targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (targetId == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
    Net net = readNetFromCaffe(findDataFile("dnn/bvlc_googlenet.prototxt"),
                               findDataFile("dnn/bvlc_googlenet.caffemodel", false));
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    if (targetId == DNN_TARGET_OPENCL)
    {
        // Initialize network for a single image in the batch but test with batch size=2.
        Mat inp = Mat(224, 224, CV_8UC3);
        randu(inp, -1, 1);
        net.setInput(blobFromImage(inp));
        net.forward();
    }

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(_tf("googlenet_0.png")) );
    inpMats.push_back( imread(_tf("googlenet_1.png")) );
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    net.setInput(blobFromImages(inpMats, 1.0f, Size(), Scalar(), false), "data");

    // BUG: https://github.com/opencv/opencv/issues/26349
    Mat out;
    if(net.getMainGraph())
        out = net.forward();
    else
        out = net.forward("prob");

    Mat ref = blobFromNPY(_tf("googlenet_prob.npy"));
    normAssert(out, ref);
}

TEST_P(Reproducibility_GoogLeNet, IntermediateBlobs)
{
    const int targetId = GetParam();
    if (targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (targetId == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
    // BUG: https://github.com/opencv/opencv/issues/26349
    Net net = readNetFromCaffe(findDataFile("dnn/bvlc_googlenet.prototxt"),
                               findDataFile("dnn/bvlc_googlenet.caffemodel", false), ENGINE_CLASSIC);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    std::vector<String> blobsNames;
    blobsNames.push_back("conv1/7x7_s2");
    blobsNames.push_back("conv1/relu_7x7");
    blobsNames.push_back("inception_4c/1x1");
    blobsNames.push_back("inception_4c/relu_1x1");
    std::vector<Mat> outs;
    Mat in = blobFromImage(imread(_tf("googlenet_0.png")), 1.0f, Size(), Scalar(), false);
    net.setInput(in, "data");
    net.forward(outs, blobsNames);
    CV_Assert(outs.size() == blobsNames.size());

    for (size_t i = 0; i < blobsNames.size(); i++)
    {
        std::string filename = blobsNames[i];
        std::replace( filename.begin(), filename.end(), '/', '#');
        Mat ref = blobFromNPY(_tf("googlenet_" + filename + ".npy"));

        normAssert(outs[i], ref, "", 1E-4, 1E-2);
    }
}

TEST_P(Reproducibility_GoogLeNet, SeveralCalls)
{
    const int targetId = GetParam();
    if (targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (targetId == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
    // BUG: https://github.com/opencv/opencv/issues/26349
    Net net = readNetFromCaffe(findDataFile("dnn/bvlc_googlenet.prototxt"),
                               findDataFile("dnn/bvlc_googlenet.caffemodel", false), ENGINE_CLASSIC);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(_tf("googlenet_0.png")) );
    inpMats.push_back( imread(_tf("googlenet_1.png")) );
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    net.setInput(blobFromImages(inpMats, 1.0f, Size(), Scalar(), false), "data");
    Mat out = net.forward();

    Mat ref = blobFromNPY(_tf("googlenet_prob.npy"));
    normAssert(out, ref);

    std::vector<String> blobsNames;
    blobsNames.push_back("conv1/7x7_s2");
    std::vector<Mat> outs;
    Mat in = blobFromImage(inpMats[0], 1.0f, Size(), Scalar(), false);
    net.setInput(in, "data");
    net.forward(outs, blobsNames);
    CV_Assert(outs.size() == blobsNames.size());

    ref = blobFromNPY(_tf("googlenet_conv1#7x7_s2.npy"));

    normAssert(outs[0], ref, "", 1E-4, 1E-2);
}

INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_GoogLeNet,
    testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));

}} // namespace
