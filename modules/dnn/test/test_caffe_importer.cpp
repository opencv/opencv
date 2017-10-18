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
#include <opencv2/dnn/shape_utils.hpp>

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_Caffe, read_gtsrb)
{
    Net net = readNetFromCaffe(_tf("gtsrb.prototxt"));
    ASSERT_FALSE(net.empty());
}

TEST(Test_Caffe, read_googlenet)
{
    Net net = readNetFromCaffe(_tf("bvlc_googlenet.prototxt"));
    ASSERT_FALSE(net.empty());
}

TEST(Reproducibility_AlexNet, Accuracy)
{
    Net net;
    {
        const string proto = findDataFile("dnn/bvlc_alexnet.prototxt", false);
        const string model = findDataFile("dnn/bvlc_alexnet.caffemodel", false);
        net = readNetFromCaffe(proto, model);
        ASSERT_FALSE(net.empty());
    }

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());

    net.setInput(blobFromImage(sample, 1.0f, Size(227, 227), Scalar(), false), "data");
    Mat out = net.forward("prob");
    Mat ref = blobFromNPY(_tf("caffe_alexnet_prob.npy"));
    normAssert(ref, out);
}

#if !defined(_WIN32) || defined(_WIN64)
TEST(Reproducibility_FCN, Accuracy)
{
    Net net;
    {
        const string proto = findDataFile("dnn/fcn8s-heavy-pascal.prototxt", false);
        const string model = findDataFile("dnn/fcn8s-heavy-pascal.caffemodel", false);
        net = readNetFromCaffe(proto, model);
        ASSERT_FALSE(net.empty());
    }

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    std::vector<int> layerIds;
    std::vector<size_t> weights, blobs;
    net.getMemoryConsumption(shape(1,3,227,227), layerIds, weights, blobs);

    net.setInput(blobFromImage(sample, 1.0f, Size(500, 500), Scalar(), false), "data");
    Mat out = net.forward("score");

    Mat refData = imread(_tf("caffe_fcn8s_prob.png"), IMREAD_ANYDEPTH);
    int shape[] = {1, 21, 500, 500};
    Mat ref(4, shape, CV_32FC1, refData.data);

    normAssert(ref, out);
}
#endif

TEST(Reproducibility_SSD, Accuracy)
{
    Net net;
    {
        const string proto = findDataFile("dnn/ssd_vgg16.prototxt", false);
        const string model = findDataFile("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", false);
        net = readNetFromCaffe(proto, model);
        ASSERT_FALSE(net.empty());
    }

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    if (sample.channels() == 4)
        cvtColor(sample, sample, COLOR_BGRA2BGR);

    Mat in_blob = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    net.setInput(in_blob, "data");
    Mat out = net.forward("detection_out");

    Mat ref = blobFromNPY(_tf("ssd_out.npy"));
    normAssert(ref, out);
}

TEST(Reproducibility_ResNet50, Accuracy)
{
    Net net = readNetFromCaffe(findDataFile("dnn/ResNet-50-deploy.prototxt", false),
                               findDataFile("dnn/ResNet-50-model.caffemodel", false));

    Mat input = blobFromImage(imread(_tf("googlenet_0.png")), 1.0f, Size(224,224), Scalar(), false);
    ASSERT_TRUE(!input.empty());

    net.setInput(input);
    Mat out = net.forward();

    Mat ref = blobFromNPY(_tf("resnet50_prob.npy"));
    normAssert(ref, out);
}

TEST(Reproducibility_SqueezeNet_v1_1, Accuracy)
{
    Net net = readNetFromCaffe(findDataFile("dnn/squeezenet_v1.1.prototxt", false),
                               findDataFile("dnn/squeezenet_v1.1.caffemodel", false));

    Mat input = blobFromImage(imread(_tf("googlenet_0.png")), 1.0f, Size(227,227), Scalar(), false);
    ASSERT_TRUE(!input.empty());

    net.setInput(input);
    Mat out = net.forward();

    Mat ref = blobFromNPY(_tf("squeezenet_v1.1_prob.npy"));
    normAssert(ref, out);
}

TEST(Reproducibility_AlexNet_fp16, Accuracy)
{
    const float l1 = 1e-5;
    const float lInf = 3e-3;

    const string proto = findDataFile("dnn/bvlc_alexnet.prototxt", false);
    const string model = findDataFile("dnn/bvlc_alexnet.caffemodel", false);

    shrinkCaffeModel(model, "bvlc_alexnet.caffemodel_fp16");
    Net net = readNetFromCaffe(proto, "bvlc_alexnet.caffemodel_fp16");

    Mat sample = imread(findDataFile("dnn/grace_hopper_227.png", false));

    net.setInput(blobFromImage(sample, 1.0f, Size(227, 227), Scalar(), false));
    Mat out = net.forward();
    Mat ref = blobFromNPY(findDataFile("dnn/caffe_alexnet_prob.npy", false));
    normAssert(ref, out, "", l1, lInf);
}

TEST(Reproducibility_GoogLeNet_fp16, Accuracy)
{
    const float l1 = 1e-5;
    const float lInf = 3e-3;

    const string proto = findDataFile("dnn/bvlc_googlenet.prototxt", false);
    const string model = findDataFile("dnn/bvlc_googlenet.caffemodel", false);

    shrinkCaffeModel(model, "bvlc_googlenet.caffemodel_fp16");
    Net net = readNetFromCaffe(proto, "bvlc_googlenet.caffemodel_fp16");

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(_tf("googlenet_0.png")) );
    inpMats.push_back( imread(_tf("googlenet_1.png")) );
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    net.setInput(blobFromImages(inpMats, 1.0f, Size(), Scalar(), false), "data");
    Mat out = net.forward("prob");

    Mat ref = blobFromNPY(_tf("googlenet_prob.npy"));
    normAssert(out, ref, "", l1, lInf);
}

// https://github.com/richzhang/colorization
TEST(Reproducibility_Colorization, Accuracy)
{
    const float l1 = 1e-5;
    const float lInf = 3e-3;

    Mat inp = blobFromNPY(_tf("colorization_inp.npy"));
    Mat ref = blobFromNPY(_tf("colorization_out.npy"));
    Mat kernel = blobFromNPY(_tf("colorization_pts_in_hull.npy"));

    const string proto = findDataFile("dnn/colorization_deploy_v2.prototxt", false);
    const string model = findDataFile("dnn/colorization_release_v2.caffemodel", false);
    Net net = readNetFromCaffe(proto, model);

    net.getLayer(net.getLayerId("class8_ab"))->blobs.push_back(kernel);
    net.getLayer(net.getLayerId("conv8_313_rh"))->blobs.push_back(Mat(1, 313, CV_32F, 2.606));

    net.setInput(inp);
    Mat out = net.forward();

    normAssert(out, ref, "", l1, lInf);
}

}
