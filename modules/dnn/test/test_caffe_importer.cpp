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
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("gtsrb.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

TEST(Test_Caffe, read_googlenet)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("bvlc_googlenet.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

TEST(Reproducibility_AlexNet, Accuracy)
{
    Net net;
    {
        const string proto = findDataFile("dnn/bvlc_alexnet.prototxt", false);
        const string model = findDataFile("dnn/bvlc_alexnet.caffemodel", false);
        Ptr<Importer> importer = createCaffeImporter(proto, model);
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());

    Size inputSize(227, 227);

    if (sample.size() != inputSize)
        resize(sample, sample, inputSize);

    net.setInput(blobFromImage(sample), "data");
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
        Ptr<Importer> importer = createCaffeImporter(proto, model);
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    Size inputSize(500, 500);
    if (sample.size() != inputSize)
        resize(sample, sample, inputSize);

    std::vector<int> layerIds;
    std::vector<size_t> weights, blobs;
    net.getMemoryConsumption(shape(1,3,227,227), layerIds, weights, blobs);

    net.setInput(blobFromImage(sample), "data");
    Mat out = net.forward("score");
    Mat ref = blobFromNPY(_tf("caffe_fcn8s_prob.npy"));
    normAssert(ref, out);
}
#endif

TEST(Reproducibility_SSD, Accuracy)
{
    Net net;
    {
        const string proto = findDataFile("dnn/ssd_vgg16.prototxt", false);
        const string model = findDataFile("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", false);
        Ptr<Importer> importer = createCaffeImporter(proto, model);
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    if (sample.channels() == 4)
        cvtColor(sample, sample, COLOR_BGRA2BGR);

    sample.convertTo(sample, CV_32F);
    resize(sample, sample, Size(300, 300));

    Mat in_blob = blobFromImage(sample);
    net.setInput(in_blob, "data");
    Mat out = net.forward("detection_out");

    Mat ref = blobFromNPY(_tf("ssd_out.npy"));
    normAssert(ref, out);
}

TEST(Reproducibility_ResNet50, Accuracy)
{
    Net net = readNetFromCaffe(findDataFile("dnn/ResNet-50-deploy.prototxt", false),
                               findDataFile("dnn/ResNet-50-model.caffemodel", false));

    Mat input = blobFromImage(imread(_tf("googlenet_0.png")), 1, Size(224,224));
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

    Mat input = blobFromImage(imread(_tf("googlenet_0.png")), 1, Size(227,227));
    ASSERT_TRUE(!input.empty());

    net.setInput(input);
    Mat out = net.forward();

    Mat ref = blobFromNPY(_tf("squeezenet_v1.1_prob.npy"));
    normAssert(ref, out);
}

}
