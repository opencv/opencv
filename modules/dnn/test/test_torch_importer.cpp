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

#ifdef ENABLE_TORCH_IMPORTER

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cvtest
{

using namespace std;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

template<typename TStr>
static std::string _tf(TStr filename, bool inTorchDir = true)
{
    String path = getOpenCVExtraDir() + "/dnn/";
    if (inTorchDir)
        path += "torch/";
    path += filename;
    return path;
}

TEST(Torch_Importer, simple_read)
{
    Net net;
    Ptr<Importer> importer;

    ASSERT_NO_THROW( importer = createTorchImporter(_tf("net_simple_net.txt"), false) );
    ASSERT_TRUE( importer != NULL );
    importer->populateNet(net);
}

static void runTorchNet(String prefix, String outLayerName = "",
                        bool check2ndBlob = false, bool isBinary = false)
{
    String suffix = (isBinary) ? ".dat" : ".txt";

    Net net;
    Ptr<Importer> importer = createTorchImporter(_tf(prefix + "_net" + suffix), isBinary);
    ASSERT_TRUE(importer != NULL);
    importer->populateNet(net);

    Mat inp, outRef;
    ASSERT_NO_THROW( inp = readTorchBlob(_tf(prefix + "_input" + suffix), isBinary) );
    ASSERT_NO_THROW( outRef = readTorchBlob(_tf(prefix + "_output" + suffix), isBinary) );

    if (outLayerName.empty())
        outLayerName = net.getLayerNames().back();

    net.setInput(inp, "0");
    std::vector<Mat> outBlobs;
    net.forward(outBlobs, outLayerName);
    normAssert(outRef, outBlobs[0]);

    if (check2ndBlob)
    {
        Mat out2 = outBlobs[1];
        Mat ref2 = readTorchBlob(_tf(prefix + "_output_2" + suffix), isBinary);
        normAssert(out2, ref2);
    }
}

TEST(Torch_Importer, run_convolution)
{
    runTorchNet("net_conv");
}

TEST(Torch_Importer, run_pool_max)
{
    runTorchNet("net_pool_max", "", true);
}

TEST(Torch_Importer, run_pool_ave)
{
    runTorchNet("net_pool_ave");
}

TEST(Torch_Importer, run_reshape)
{
    runTorchNet("net_reshape");
    runTorchNet("net_reshape_batch");
    runTorchNet("net_reshape_single_sample");
}

TEST(Torch_Importer, run_linear)
{
    runTorchNet("net_linear_2d");
}

TEST(Torch_Importer, run_paralel)
{
    runTorchNet("net_parallel", "l5_torchMerge");
}

TEST(Torch_Importer, run_concat)
{
    runTorchNet("net_concat", "l5_torchMerge");
}

TEST(Torch_Importer, run_deconv)
{
    runTorchNet("net_deconv");
}

TEST(Torch_Importer, run_batch_norm)
{
    runTorchNet("net_batch_norm");
}

TEST(Torch_Importer, net_prelu)
{
    runTorchNet("net_prelu");
}

TEST(Torch_Importer, net_cadd_table)
{
    runTorchNet("net_cadd_table");
}

TEST(Torch_Importer, net_softmax)
{
    runTorchNet("net_softmax");
    runTorchNet("net_softmax_spatial");
}

TEST(Torch_Importer, net_logsoftmax)
{
    runTorchNet("net_logsoftmax");
    runTorchNet("net_logsoftmax_spatial");
}

TEST(Torch_Importer, ENet_accuracy)
{
    Net net;
    {
        const string model = findDataFile("dnn/Enet-model-best.net", false);
        Ptr<Importer> importer = createTorchImporter(model, true);
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Mat sample = imread(_tf("street.png", false));
    Mat inputBlob = blobFromImage(sample, 1./255);

    net.setInput(inputBlob, "");
    Mat out = net.forward();
    Mat ref = blobFromNPY(_tf("torch_enet_prob.npy", false));
    // Due to numerical instability in Pooling-Unpooling layers (indexes jittering)
    // thresholds for ENet must be changed. Accuracy of resuults was checked on
    // Cityscapes dataset and difference in mIOU with Torch is 10E-4%
    normAssert(ref, out, "", 0.00044, 0.44);

    const int N = 3;
    for (int i = 0; i < N; i++)
    {
        net.setInput(inputBlob, "");
        Mat out = net.forward();
        normAssert(ref, out, "", 0.00044, 0.44);
    }
}

}

#endif
