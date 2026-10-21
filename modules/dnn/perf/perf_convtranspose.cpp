// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// ConvTranspose2 (a layer of the new graph engine) on single-node ONNX models.

#include "perf_precomp.hpp"

namespace opencv_test {

// model name, input channels, input height and width
typedef TestBaseWithParam<tuple<std::string, int, int> > Layer_ConvTranspose2;

PERF_TEST_P_(Layer_ConvTranspose2, forward)
{
    const std::string name = get<0>(GetParam());
    const int channels = get<1>(GetParam()), size = get<2>(GetParam());

    const std::string model = "dnn/onnx/models/convtranspose_perf_" + name + ".onnx";
    Net net = readNetFromONNX(findDataFile(model, true), ENGINE_OPENCV);
    Mat input({1, channels, size, size}, CV_32F);
    randu(input, -1.f, 1.f);
    net.setInput(input);
    net.forward();

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_ConvTranspose2, testing::Values(
    make_tuple(std::string("sam2_256_64"), 256, 64),
    make_tuple(std::string("sam2_64_32"), 64, 128),
    make_tuple(std::string("sam3_256_64"), 256, 72),
    make_tuple(std::string("sam3_64_32"), 64, 144),
    make_tuple(std::string("unet_256_128_k4s2p1"), 256, 32),
    make_tuple(std::string("gan_128_64_k4s2p1"), 128, 64),
    make_tuple(std::string("128_64_k3s2p1op1"), 128, 64),
    make_tuple(std::string("64_64_k3s1p1"), 64, 64),
    make_tuple(std::string("group4_128_128_k4s2p1"), 128, 32),
    make_tuple(std::string("depthwise_64_k4s2p1"), 64, 64)
));

} // namespace opencv_test
