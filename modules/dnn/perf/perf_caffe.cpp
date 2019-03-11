// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// Recommends run this performance test via
// ./bin/opencv_perf_dnn 2> /dev/null | grep "PERFSTAT" -A 3
// because whole output includes Caffe's logs.
//
// Note: Be sure that interesting version of Caffe was linked.
// Note: There is an impact on Halide performance. Comment this tests if you
//       want to run the last one.
//
// How to build Intel-Caffe with MKLDNN backend
// ============================================
// mkdir build && cd build
// cmake -DCMAKE_BUILD_TYPE=Release \
//       -DUSE_MKLDNN_AS_DEFAULT_ENGINE=ON \
//       -DUSE_MKL2017_AS_DEFAULT_ENGINE=OFF \
//       -DCPU_ONLY=ON \
//       -DCMAKE_INSTALL_PREFIX=/usr/local .. && make -j8
// sudo make install
//
// In case of problems with cublas_v2.h at include/caffe/util/device_alternate.hpp: add line
// #define CPU_ONLY
// before the first line
// #ifdef CPU_ONLY  // CPU-only Caffe.

#if defined(HAVE_CAFFE) || defined(HAVE_CLCAFFE)

#include "perf_precomp.hpp"
#include <iostream>
#include <caffe/caffe.hpp>

namespace opencv_test {

static caffe::Net<float>* initNet(std::string proto, std::string weights)
{
    proto = findDataFile(proto, false);
    weights = findDataFile(weights, false);

#ifdef HAVE_CLCAFFE
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);

    caffe::Net<float>* net =
        new caffe::Net<float>(proto, caffe::TEST, caffe::Caffe::GetDefaultDevice());
#else
    caffe::Caffe::set_mode(caffe::Caffe::CPU);

    caffe::Net<float>* net = new caffe::Net<float>(proto, caffe::TEST);
#endif

    net->CopyTrainedLayersFrom(weights);

    caffe::Blob<float>* input = net->input_blobs()[0];

    CV_Assert(input->num() == 1);
    CV_Assert(input->channels() == 3);

    Mat inputMat(input->height(), input->width(), CV_32FC3, (char*)input->cpu_data());
    randu(inputMat, 0.0f, 1.0f);

    net->Forward();
    return net;
}

PERF_TEST(AlexNet_caffe, CaffePerfTest)
{
    caffe::Net<float>* net = initNet("dnn/bvlc_alexnet.prototxt",
                                     "dnn/bvlc_alexnet.caffemodel");
    TEST_CYCLE() net->Forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(GoogLeNet_caffe, CaffePerfTest)
{
    caffe::Net<float>* net = initNet("dnn/bvlc_googlenet.prototxt",
                                     "dnn/bvlc_googlenet.caffemodel");
    TEST_CYCLE() net->Forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(ResNet50_caffe, CaffePerfTest)
{
    caffe::Net<float>* net = initNet("dnn/ResNet-50-deploy.prototxt",
                                     "dnn/ResNet-50-model.caffemodel");
    TEST_CYCLE() net->Forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(SqueezeNet_v1_1_caffe, CaffePerfTest)
{
    caffe::Net<float>* net = initNet("dnn/squeezenet_v1.1.prototxt",
                                     "dnn/squeezenet_v1.1.caffemodel");
    TEST_CYCLE() net->Forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(MobileNet_SSD, CaffePerfTest)
{
    caffe::Net<float>* net = initNet("dnn/MobileNetSSD_deploy.prototxt",
                                     "dnn/MobileNetSSD_deploy.caffemodel");
    TEST_CYCLE() net->Forward();
    SANITY_CHECK_NOTHING();
}

} // namespace
#endif  // HAVE_CAFFE
