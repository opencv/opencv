// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opencl/ocl_defs.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

namespace opencv_test { namespace {

TEST(blobFromImage_4ch, Regression)
{
    Mat ch[4];
    for(int i = 0; i < 4; i++)
        ch[i] = Mat::ones(10, 10, CV_8U)*i;

    Mat img;
    merge(ch, 4, img);
    Mat blob = dnn::blobFromImage(img, 1., Size(), Scalar(), false, false);

    for(int i = 0; i < 4; i++)
    {
        ch[i] = Mat(img.rows, img.cols, CV_32F, blob.ptr(0, i));
        ASSERT_DOUBLE_EQ(cvtest::norm(ch[i], cv::NORM_INF), i);
    }
}

TEST(blobFromImage, allocated)
{
    int size[] = {1, 3, 4, 5};
    Mat img(size[2], size[3], CV_32FC(size[1]));
    Mat blob(4, size, CV_32F);
    void* blobData = blob.data;
    dnn::blobFromImage(img, blob, 1.0 / 255, Size(), Scalar(), false, false);
    ASSERT_EQ(blobData, blob.data);
}

TEST(imagesFromBlob, Regression)
{
    int nbOfImages = 8;

    std::vector<cv::Mat> inputImgs(nbOfImages);
    for (int i = 0; i < nbOfImages; i++)
    {
        inputImgs[i] = cv::Mat::ones(100, 100, CV_32FC3);
        cv::randu(inputImgs[i], cv::Scalar::all(0), cv::Scalar::all(1));
    }

    cv::Mat blob = cv::dnn::blobFromImages(inputImgs, 1., cv::Size(), cv::Scalar(), false, false);
    std::vector<cv::Mat> outputImgs;
    cv::dnn::imagesFromBlob(blob, outputImgs);

    for (int i = 0; i < nbOfImages; i++)
    {
        ASSERT_EQ(cv::countNonZero(inputImgs[i] != outputImgs[i]), 0);
    }
}

TEST(readNet, Regression)
{
    Net net = readNet(findDataFile("dnn/squeezenet_v1.1.prototxt", false),
                      findDataFile("dnn/squeezenet_v1.1.caffemodel", false));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/opencv_face_detector.caffemodel", false),
                  findDataFile("dnn/opencv_face_detector.prototxt", false));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/openface_nn4.small2.v1.t7", false));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/tiny-yolo-voc.cfg", false),
                  findDataFile("dnn/tiny-yolo-voc.weights", false));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/ssd_mobilenet_v1_coco.pbtxt", false),
                  findDataFile("dnn/ssd_mobilenet_v1_coco.pb", false));
    EXPECT_FALSE(net.empty());
}

class FirstCustomLayer CV_FINAL : public Layer
{
public:
    FirstCustomLayer(const LayerParams &params) : Layer(params) {}

    static Ptr<Layer> create(LayerParams& params)
    {
        return Ptr<Layer>(new FirstCustomLayer(params));
    }

    void forward(InputArrayOfArrays, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);
        outputs[0].setTo(1);
    }
};

class SecondCustomLayer CV_FINAL : public Layer
{
public:
    SecondCustomLayer(const LayerParams &params) : Layer(params) {}

    static Ptr<Layer> create(LayerParams& params)
    {
        return Ptr<Layer>(new SecondCustomLayer(params));
    }

    void forward(InputArrayOfArrays, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);
        outputs[0].setTo(2);
    }
};

TEST(LayerFactory, custom_layers)
{
    LayerParams lp;
    lp.name = "name";
    lp.type = "CustomType";

    Mat inp(1, 1, CV_32FC1);
    for (int i = 0; i < 3; ++i)
    {
        if (i == 0)      { CV_DNN_REGISTER_LAYER_CLASS(CustomType, FirstCustomLayer); }
        else if (i == 1) { CV_DNN_REGISTER_LAYER_CLASS(CustomType, SecondCustomLayer); }
        else if (i == 2) { LayerFactory::unregisterLayer("CustomType"); }

        Net net;
        net.addLayerToPrev(lp.name, lp.type, lp);

        net.setInput(inp);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        Mat output = net.forward();

        if (i == 0)      { EXPECT_EQ(output.at<float>(0), 1); }
        else if (i == 1) { EXPECT_EQ(output.at<float>(0), 2); }
        else if (i == 2) { EXPECT_EQ(output.at<float>(0), 1); }
    }
    LayerFactory::unregisterLayer("CustomType");
}

typedef testing::TestWithParam<tuple<float, Vec3f, int, tuple<Backend, Target> > > setInput;
TEST_P(setInput, normalization)
{
    const float kScale = get<0>(GetParam());
    const Scalar kMean = get<1>(GetParam());
    const int dtype    = get<2>(GetParam());
    const int backend  = get<0>(get<3>(GetParam()));
    const int target   = get<1>(get<3>(GetParam()));
    const bool kSwapRB = true;

    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16 && dtype != CV_32F)
        throw SkipTestException("");

    Mat inp(5, 5, CV_8UC3);
    randu(inp, 0, 255);
    Mat ref = blobFromImage(inp, kScale, Size(), kMean, kSwapRB, /*crop*/false);

    LayerParams lp;
    Net net;
    net.addLayerToPrev("testLayer", "Identity", lp);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat blob = blobFromImage(inp, 1.0, Size(), Scalar(), kSwapRB, /*crop*/false, dtype);
    ASSERT_EQ(blob.type(), dtype);
    net.setInput(blob, "", kScale, kMean);
    Mat out = net.forward();
    ASSERT_EQ(out.type(), CV_32F);
    normAssert(ref, out, "", 4e-4, 1e-3);
}

INSTANTIATE_TEST_CASE_P(/**/, setInput, Combine(
  Values(1.0f, 1.0 / 127.5),
  Values(Vec3f(), Vec3f(50, 50, 50), Vec3f(10, 50, 140)),
  Values(CV_32F, CV_8U),
  dnnBackendsAndTargets()
));

class CustomLayerWithDeprecatedForward CV_FINAL : public Layer
{
public:
    CustomLayerWithDeprecatedForward(const LayerParams &params) : Layer(params) {}

    static Ptr<Layer> create(LayerParams& params)
    {
        return Ptr<Layer>(new CustomLayerWithDeprecatedForward(params));
    }

    virtual void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals) CV_OVERRIDE
    {
        CV_Assert_N(inputs[0]->depth() == CV_32F, outputs[0].depth() == CV_32F);
        cv::add(*inputs[0], 0.5f, outputs[0]);
    }
};

class CustomLayerWithDeprecatedForwardAndFallback CV_FINAL : public Layer
{
public:
    CustomLayerWithDeprecatedForwardAndFallback(const LayerParams &params) : Layer(params) {}

    static Ptr<Layer> create(LayerParams& params)
    {
        return Ptr<Layer>(new CustomLayerWithDeprecatedForwardAndFallback(params));
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs, OutputArrayOfArrays internals) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(preferableTarget == DNN_TARGET_OPENCL || preferableTarget == DNN_TARGET_OPENCL_FP16,
                   forward_ocl(inputs, outputs, internals));

        Layer::forward_fallback(inputs, outputs, internals);
    }

    virtual void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals) CV_OVERRIDE
    {
        CV_Assert_N(inputs[0]->depth() == CV_32F, outputs[0].depth() == CV_32F);
        cv::add(*inputs[0], 0.5f, outputs[0]);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        if (inputs_arr.depth() != CV_32F)
            return false;

        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        inputs_arr.getUMatVector(inputs);
        outputs_arr.getUMatVector(outputs);
        cv::add(inputs[0], 0.5f, outputs[0]);
        return true;
    }
#endif
};

typedef testing::TestWithParam<tuple<Backend, Target> > DeprecatedForward;
TEST_P(DeprecatedForward, CustomLayer)
{
    const int backend  = get<0>(GetParam());
    const int target   = get<1>(GetParam());

    Mat inp(5, 5, CV_32FC1);
    randu(inp, -1.0f, 1.0f);
    inp = blobFromImage(inp);

    CV_DNN_REGISTER_LAYER_CLASS(CustomType, CustomLayerWithDeprecatedForward);
    try
    {
        LayerParams lp;
        Net net;
        net.addLayerToPrev("testLayer", "CustomType", lp);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(inp);
        Mat out = net.forward();
        normAssert(out, inp + 0.5f, "", 2e-4, 7e-4);
    }
    catch (...)
    {
        LayerFactory::unregisterLayer("CustomType");
        throw;
    }
    LayerFactory::unregisterLayer("CustomType");
}

TEST_P(DeprecatedForward, CustomLayerWithFallback)
{
    const int backend  = get<0>(GetParam());
    const int target   = get<1>(GetParam());

    Mat inp(5, 5, CV_32FC1);
    randu(inp, -1.0f, 1.0f);
    inp = blobFromImage(inp);

    CV_DNN_REGISTER_LAYER_CLASS(CustomType, CustomLayerWithDeprecatedForwardAndFallback);
    try
    {
        LayerParams lp;
        Net net;
        net.addLayerToPrev("testLayer", "CustomType", lp);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(inp);
        Mat out = net.forward();
        normAssert(out, inp + 0.5f, "", 2e-4, 7e-4);
    }
    catch (...)
    {
        LayerFactory::unregisterLayer("CustomType");
        throw;
    }
    LayerFactory::unregisterLayer("CustomType");
}

INSTANTIATE_TEST_CASE_P(/**/, DeprecatedForward, dnnBackendsAndTargets());

}} // namespace
