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
    Net net = readNet(findDataFile("dnn/squeezenet_v1.1.prototxt"),
                      findDataFile("dnn/squeezenet_v1.1.caffemodel", false));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/opencv_face_detector.caffemodel", false),
                  findDataFile("dnn/opencv_face_detector.prototxt"));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/openface_nn4.small2.v1.t7", false));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/tiny-yolo-voc.cfg"),
                  findDataFile("dnn/tiny-yolo-voc.weights", false));
    EXPECT_FALSE(net.empty());
    net = readNet(findDataFile("dnn/ssd_mobilenet_v1_coco.pbtxt"),
                  findDataFile("dnn/ssd_mobilenet_v1_coco.pb", false));
    EXPECT_FALSE(net.empty());
}

typedef testing::TestWithParam<tuple<Backend, Target> > dump;
TEST_P(dump, Regression)
{
    const int backend  = get<0>(GetParam());
    const int target   = get<1>(GetParam());
    Net net = readNet(findDataFile("dnn/squeezenet_v1.1.prototxt"),
                      findDataFile("dnn/squeezenet_v1.1.caffemodel", false));

    int size[] = {1, 3, 227, 227};
    Mat input = cv::Mat::ones(4, size, CV_32F);
    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    EXPECT_FALSE(net.dump().empty());
    net.forward();
    EXPECT_FALSE(net.dump().empty());
}

INSTANTIATE_TEST_CASE_P(/**/, dump, dnnBackendsAndTargets());

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
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

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

TEST(Net, forwardAndRetrieve)
{
    std::string prototxt =
        "input: \"data\"\n"
        "layer {\n"
        "  name: \"testLayer\"\n"
        "  type: \"Slice\"\n"
        "  bottom: \"data\"\n"
        "  top: \"firstCopy\"\n"
        "  top: \"secondCopy\"\n"
        "  slice_param {\n"
        "    axis: 0\n"
        "    slice_point: 2\n"
        "  }\n"
        "}";
    Net net = readNetFromCaffe(&prototxt[0], prototxt.size());
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat inp(4, 5, CV_32F);
    randu(inp, -1, 1);
    net.setInput(inp);

    std::vector<String> outNames;
    outNames.push_back("testLayer");
    std::vector<std::vector<Mat> > outBlobs;

    net.forward(outBlobs, outNames);

    EXPECT_EQ(outBlobs.size(), 1);
    EXPECT_EQ(outBlobs[0].size(), 2);
    normAssert(outBlobs[0][0], inp.rowRange(0, 2), "first part");
    normAssert(outBlobs[0][1], inp.rowRange(2, 4), "second part");
}

#ifdef HAVE_INF_ENGINE
static const std::chrono::milliseconds async_timeout(500);

// This test runs network in synchronous mode for different inputs and then
// runs the same model asynchronously for the same inputs.
typedef testing::TestWithParam<tuple<int, Target> > Async;
TEST_P(Async, set_and_forward_single)
{
    const int dtype = get<0>(GetParam());
    const int target = get<1>(GetParam());

    const std::string suffix = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? "_fp16" : "";
    const std::string& model = findDataFile("dnn/layers/layer_convolution" + suffix + ".bin");
    const std::string& proto = findDataFile("dnn/layers/layer_convolution" + suffix + ".xml");

    Net netSync = readNet(model, proto);
    netSync.setPreferableTarget(target);

    Net netAsync = readNet(model, proto);
    netAsync.setPreferableTarget(target);

    // Generate inputs.
    const int numInputs = 10;
    std::vector<Mat> inputs(numInputs);
    int blobSize[] = {2, 6, 75, 113};
    for (int i = 0; i < numInputs; ++i)
    {
        inputs[i].create(4, &blobSize[0], dtype);
        randu(inputs[i], 0, 255);
    }

    // Run synchronously.
    std::vector<Mat> refs(numInputs);
    for (int i = 0; i < numInputs; ++i)
    {
        netSync.setInput(inputs[i]);
        refs[i] = netSync.forward().clone();
    }

    // Run asynchronously. To make test more robust, process inputs in the reversed order.
    for (int i = numInputs - 1; i >= 0; --i)
    {
        netAsync.setInput(inputs[i]);

        AsyncArray out = netAsync.forwardAsync();
        ASSERT_TRUE(out.valid());
        Mat result;
        EXPECT_TRUE(out.get(result, async_timeout));
        normAssert(refs[i], result, format("Index: %d", i).c_str(), 0, 0);
    }
}

TEST_P(Async, set_and_forward_all)
{
    const int dtype = get<0>(GetParam());
    const int target = get<1>(GetParam());

    const std::string suffix = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? "_fp16" : "";
    const std::string& model = findDataFile("dnn/layers/layer_convolution" + suffix + ".bin");
    const std::string& proto = findDataFile("dnn/layers/layer_convolution" + suffix + ".xml");


    Net netSync = readNet(model, proto);
    netSync.setPreferableTarget(target);

    Net netAsync = readNet(model, proto);
    netAsync.setPreferableTarget(target);

    // Generate inputs.
    const int numInputs = 10;
    std::vector<Mat> inputs(numInputs);
    int blobSize[] = {2, 6, 75, 113};
    for (int i = 0; i < numInputs; ++i)
    {
        inputs[i].create(4, &blobSize[0], dtype);
        randu(inputs[i], 0, 255);
    }

    // Run synchronously.
    std::vector<Mat> refs(numInputs);
    for (int i = 0; i < numInputs; ++i)
    {
        netSync.setInput(inputs[i]);
        refs[i] = netSync.forward().clone();
    }

    // Run asynchronously. To make test more robust, process inputs in the reversed order.
    std::vector<AsyncArray> outs(numInputs);
    for (int i = numInputs - 1; i >= 0; --i)
    {
        netAsync.setInput(inputs[i]);
        outs[i] = netAsync.forwardAsync();
    }

    for (int i = numInputs - 1; i >= 0; --i)
    {
        ASSERT_TRUE(outs[i].valid());
        Mat result;
        EXPECT_TRUE(outs[i].get(result, async_timeout));
        normAssert(refs[i], result, format("Index: %d", i).c_str(), 0, 0);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Async, Combine(
  Values(CV_32F, CV_8U),
  testing::ValuesIn(getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE))
));
#endif  // HAVE_INF_ENGINE

}} // namespace
