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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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
#include <opencv2/core/ocl.hpp>
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

#ifdef HAVE_INF_ENGINE
#include <thread>
#endif

namespace opencv_test { namespace {

template<typename TString>
static String _tf(TString filename)
{
    String basetestdir = getOpenCVExtraDir();
    size_t len = basetestdir.size();
    if(len > 0 && basetestdir[len-1] != '/' && basetestdir[len-1] != '\\')
        return (basetestdir + "/dnn/layers") + filename;
    return (basetestdir + "dnn/layers/") + filename;
}

void runLayer(Ptr<Layer> layer, std::vector<Mat> &inpBlobs, std::vector<Mat> &outBlobs)
{
    size_t ninputs = inpBlobs.size();
    std::vector<Mat> inp(ninputs), outp, intp;
    std::vector<MatShape> inputs, outputs, internals;

    for (size_t i = 0; i < ninputs; i++)
    {
        inp[i] = inpBlobs[i].clone();
        inputs.push_back(shape(inp[i]));
    }

    layer->getMemoryShapes(inputs, 0, outputs, internals);
    for (size_t i = 0; i < outputs.size(); i++)
    {
        outp.push_back(Mat(outputs[i], CV_32F));
    }
    for (size_t i = 0; i < internals.size(); i++)
    {
        intp.push_back(Mat(internals[i], CV_32F));
    }

    layer->finalize(inp, outp);
    layer->forward(inp, outp, intp);

    size_t noutputs = outp.size();
    outBlobs.resize(noutputs);
    for (size_t i = 0; i < noutputs; i++)
        outBlobs[i] = outp[i];
}

class Test_Caffe_layers : public DNNTestLayer
{
public:
    void testLayerUsingCaffeModels(const String& basename, bool useCaffeModel = false,
                                   bool useCommonInputBlob = true, double l1 = 0.0, double lInf = 0.0,
                                   int numInps = 1, int numOuts = 1)
    {
        CV_Assert_N(numInps >= 1, numInps <= 10, numOuts >= 1, numOuts <= 10);
        String prototxt = _tf(basename + ".prototxt");
        String caffemodel = _tf(basename + ".caffemodel");

        std::vector<Mat> inps, refs, outs;

        if (numInps > 1)
        {
            for (int i = 0; i < numInps; i++)
            {
                String inpfile = _tf(basename + cv::format(".input_%d.npy", i));
                inps.push_back(blobFromNPY(inpfile));
            }
        }
        else
        {
            String inpfile = (useCommonInputBlob) ? _tf("blob.npy") : _tf(basename + ".input.npy");
            inps.push_back(blobFromNPY(inpfile));
        }

        if (numOuts > 1)
        {
            for (int i = 0; i < numOuts; i++)
            {
                String outfile = _tf(basename + cv::format("_%d.npy", i));
                refs.push_back(blobFromNPY(outfile));
            }
        }
        else
        {
            String outfile = _tf(basename + ".npy");
            refs.push_back(blobFromNPY(outfile));
        }

        Net net = readNetFromCaffe(prototxt, (useCaffeModel) ? caffemodel : String());
        ASSERT_FALSE(net.empty());
        checkBackend(&inps[0], &refs[0]);

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        String inp_name = "input";
        if (numInps > 1)
        {
            for (int i = 0; i < numInps; i++)
            {
                net.setInput(inps[i], inp_name + cv::format("_%d", i));
            }
        }
        else
        {
            net.setInput(inps.back(), inp_name);
        }

        net.forward(outs);
        for (int i = 0; i < refs.size(); i++)
        {
            normAssert(refs[i], outs[i], "", l1 ? l1 : default_l1, lInf ? lInf : default_lInf);
        }
    }
};

TEST_P(Test_Caffe_layers, Softmax)
{
    testLayerUsingCaffeModels("layer_softmax");
}

TEST_P(Test_Caffe_layers, LRN)
{
    double l1 = 0.0, lInf = 0.0;
    // The OpenCL kernels use the native_ math functions which have
    // implementation defined accuracy, so we use relaxed thresholds. See
    // https://github.com/opencv/opencv/issues/9821 for more details.
    if (target == DNN_TARGET_OPENCL)
    {
        l1 = 0.01;
        lInf = 0.01;
    }
    testLayerUsingCaffeModels("layer_lrn_spatial", false, true, l1, lInf);
    testLayerUsingCaffeModels("layer_lrn_channels", false, true, l1, lInf);
}

TEST_P(Test_Caffe_layers, Convolution)
{
    testLayerUsingCaffeModels("layer_convolution", true);
}

TEST_P(Test_Caffe_layers, DeConvolution)
{
    if(target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    testLayerUsingCaffeModels("layer_deconvolution", true, false);
}

TEST_P(Test_Caffe_layers, InnerProduct)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Ngraph operation Reshape with name Reshape_4219609 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif

    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

    double l1 = 0.0, lInf = 0.0;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
    {
        l1 = 5e-3;
        lInf = 2e-2;
    }
    testLayerUsingCaffeModels("layer_inner_product", true, true, l1, lInf);
}

TEST_P(Test_Caffe_layers, Pooling_max)
{
    testLayerUsingCaffeModels("layer_pooling_max");
}

TEST_P(Test_Caffe_layers, Pooling_ave)
{
    testLayerUsingCaffeModels("layer_pooling_ave");
}

TEST_P(Test_Caffe_layers, MVN)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); /* MVN is unsupported */

    testLayerUsingCaffeModels("layer_mvn");
}

void testReshape(const MatShape& inputShape, const MatShape& targetShape,
                 int axis = 0, int num_axes = -1,
                 MatShape mask = MatShape())
{
    LayerParams params;
    params.set("axis", axis);
    params.set("num_axes", num_axes);
    if (!mask.empty())
    {
        params.set("dim", DictValue::arrayInt<int*>(&mask[0], mask.size()));
    }

    Mat inp(inputShape.size(), &inputShape[0], CV_32F);
    std::vector<Mat> inpVec(1, inp);
    std::vector<Mat> outVec, intVec;

    Ptr<Layer> rl = LayerFactory::createLayerInstance("Reshape", params);
    runLayer(rl, inpVec, outVec);

    Mat& out = outVec[0];
    MatShape shape(out.size.p, out.size.p + out.dims);
    EXPECT_EQ(shape, targetShape);
}

TEST(Layer_Test_Reshape, Accuracy)
{
    {
        int inp[] = {4, 3, 1, 2};
        int out[] = {4, 3, 2};
        testReshape(MatShape(inp, inp + 4), MatShape(out, out + 3), 2, 1);
    }
    {
        int inp[] = {1, 128, 4, 4};
        int out[] = {1, 2048};
        int mask[] = {-1, 2048};
        testReshape(MatShape(inp, inp + 4), MatShape(out, out + 2), 0, -1,
                    MatShape(mask, mask + 2));
    }
    {
        int inp[] = {1, 2, 3};
        int out[] = {3, 1, 2};
        int mask[] = {3, 1, 2};
        testReshape(MatShape(inp, inp + 3), MatShape(out, out + 3), 0, -1,
                    MatShape(mask, mask + 3));
    }
}

TEST_P(Test_Caffe_layers, BatchNorm)
{
    testLayerUsingCaffeModels("layer_batch_norm", true);
    testLayerUsingCaffeModels("layer_batch_norm_local_stats", true, false);
}

TEST_P(Test_Caffe_layers, ReLU)
{
    testLayerUsingCaffeModels("layer_relu");
}

TEST_P(Test_Caffe_layers, Dropout)
{
    testLayerUsingCaffeModels("layer_dropout");
}

TEST_P(Test_Caffe_layers, Concat)
{
    if (cvtest::skipUnstableTests && (backend == DNN_BACKEND_VKCOM))
    {
        throw SkipTestException("Test_Caffe_layers.Concat test produces unstable result with Vulkan");
    }

#if defined(INF_ENGINE_RELEASE)
#if INF_ENGINE_VER_MAJOR_GE(2019010000) && INF_ENGINE_VER_MAJOR_LT(2019020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif INF_ENGINE_VER_MAJOR_EQ(2019020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 &&
        (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

#if INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH &&
        (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

#endif
    testLayerUsingCaffeModels("layer_concat");
    testLayerUsingCaffeModels("layer_concat_optim", true, false);
    testLayerUsingCaffeModels("layer_concat_shared_input", true, false);
}

TEST_P(Test_Caffe_layers, Fused_Concat)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    checkBackend();

    // Test case
    // input
    //   |
    //   v
    // some_layer
    // |   |
    // v   v
    // concat
    Net net;
    int interLayer;
    {
        LayerParams lp;
        lp.type = "AbsVal";
        lp.name = "someLayer";
        interLayer = net.addLayerToPrev(lp.name, lp.type, lp);
    }
    {
        LayerParams lp;
        lp.set("axis", 1);
        lp.type = "Concat";
        lp.name = "testConcat";
        int id = net.addLayer(lp.name, lp.type, lp);
        net.connect(interLayer, 0, id, 0);
        net.connect(interLayer, 0, id, 1);
    }
    int shape[] = {1, 2, 3, 4};
    Mat input(4, shape, CV_32F);
    randu(input, 0.0f, 1.0f);  // [0, 1] to make AbsVal an identity transformation.

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat out = net.forward();

    normAssert(slice(out, Range::all(), Range(0, 2), Range::all(), Range::all()), input, "", default_l1, default_lInf);
    normAssert(slice(out, Range::all(), Range(2, 4), Range::all(), Range::all()), input, "", default_l1, default_lInf);
}

TEST_P(Test_Caffe_layers, Eltwise)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
    testLayerUsingCaffeModels("layer_eltwise");
}

TEST_P(Test_Caffe_layers, PReLU)
{
    double lInf = (target == DNN_TARGET_MYRIAD || target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16) ? 0.021 : 0.0;
    testLayerUsingCaffeModels("layer_prelu", true, true, 0.0, lInf);
}

// TODO: fix an unstable test case
TEST_P(Test_Caffe_layers, layer_prelu_fc)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    // Reference output values are in range [-0.0001, 10.3906]
    double l1 = (target == DNN_TARGET_MYRIAD) ? 0.005 : 0.0;
    double lInf = (target == DNN_TARGET_MYRIAD) ? 0.021 : 0.0;
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
    {
        l1 = 0.006f; lInf = 0.05f;
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.01f; lInf = 0.05f;
    }
#endif
    testLayerUsingCaffeModels("layer_prelu_fc", true, false, l1, lInf);
}

TEST_P(Test_Caffe_layers, Reshape_Split_Slice)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2023000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    Net net = readNetFromCaffe(_tf("reshape_and_slice_routines.prototxt"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input(6, 12, CV_32F);
    RNG rng(0);
    rng.fill(input, RNG::UNIFORM, -1, 1);

    net.setInput(input, "input");
    Mat output = net.forward("output");

    normAssert(input, output, "", default_l1, default_lInf);
}

TEST_P(Test_Caffe_layers, Conv_Elu)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_RELEASE <= 2018050000
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    Net net = readNetFromTensorflow(_tf("layer_elu_model.pb"));
    ASSERT_FALSE(net.empty());

    Mat inp = blobFromNPY(_tf("layer_elu_in.npy"));
    Mat ref = blobFromNPY(_tf("layer_elu_out.npy"));

    net.setInput(inp, "input");
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat out = net.forward();

    double l1 = default_l1, lInf = default_lInf;
    if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.0002;
        lInf = 0.0005;
    }
    normAssert(ref, out, "", l1, lInf);
}

class Layer_LSTM_Test : public ::testing::Test
{
public:
    int numInp, numOut;
    Mat Wh, Wx, b, h, c;
    Ptr<LSTMLayer> layer;
    std::vector<Mat> inputs, outputs;

    Layer_LSTM_Test() {}

    void init(const MatShape &inpShape_, const MatShape &outShape_,
              bool produceCellOutput, bool useTimestampDim)
    {
        numInp = total(inpShape_);
        numOut = total(outShape_);

        Wh = Mat::ones(4 * numOut, numOut, CV_32F);
        Wx = Mat::ones(4 * numOut, numInp, CV_32F);
        b  = Mat::ones(4 * numOut, 1, CV_32F);
        h  = Mat::ones(4, numOut, CV_32F);
        c  = Mat::ones(4, numOut, CV_32F);

        LayerParams lp;
        lp.blobs.resize(5);
        lp.blobs[0] = Wh;
        lp.blobs[1] = Wx;
        lp.blobs[2] = b;
        lp.blobs[3] = h;
        lp.blobs[4] = c;

        lp.set<bool>("produce_cell_output", produceCellOutput);
        lp.set<bool>("use_timestamp_dim", useTimestampDim);

        layer = LSTMLayer::create(lp);
        layer->setOutShape(outShape_);
    }
};

TEST_F(Layer_LSTM_Test, get_set_test)
{
    const int TN = 4;
    MatShape inpShape = shape(5, 3, 2);
    MatShape outShape = shape(3, 1, 2);
    MatShape inpResShape = concat(shape(TN), inpShape);
    MatShape outResShape = concat(shape(TN), outShape);

    init(inpShape, outShape, true, false);
    layer->setOutShape(outShape);

    Mat C((int)outResShape.size(), &outResShape[0], CV_32F);
    randu(C, -1., 1.);
    Mat H = C.clone();
    randu(H, -1., 1.);

    Mat inp((int)inpResShape.size(), &inpResShape[0], CV_32F);
    randu(inp, -1., 1.);

    inputs.push_back(inp);
    runLayer(layer, inputs, outputs);

    EXPECT_EQ(2u, outputs.size());

    print(outResShape, "outResShape");
    print(shape(outputs[0]), "out0");
    print(shape(outputs[0]), "out1");

    EXPECT_EQ(outResShape, shape(outputs[0]));
    EXPECT_EQ(outResShape, shape(outputs[1]));

    EXPECT_EQ(0, layer->inputNameToIndex("x"));
    EXPECT_EQ(0, layer->outputNameToIndex("h"));
    EXPECT_EQ(1, layer->outputNameToIndex("c"));
}

TEST(Layer_LSTM_Test_Accuracy_with_, CaffeRecurrent)
{
    LayerParams lp;
    lp.blobs.resize(5);
    lp.blobs[0] = blobFromNPY(_tf("lstm.prototxt.w_2.npy"));  // Wh
    lp.blobs[1] = blobFromNPY(_tf("lstm.prototxt.w_0.npy"));  // Wx
    lp.blobs[2] = blobFromNPY(_tf("lstm.prototxt.w_1.npy"));  // bias
    lp.blobs[3] = Mat::zeros(2, 17, CV_32F);                     // h_0
    lp.blobs[4] = Mat::zeros(2, 17, CV_32F);                     // c_0
    Ptr<LSTMLayer> layer = LSTMLayer::create(lp);

    Mat inp = blobFromNPY(_tf("recurrent.input.npy"));
    std::vector<Mat> inputs(1, inp), outputs;
    runLayer(layer, inputs, outputs);

    Mat h_t_reference = blobFromNPY(_tf("lstm.prototxt.h_1.npy"));
    normAssert(h_t_reference, outputs[0]);
}

TEST(Layer_LSTM_Test_Accuracy_with_, HiddenParams)
{
    Mat Wx = blobFromNPY(_tf("lstm.hidden.W.npy"));
    Mat Wh = blobFromNPY(_tf("lstm.hidden.R.npy"));
    Mat b = blobFromNPY(_tf("lstm.hidden.B.npy"));
    Mat h0 = blobFromNPY(_tf("lstm.hidden.h0.npy"));
    Mat c0 = blobFromNPY(_tf("lstm.hidden.c0.npy"));

    const int numHidden = 3;
    const int numDirs = Wx.size[0];
    const int numFeatures = Wx.size[2];

    b = b.reshape(1, b.size[0]);
    Mat bx = b.colRange(0, b.cols / 2);
    Mat bh = b.colRange(b.cols / 2, b.cols);
    b = bx + bh;

    // IFGO->IGFO
    for (int k = 0; k < numDirs; ++k)
    {
        float* WxData = Wx.ptr<float>(k);
        float* WhData = Wh.ptr<float>(k);
        float* biasData = b.ptr<float>(k);
        for (int j = 0; j < numHidden; ++j)
        {
            for (int i = 0; i < numFeatures; ++i)
            {
                std::swap(WxData[(numHidden + j) * numFeatures + i],
                          WxData[(numHidden * 2 + j) * numFeatures + i]);
            }
            for (int i = 0; i < numHidden; ++i)
            {
                std::swap(WhData[(numHidden + j) * numHidden + i],
                          WhData[(numHidden * 2 + j) * numHidden + i]);
            }
            std::swap(biasData[numHidden + j], biasData[numHidden * 2 + j]);
        }
    }

    Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
    Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);
    h0 = h0.reshape(1, h0.size[0] * h0.size[1]);
    c0 = c0.reshape(1, c0.size[0] * c0.size[1]);

    LayerParams lstmParams;
    lstmParams.blobs.resize(5);
    lstmParams.blobs[0] = Wh;
    lstmParams.blobs[1] = Wx;
    lstmParams.blobs[2] = b;
    lstmParams.blobs[3] = h0;
    lstmParams.blobs[4] = c0;
    lstmParams.set("bidirectional", false);
    Ptr<LSTMLayer> layer = LSTMLayer::create(lstmParams);

    Mat inp = blobFromNPY(_tf("lstm.hidden.input.npy"));
    std::vector<Mat> inputs(1, inp), outputs;
    runLayer(layer, inputs, outputs);

    Mat h_t_reference = blobFromNPY(_tf("lstm.hidden.output.npy"));
    normAssert(h_t_reference, outputs[0]);
}

TEST(Layer_GRU_Test_Accuracy_with_, Pytorch)
{
    Mat Wx = blobFromNPY(_tf("gru.W.npy"));
    Mat Wh = blobFromNPY(_tf("gru.R.npy"));
    Mat b = blobFromNPY(_tf("gru.B.npy"));
    Mat h0 = blobFromNPY(_tf("gru.h0.npy"));

    Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
    Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);
    h0 = h0.reshape(1, h0.size[0] * h0.size[1]);
    b = b.reshape(1, b.size[0]);

    LayerParams gruParams;
    gruParams.blobs.resize(4);
    gruParams.blobs[0] = Wh;
    gruParams.blobs[1] = Wx;
    gruParams.blobs[2] = b;
    gruParams.blobs[3] = h0;
    gruParams.set("bidirectional", false);
    Ptr<GRULayer> layer = GRULayer::create(gruParams);

    Mat inp = blobFromNPY(_tf("gru.input.npy"));
    std::vector<Mat> inputs(1, inp), outputs;
    runLayer(layer, inputs, outputs);

    Mat h_t_reference = blobFromNPY(_tf("gru.output.npy"));
    normAssert(h_t_reference, outputs[0]);
}

TEST(Layer_RNN_Test_Accuracy_with_, CaffeRecurrent)
{
    Ptr<RNNLayer> layer = RNNLayer::create(LayerParams());

    layer->setWeights(
                blobFromNPY(_tf("rnn.prototxt.w_0.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_1.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_2.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_3.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_4.npy")) );

    std::vector<Mat> output, input(1, blobFromNPY(_tf("recurrent.input.npy")));
    runLayer(layer, input, output);

    Mat h_ref = blobFromNPY(_tf("rnn.prototxt.h_1.npy"));
    normAssert(h_ref, output[0]);
}

TEST(Layer_LSTM_Test_Accuracy_, Reverse)
{
    // This handcrafted setup calculates (approximately) the prefix sum of the
    // input, assuming the inputs are suitably small.
    cv::Mat input(2, 1, CV_32FC1);
    input.at<float>(0, 0) = 1e-5f;
    input.at<float>(1, 0) = 2e-5f;

    cv::Mat Wx(4, 1, CV_32FC1);
    Wx.at<float>(0, 0) = 0.f;  // Input gate
    Wx.at<float>(1, 0) = 0.f;  // Forget gate
    Wx.at<float>(2, 0) = 0.f;  // Output gate
    Wx.at<float>(3, 0) = 1.f;  // Update signal

    cv::Mat Wh(4, 1, CV_32FC1);
    Wh.at<float>(0, 0) = 0.f;  // Input gate
    Wh.at<float>(1, 0) = 0.f;  // Forget gate
    Wh.at<float>(2, 0) = 0.f;  // Output gate
    Wh.at<float>(3, 0) = 0.f;  // Update signal

    cv::Mat bias(4, 1, CV_32FC1);
    bias.at<float>(0, 0) = 1e10f;  // Input gate - always allows input to c
    bias.at<float>(1, 0) = 1e10f;  // Forget gate - never forget anything on c
    bias.at<float>(2, 0) = 1e10f;  // Output gate - always output everything
    bias.at<float>(3, 0) = 0.f;  // Update signal

    cv::Mat hInternal = cv::Mat::zeros(1, 1, CV_32FC1);
    cv::Mat cInternal = cv::Mat::zeros(1, 1, CV_32FC1);

    LayerParams lp;
    lp.set("reverse", true);
    lp.set("use_timestamp_dim", true);
    lp.blobs.clear();
    lp.blobs.push_back(Wh);
    lp.blobs.push_back(Wx);
    lp.blobs.push_back(bias);
    lp.blobs.push_back(hInternal);
    lp.blobs.push_back(cInternal);

    cv::Ptr<cv::dnn::LSTMLayer> layer = LSTMLayer::create(lp);
    std::vector<cv::Mat> outputs;
    std::vector<cv::Mat> inputs;
    inputs.push_back(input);
    runLayer(layer, inputs, outputs);

    ASSERT_EQ(1, outputs.size());
    cv::Mat out = outputs[0];
    ASSERT_EQ(3, out.dims);
    ASSERT_EQ(shape(2, 1, 1), shape(out));
    float* data = reinterpret_cast<float*>(out.data);
    EXPECT_NEAR(std::tanh(1e-5f) + std::tanh(2e-5f), data[0], 1e-10);
    EXPECT_NEAR(std::tanh(2e-5f), data[1], 1e-10);
}


class Layer_RNN_Test : public ::testing::Test
{
public:
    int nX, nH, nO, nT, nS;
    Mat Whh, Wxh, bh, Who, bo;
    Ptr<RNNLayer> layer;

    std::vector<Mat> inputs, outputs;

    Layer_RNN_Test()
    {
        nT = 3;
        nS = 5;
        nX = 31;
        nH = 64;
        nO = 100;

        Whh = Mat::ones(nH, nH, CV_32F);
        Wxh = Mat::ones(nH, nX, CV_32F);
        bh  = Mat::ones(nH, 1, CV_32F);
        Who = Mat::ones(nO, nH, CV_32F);
        bo  = Mat::ones(nO, 1, CV_32F);

        layer = RNNLayer::create(LayerParams());
        layer->setProduceHiddenOutput(true);
        layer->setWeights(Wxh, bh, Whh, Who, bo);
    }
};

TEST_F(Layer_RNN_Test, get_set_test)
{
    int sz[] = { nT, nS, 1, nX };
    Mat inp(4, sz, CV_32F);
    randu(inp, -1., 1.);
    inputs.push_back(inp);
    runLayer(layer, inputs, outputs);

    EXPECT_EQ(outputs.size(), 2u);
    EXPECT_EQ(shape(outputs[0]), shape(nT, nS, nO));
    EXPECT_EQ(shape(outputs[1]), shape(nT, nS, nH));
}

TEST_P(Test_Caffe_layers, Accum)
{
#ifdef OPENCV_DNN_EXTERNAL_PROTOBUF
    throw SkipTestException("Requires patched protobuf");
#else
    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL, CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    testLayerUsingCaffeModels("accum", false, false, 0.0, 0.0, 2);
    testLayerUsingCaffeModels("accum_ref", false, false, 0.0, 0.0, 2);
#endif
}

TEST_P(Test_Caffe_layers, FlowWarp)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    testLayerUsingCaffeModels("flow_warp", false, false, 0.0, 0.0, 2);
}

TEST_P(Test_Caffe_layers, ChannelNorm)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    testLayerUsingCaffeModels("channel_norm", false, false);
}

TEST_P(Test_Caffe_layers, DataAugmentation)
{
#ifdef OPENCV_DNN_EXTERNAL_PROTOBUF
    throw SkipTestException("Requires patched protobuf");
#else
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    testLayerUsingCaffeModels("data_augmentation", true, false);
    testLayerUsingCaffeModels("data_augmentation_2x1", true, false);
    testLayerUsingCaffeModels("data_augmentation_8x6", true, false);
#endif
}

TEST_P(Test_Caffe_layers, Resample)
{
#ifdef OPENCV_DNN_EXTERNAL_PROTOBUF
    throw SkipTestException("Requires patched protobuf");
#else
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2023000000)
    if (backend != DNN_BACKEND_OPENCV)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testLayerUsingCaffeModels("nearest_2inps", false, false, 0.0, 0.0, 2);
    testLayerUsingCaffeModels("nearest", false, false);
#endif
}

TEST_P(Test_Caffe_layers, Correlation)
{
#ifdef OPENCV_DNN_EXTERNAL_PROTOBUF
    throw SkipTestException("Requires patched protobuf");
#else
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER,
                     CV_TEST_TAG_DNN_SKIP_OPENCL, CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    testLayerUsingCaffeModels("correlation", false, false, 0.0, 0.0, 2);
#endif
}

TEST_P(Test_Caffe_layers, Convolution2Inputs)
{
    testLayerUsingCaffeModels("conv_2_inps", true, false, 0.0, 0.0, 2);
}

TEST_P(Test_Caffe_layers, ROIPooling_Accuracy)
{
    Net net = readNetFromCaffe(_tf("net_roi_pooling.prototxt"));
    ASSERT_FALSE(net.empty());

    Mat inp = blobFromNPY(_tf("net_roi_pooling.input.npy"));
    Mat rois = blobFromNPY(_tf("net_roi_pooling.rois.npy"));
    Mat ref = blobFromNPY(_tf("net_roi_pooling.npy"));

    checkBackend(&inp, &ref);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.setInput(inp, "input");
    net.setInput(rois, "rois");

    Mat out = net.forward();

    double l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1e-3 : 1e-5;
    double lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1e-3 : 1e-4;
    if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 2e-4;
        lInf = 9e-4;
    }
    normAssert(out, ref, "", l1, lInf);
}

TEST_P(Test_Caffe_layers, FasterRCNN_Proposal)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); /* Proposal layer is unsupported */

    Net net = readNetFromCaffe(_tf("net_faster_rcnn_proposal.prototxt"));

    Mat scores = blobFromNPY(_tf("net_faster_rcnn_proposal.scores.npy"));
    Mat deltas = blobFromNPY(_tf("net_faster_rcnn_proposal.deltas.npy"));
    Mat imInfo = (Mat_<float>(1, 3) << 600, 800, 1.6f);

    net.setInput(scores, "rpn_cls_prob_reshape");
    net.setInput(deltas, "rpn_bbox_pred");
    net.setInput(imInfo, "im_info");

    std::vector<Mat> outs;
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    net.forward(outs, "output");

    for (int i = 0; i < 2; ++i)
    {
        Mat ref = blobFromNPY(_tf(i == 0 ? "net_faster_rcnn_proposal.out_rois.npy" :
                                           "net_faster_rcnn_proposal.out_scores.npy"));
        const int numDets = ref.size[0];
        EXPECT_LE(numDets, outs[i].size[0]);
        normAssert(outs[i].rowRange(0, numDets), ref);

        if (numDets < outs[i].size[0])
        {
            EXPECT_EQ(countNonZero(outs[i].rowRange(numDets, outs[i].size[0])), 0);
        }
    }
}

typedef testing::TestWithParam<tuple<Vec4i, Vec2i, bool> > Scale_untrainable;
TEST_P(Scale_untrainable, Accuracy)
{
    Vec4i inpShapeVec = get<0>(GetParam());
    int axis = get<1>(GetParam())[0];
    int weightsDims = get<1>(GetParam())[1];
    bool testFusion = get<2>(GetParam());
    const int inpShape[] = {inpShapeVec[0], inpShapeVec[1], inpShapeVec[2], inpShapeVec[3]};

    // Create a network with two inputs. Scale layer multiplies a first input to
    // a second one. See http://caffe.berkeleyvision.org/tutorial/layers/scale.html
    Net net;
    // Check that this version of Scale layer won't be fused with Convolution layer.
    if (testFusion)
    {
        LayerParams lp;
        lp.set("kernel_size", 1);
        lp.set("num_output", 3);
        lp.set("group", 3);
        lp.set("bias_term", false);
        lp.type = "Convolution";
        lp.name = "testConv";

        std::vector<int> weightsShape(4);
        weightsShape[0] = 3;  // #outChannels
        weightsShape[1] = 1;  // #inpChannels / group
        weightsShape[2] = 1;  // height
        weightsShape[3] = 1;  // width
        Mat weights(weightsShape, CV_32F);
        weights.setTo(1);
        lp.blobs.push_back(weights);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }
    LayerParams lp;
    lp.type = "Scale";
    lp.name = "testLayer";
    lp.set("axis", axis);
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    Mat input(4, inpShape, CV_32F);
    Mat weights(weightsDims, &inpShape[axis], CV_32F);
    randu(input, -1, 1);
    randu(weights, -1, 1);

    std::vector<String> inpNames(2);
    inpNames[0] = "scale_input";
    inpNames[1] = "scale_weights";
    net.setInputsNames(inpNames);
    net.setInput(input, inpNames[0]);
    net.setInput(weights, inpNames[1]);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    Mat out = net.forward();

    Mat ref(input.dims, input.size, CV_32F);
    float* inpData = (float*)input.data;
    float* refData = (float*)ref.data;
    float* weightsData = (float*)weights.data;
    int spatialSize = 1;
    for (int i = axis + weightsDims; i < 4; ++i)
        spatialSize *= inpShape[i];
    for (int i = 0; i < ref.total(); ++i)
    {
        float w = weightsData[(i / spatialSize) % weights.total()];
        refData[i] = inpData[i] * w;
    }
    normAssert(out, ref);
}

INSTANTIATE_TEST_CASE_P(Layer_Test, Scale_untrainable, Combine(
/*input size*/   Values(Vec4i(2, 3, 4, 5)),
/*axis, #dims*/  Values(Vec2i(0, 1), Vec2i(0, 2), Vec2i(0, 3), Vec2i(0, 4),
                                     Vec2i(1, 1), Vec2i(1, 2), Vec2i(1, 3),
                                                  Vec2i(2, 1), Vec2i(2, 2),
                                                               Vec2i(3, 1)),
/*conv fusion*/  testing::Bool()
));

typedef testing::TestWithParam<tuple<Vec4i, Vec4i, int, int, int> > Crop;
TEST_P(Crop, Accuracy)
{
    Vec4i inpShapeVec = get<0>(GetParam());
    Vec4i sizShapeVec = get<1>(GetParam());
    int axis = get<2>(GetParam());
    int numOffsets = get<3>(GetParam());
    int offsetVal = get<4>(GetParam());
    const int inpShape[] = {inpShapeVec[0], inpShapeVec[1], inpShapeVec[2], inpShapeVec[3]};
    const int sizShape[] = {sizShapeVec[0], sizShapeVec[1], sizShapeVec[2], sizShapeVec[3]};

    // Create a network with two inputs. Crop layer crops a first input to
    // the size of a second one.
    // See http://caffe.berkeleyvision.org/tutorial/layers/crop.html
    Net net;

    LayerParams lp;
    lp.name = "testCrop";
    lp.type = "Crop";
    lp.set("axis", axis);
    if (numOffsets > 0)
    {
        std::vector<int> offsets(numOffsets, offsetVal);
        lp.set("offset", DictValue::arrayInt<int*>(&offsets[0], offsets.size()));
    }
    else
        offsetVal = 0;
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    Mat inpImage(4, inpShape, CV_32F);
    Mat sizImage(4, sizShape, CV_32F);
    randu(inpImage, -1, 1);
    randu(sizImage, -1, 1);

    std::vector<String> inpNames(2);
    inpNames[0] = "cropImage";
    inpNames[1] = "sizImage";
    net.setInputsNames(inpNames);
    net.setInput(inpImage, inpNames[0]);
    net.setInput(sizImage, inpNames[1]);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    // There are a few conditions that represent invalid input to the crop
    // layer, so in those cases we want to verify an exception is thrown.

    bool shouldThrowException = false;
    if (numOffsets > 1 && numOffsets != 4 - axis)
        shouldThrowException = true;
    else
        for (int i = axis; i < 4; i++)
            if (sizShape[i] + offsetVal > inpShape[i])
                shouldThrowException = true;

    Mat out;
    if (shouldThrowException)
    {
        ASSERT_ANY_THROW(out = net.forward());
        return;
    }
    else
        out = net.forward();

    // Finally, compare the cropped output blob from the DNN layer (out)
    // to a reference blob (ref) that we compute here.

    std::vector<Range> crop_range;
    crop_range.resize(4, Range::all());
    for (int i = axis; i < 4; i++)
        crop_range[i] = Range(offsetVal, sizShape[i] + offsetVal);

    Mat ref(sizImage.dims, sizImage.size, CV_32F);
    inpImage(&crop_range[0]).copyTo(ref);
    normAssert(out, ref);
}

INSTANTIATE_TEST_CASE_P(Layer_Test, Crop, Combine(
/*input blob shape*/    Values(Vec4i(1, 3, 20, 30)),
/*cropsize blob shape*/ Values(Vec4i(1, 3, 10, 12)),
/*start axis*/          Values(0, 1, 2),
/*number of offsets*/   Values(0, 1, 2, 4),
/*offset value*/        Values(3, 4)
));

// Check that by default average pooling layer should not count zero padded values
// into the normalization area.
TEST_P(Test_Caffe_layers, Average_pooling_kernel_area)
{
    LayerParams lp;
    lp.name = "testAvePool";
    lp.type = "Pooling";
    lp.set("kernel_size", 2);
    lp.set("stride", 2);
    lp.set("pool", "AVE");

    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    // 1 2 | 3
    // 4 5 | 6
    // ----+--
    // 7 8 | 9
    Mat inp = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    Mat ref = (Mat_<float>(2, 2) << (1 + 2 + 4 + 5) / 4.f, (3 + 6) / 2.f, (7 + 8) / 2.f, 9);
    Mat tmp = blobFromImage(inp);
    net.setInput(blobFromImage(inp));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat out = net.forward();
    normAssert(out, blobFromImage(ref));
}

TEST_P(Test_Caffe_layers, PriorBox_repeated)
{
    Net net = readNet(_tf("prior_box.prototxt"));
    int inp_size[] = {1, 3, 10, 10};
    int shape_size[] = {1, 2, 3, 4};
    Mat inp(4, inp_size, CV_32F);
    randu(inp, -1.0f, 1.0f);
    Mat shape(4, shape_size, CV_32F);
    randu(shape, -1.0f, 1.0f);
    net.setInput(inp, "data");
    net.setInput(shape, "shape");
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat out = net.forward();
    Mat ref = blobFromNPY(_tf("priorbox_output.npy"));

    double l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1e-3 : 1e-5;
    double lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1e-3 : 1e-4;
    if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 7e-5;
        lInf = 0.0005;
    }
    normAssert(out, ref, "", l1, lInf);
}

// Test PriorBoxLayer in case of no aspect ratios (just squared proposals).
TEST_P(Test_Caffe_layers, PriorBox_squares)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    LayerParams lp;
    lp.name = "testPriorBox";
    lp.type = "PriorBox";
    lp.set("min_size", 2);
    lp.set("flip", true);
    lp.set("clip", true);
    float variance[] = {0.1f, 0.1f, 0.2f, 0.2f};
    float aspectRatios[] = {1.0f};  // That should be ignored.
    lp.set("variance", DictValue::arrayReal<float*>(&variance[0], 4));
    lp.set("aspect_ratio", DictValue::arrayReal<float*>(&aspectRatios[0], 1));

    Net net;
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 0, id, 1);  // The second input is an input image. Shapes are used for boxes normalization.
    Mat inp(1, 2, CV_32F);
    randu(inp, -1, 1);
    net.setInput(blobFromImage(inp));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat out = net.forward();

    Mat ref = (Mat_<float>(4, 4) << 0.0, 0.0, 0.75, 1.0,
                                       0.25, 0.0, 1.0, 1.0,
                                       0.1f, 0.1f, 0.2f, 0.2f,
                                       0.1f, 0.1f, 0.2f, 0.2f);
    double l1 = 1e-5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16)
        l1 = 2e-5;
    normAssert(out.reshape(1, 4), ref, "", l1);
}

typedef TestWithParam<tuple<int, int> > Layer_Test_DWconv_Prelu;
TEST_P(Layer_Test_DWconv_Prelu, Accuracy)
{
    // Test case
    // input       img size 3x16x16  value all 1
    //   |
    //   v
    // dw_conv     weight[0]=-1 weight[1]=-2 weight[2]=-3   bias={1,2,3}
    //   |
    //   v
    // prelu       weight={1,2,3}
    //   |
    //   v
    // output      out size 3x14x14  if right: out[0]=-8 out[0]=-32 out[0]=-72
    //             but current opencv output: out[0]=-24 out[0]=-48 out[0]=-72

    const int num_input = get<0>(GetParam());   //inpChannels
    const int group = 3;                        //outChannels=group when group>1
    const int num_output = get<1>(GetParam());
    const int kernel_depth = num_input/group;
    CV_Assert_N(num_output >= group, num_output % group == 0, num_input % group == 0);

    Net net;
    //layer 1: dwconv
    LayerParams lp;
    lp.name = "dwconv";
    lp.type = "Convolution";
    lp.set("kernel_size", 3);
    lp.set("num_output", num_output);
    lp.set("pad", 0);
    lp.set("group", group);
    lp.set("stride", 1);
    lp.set("engine", "CAFFE");
    lp.set("bias_term", "true");

    std::vector<int> weightsShape(4);
    weightsShape[0] = num_output;   // #outChannels
    weightsShape[1] = kernel_depth; // #inpChannels / group
    weightsShape[2] = 3;            // height
    weightsShape[3] = 3;            // width
    Mat weights(weightsShape, CV_32F, Scalar(1));

    //assign weights
    for (int i = 0; i < weightsShape[0]; ++i)
    {
        for (int j = 0; j < weightsShape[1]; ++j)
        {
            for (int k = 0; k < weightsShape[2]; ++k)
            {
                for (int l = 0; l < weightsShape[3]; ++l)
                {
                    weights.ptr<float>(i, j, k)[l]=-1*(i+1);
                }
            }
        }
    }
    lp.blobs.push_back(weights);

    //assign bias
    Mat bias(1, num_output, CV_32F, Scalar(1));
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < num_output; ++j)
        {
            bias.ptr<float>(i)[j]=j+1;
        }
    }
    lp.blobs.push_back(bias);
    net.addLayerToPrev(lp.name, lp.type, lp);

    //layer 2: prelu
    LayerParams lpr;
    lpr.name = "dw_relu";
    lpr.type = "PReLU";
    Mat weightsp(1, num_output, CV_32F, Scalar(1));

    //assign weights
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < num_output; ++j)
        {
            weightsp.ptr<float>(i)[j]=j+1;
        }
    }

    lpr.blobs.push_back(weightsp);
    net.addLayerToPrev(lpr.name, lpr.type, lpr);

    int shape[] = {1, num_input, 16, 16};
    Mat in_blob(4, &shape[0], CV_32FC1, Scalar(1));

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.enableWinograd(false);
    net.setInput(in_blob);
    Mat out = net.forward();

    //assign target
    std::vector<int> outShape(4);
    outShape[0] = 1;
    outShape[1] = num_output;       // outChannels
    outShape[2] = 14;          // height
    outShape[3] = 14;          // width
    Mat target(outShape, CV_32F, Scalar(1));
    for (int i = 0; i < outShape[0]; ++i)
    {
        for (int j = 0; j < outShape[1]; ++j)
        {
            for (int k = 0; k < outShape[2]; ++k)
            {
                for (int l = 0; l < outShape[3]; ++l)
                {
                    target.ptr<float>(i, j, k)[l]=(-9*kernel_depth*(j+1)+j+1)*(j+1);
                }
            }
        }
    }

    normAssert(out, target);
}
INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_DWconv_Prelu, Combine(Values(3, 6), Values(3, 6)));

#ifdef HAVE_INF_ENGINE
// Using Intel's Model Optimizer generate .xml and .bin files:
// ./ModelOptimizer -w /path/to/caffemodel -d /path/to/prototxt \
//                  -p FP32 -i -b ${batch_size} -o /path/to/output/folder
typedef testing::TestWithParam<tuple<Backend, Target> > Layer_Test_Convolution_DLDT;
TEST_P(Layer_Test_Convolution_DLDT, Accuracy)
{
    const Backend backendId = get<0>(GetParam());
    const Target targetId = get<1>(GetParam());

    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && targetId == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    if (backendId != DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && backendId != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        throw SkipTestException("No support for async forward");

    ASSERT_EQ(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, backendId);

    Net netDefault = readNet(_tf("layer_convolution.caffemodel"), _tf("layer_convolution.prototxt"));
    Net net = readNet(_tf("layer_convolution.xml"), _tf("layer_convolution.bin"));

    Mat inp = blobFromNPY(_tf("blob.npy"));

    netDefault.setInput(inp);
    netDefault.setPreferableBackend(DNN_BACKEND_OPENCV);
    Mat outDefault = netDefault.forward();

    net.setInput(inp);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    Mat out = net.forward();

    double l1 = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 1.5e-3 : 1e-5;
    double lInf = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 1.8e-2 : 1e-4;
    normAssert(outDefault, out, "", l1, lInf);

    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    ASSERT_EQ(net.getLayer(outLayers[0])->name, "output");
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        ASSERT_EQ(net.getLayer(outLayers[0])->type, "Convolution");
    else
        ASSERT_EQ(net.getLayer(outLayers[0])->type, "Result");
}

TEST_P(Layer_Test_Convolution_DLDT, setInput_uint8)
{
    const Backend backendId = get<0>(GetParam());
    const Target targetId = get<1>(GetParam());

    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && targetId == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    if (backendId != DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && backendId != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        throw SkipTestException("No support for async forward");

    ASSERT_EQ(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, backendId);

    int blobSize[] = {2, 6, 75, 113};
    Mat inputs[] = {Mat(4, &blobSize[0], CV_8U), Mat()};

    randu(inputs[0], 0, 255);
    inputs[0].convertTo(inputs[1], CV_32F);

    Mat outs[2];
    for (int i = 0; i < 2; ++i)
    {
        Net net = readNet(_tf("layer_convolution.xml"), _tf("layer_convolution.bin"));
        net.setPreferableBackend(backendId);
        net.setPreferableTarget(targetId);
        net.setInput(inputs[i]);
        outs[i] = net.forward();
        ASSERT_EQ(outs[i].type(), CV_32F);
    }
    if (targetId != DNN_TARGET_MYRIAD)
        normAssert(outs[0], outs[1]);
}

TEST_P(Layer_Test_Convolution_DLDT, multithreading)
{
    const Backend backendId = get<0>(GetParam());
    const Target targetId = get<1>(GetParam());

    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && targetId == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    if (backendId != DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && backendId != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        throw SkipTestException("No support for async forward");

    ASSERT_EQ(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, backendId);

    std::string xmlPath = _tf("layer_convolution.xml");
    std::string binPath = _tf("layer_convolution.bin");
    Net firstNet = readNet(xmlPath, binPath);
    Net secondNet = readNet(xmlPath, binPath);
    Mat inp = blobFromNPY(_tf("blob.npy"));

    firstNet.setInput(inp);
    secondNet.setInput(inp);
    firstNet.setPreferableBackend(backendId);
    firstNet.setPreferableTarget(targetId);
    secondNet.setPreferableBackend(backendId);
    secondNet.setPreferableTarget(targetId);

    Mat out1, out2;
    std::thread t1([&]{out1 = firstNet.forward();});
    std::thread t2([&]{out2 = secondNet.forward();});

    t1.join();
    t2.join();

    Mat ref = blobFromNPY(_tf("layer_convolution.npy"));
    double l1 = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 1.5e-3 : 1e-5;
    double lInf = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 1.8e-2 : 1e-4;
    normAssert(out1, ref, "first thread", l1, lInf);
    normAssert(out2, ref, "second thread", l1, lInf);
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_Convolution_DLDT,
    dnnBackendsAndTargetsIE()
);

// 1. Create a .prototxt file with the following network:
// layer {
//   type: "Input" name: "data" top: "data"
//   input_param { shape { dim: 1 dim: 2 dim: 3 } }
// }
// layer {
//   type: "Input" name: "second_input" top: "second_input"
//   input_param { shape { dim: 1 dim: 2 dim: 3 } }
// }
// layer {
//  type: "Eltwise" name: "output" top: "output"
//  bottom: "data" bottom: "second_input"
//  eltwise_param { operation: SUM }
// }
//
// 2. Create a .caffemodel file using Caffe:
//
// import caffe
// net = caffe.Net('/path/to/prototxt', caffe.TEST)
// net.save('/path/to/caffemodel')
//
// 3. Convert using ModelOptimizer.
typedef testing::TestWithParam<tuple<int, int, Target, std::vector<int> > > Test_DLDT_two_inputs_3dim;
TEST_P(Test_DLDT_two_inputs_3dim, as_IR)
{
    int firstInpType = get<0>(GetParam());
    int secondInpType = get<1>(GetParam());
    Target targetId = get<2>(GetParam());

    Net net = readNet(_tf("net_two_inputs.xml"), _tf("net_two_inputs.bin"));
    std::vector<int> inpSize = get<3>(GetParam());
    Mat firstInp(3, inpSize.data(), firstInpType);
    Mat secondInp(3, inpSize.data(), secondInpType);
    randu(firstInp, 0, 255);
    randu(secondInp, 0, 255);

    net.setInput(firstInp, "data");
    net.setInput(secondInp, "second_input");
    net.setPreferableTarget(targetId);

    double l1 = ((targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) &&
                 (firstInpType == CV_32F || secondInpType == CV_32F)) ? 0.06 : 0.0;
    double lInf = ((targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) &&
                   (firstInpType == CV_32F || secondInpType == CV_32F)) ? 0.23 : 0.0;

    Mat out = net.forward();

    Mat ref;
    cv::add(firstInp, secondInp, ref, Mat(), CV_32F);
    normAssert(out, ref, "", l1, lInf);
}

std::vector< std::vector<int> > list_sizes{ {1, 2, 3}, {3, 2, 1}, {5, 5, 5}, {13, 7, 11} };

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_DLDT_two_inputs_3dim, Combine(
  Values(CV_8U, CV_32F), Values(CV_8U, CV_32F),
  testing::ValuesIn(getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)),
  testing::ValuesIn(list_sizes)
));

class UnsupportedLayer : public Layer
{
public:
    UnsupportedLayer(const LayerParams &params) : Layer(params) {}

    static Ptr<Layer> create(const LayerParams& params)
    {
        return Ptr<Layer>(new UnsupportedLayer(params));
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void forward(cv::InputArrayOfArrays inputs, cv::OutputArrayOfArrays outputs, cv::OutputArrayOfArrays internals) CV_OVERRIDE {}
};

typedef DNNTestLayer Test_DLDT_layers;

static void test_dldt_fused_output(Backend backend, Target target)
{
    static const int kNumChannels = 3;
    Net net;
    {
        LayerParams lp;
        lp.set("kernel_size", 1);
        lp.set("num_output", 3);
        lp.set("bias_term", false);
        lp.type = "Convolution";
        lp.name = "testConv";
        lp.blobs.push_back(Mat({kNumChannels, 1, 1, 1}, CV_32F, Scalar(1)));
        net.addLayerToPrev(lp.name, lp.type, lp);
    }
    {
        LayerParams lp;
        lp.set("bias_term", false);
        lp.type = "Scale";
        lp.name = "testScale";
        lp.blobs.push_back(Mat({kNumChannels}, CV_32F, Scalar(1)));
        net.addLayerToPrev(lp.name, lp.type, lp);
    }
    {
        LayerParams lp;
        net.addLayerToPrev("unsupported_layer", "Unsupported", lp);
    }
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    net.setInput(Mat({1, 1, 2, 3}, CV_32FC1, Scalar(1)));
    net.forward();
}

TEST_P(Test_DLDT_layers, fused_output)
{
    CV_DNN_REGISTER_LAYER_CLASS(Unsupported, UnsupportedLayer);
    try
    {
        test_dldt_fused_output(backend, target);
    }
    catch (const std::exception& e)
    {
        ADD_FAILURE() << "Exception: " << e.what();
    }
    catch(...)
    {
        ADD_FAILURE() << "Unknown exception";
    }
    LayerFactory::unregisterLayer("Unsupported");
}

TEST_P(Test_DLDT_layers, multiple_networks)
{
    Net nets[2];
    for (int i = 0; i < 2; ++i)
    {
        nets[i].setInputsNames(std::vector<String>(1, format("input_%d", i)));

        LayerParams lp;
        lp.set("kernel_size", 1);
        lp.set("num_output", 1);
        lp.set("bias_term", false);
        lp.type = "Convolution";
        lp.name = format("testConv_%d", i);
        lp.blobs.push_back(Mat({1, 1, 1, 1}, CV_32F, Scalar(1 + i)));
        nets[i].addLayerToPrev(lp.name, lp.type, lp);
        nets[i].setPreferableBackend(backend);
        nets[i].setPreferableTarget(target);
        nets[i].setInput(Mat({1, 1, 2, 3}, CV_32FC1, Scalar(1)));
    }
    Mat out_1 = nets[0].forward();
    Mat out_2 = nets[1].forward();
    // After the second model is initialized we try to receive an output from the first network again.
    out_1 = nets[0].forward();
    normAssert(2 * out_1, out_2);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_DLDT_layers, dnnBackendsAndTargets());

#endif  // HAVE_INF_ENGINE

// Test a custom layer.
class CustomInterpLayer CV_FINAL : public Layer
{
public:
    CustomInterpLayer(const LayerParams &params) : Layer(params)
    {
        zoomFactor = params.get<int>("zoom_factor", 0);
        outWidth = params.get<int>("width", 0);
        outHeight = params.get<int>("height", 0);
    }

    static Ptr<Layer> create(LayerParams& params)
    {
        return Ptr<Layer>(new CustomInterpLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &internals) const CV_OVERRIDE
    {
        const int batchSize = inputs[0][0];
        const int numChannels = inputs[0][1];
        const int inpHeight = inputs[0][2];
        const int inpWidth = inputs[0][3];

        std::vector<int> outShape(4);
        outShape[0] = batchSize;
        outShape[1] = numChannels;
        outShape[2] = outHeight != 0 ? outHeight : (inpHeight + (inpHeight - 1) * (zoomFactor - 1));
        outShape[3] = outWidth != 0 ? outWidth : (inpWidth + (inpWidth - 1) * (zoomFactor - 1));
        outputs.assign(1, outShape);
        return false;
    }

    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);

        if (!outWidth && !outHeight)
        {
            outHeight = outputs[0].size[2];
            outWidth = outputs[0].size[3];
        }
    }

    // Implementation of this custom layer is based on https://github.com/cdmh/deeplab-public/blob/master/src/caffe/layers/interp_layer.cpp
    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        Mat& inp = inputs[0];
        Mat& out = outputs[0];
        const float* inpData = (float*)inp.data;
        float* outData = (float*)out.data;

        const int batchSize = inp.size[0];
        const int numChannels = inp.size[1];
        const int inpHeight = inp.size[2];
        const int inpWidth = inp.size[3];

        const float rheight = (outHeight > 1) ? static_cast<float>(inpHeight - 1) / (outHeight - 1) : 0.f;
        const float rwidth = (outWidth > 1) ? static_cast<float>(inpWidth - 1) / (outWidth - 1) : 0.f;
        for (int h2 = 0; h2 < outHeight; ++h2)
        {
            const float h1r = rheight * h2;
            const int h1 = h1r;
            const int h1p = (h1 < inpHeight - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = 1.f - h1lambda;
            for (int w2 = 0; w2 < outWidth; ++w2)
            {
                const float w1r = rwidth * w2;
                const int w1 = w1r;
                const int w1p = (w1 < inpWidth - 1) ? 1 : 0;
                const float w1lambda = w1r - w1;
                const float w0lambda = 1.f - w1lambda;
                const float* pos1 = inpData + h1 * inpWidth + w1;
                float* pos2 = outData + h2 * outWidth + w2;
                for (int c = 0; c < batchSize * numChannels; ++c)
                {
                    pos2[0] =
                      h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                      h1lambda * (w0lambda * pos1[h1p * inpWidth] + w1lambda * pos1[h1p * inpWidth + w1p]);
                    pos1 += inpWidth * inpHeight;
                    pos2 += outWidth * outHeight;
                }
            }
        }
    }

private:
    int outWidth, outHeight, zoomFactor;
};

TEST_P(Test_Caffe_layers, Interp)
{
#ifdef OPENCV_DNN_EXTERNAL_PROTOBUF
    throw SkipTestException("Requires patched protobuf");
#else
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021030000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
#endif

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);

    // Test a custom layer.
    CV_DNN_REGISTER_LAYER_CLASS(Interp, CustomInterpLayer);
    try
    {
        testLayerUsingCaffeModels("layer_interp", false, false);
    }
    catch (...)
    {
        LayerFactory::unregisterLayer("Interp");
        throw;
    }
    LayerFactory::unregisterLayer("Interp");

    // Test an implemented layer.
    testLayerUsingCaffeModels("layer_interp", false, false);
#endif
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_Caffe_layers, dnnBackendsAndTargets());

TEST(Layer_Test_PoolingIndices, Accuracy)
{
    Net net;

    LayerParams lp;
    lp.set("pool", "max");
    lp.set("kernel_w", 2);
    lp.set("kernel_h", 2);
    lp.set("stride_w", 2);
    lp.set("stride_h", 2);
    lp.set("pad_w", 0);
    lp.set("pad_h", 0);
    lp.name = "testLayer.name";  // This test also checks that OpenCV lets use names with dots.
    lp.type = "Pooling";
    net.addLayerToPrev(lp.name, lp.type, lp);

    Mat inp(10, 10, CV_8U);
    randu(inp, 0, 255);

    Mat maxValues(5, 5, CV_32F, Scalar(-1)), indices(5, 5, CV_32F, Scalar(-1));
    for (int y = 0; y < 10; ++y)
    {
        int dstY = y / 2;
        for (int x = 0; x < 10; ++x)
        {
            int dstX = x / 2;
            uint8_t val = inp.at<uint8_t>(y, x);
            if ((float)inp.at<uint8_t>(y, x) > maxValues.at<float>(dstY, dstX))
            {
                maxValues.at<float>(dstY, dstX) = val;
                indices.at<float>(dstY, dstX) = y * 10 + x;
            }
        }
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setInput(blobFromImage(inp));

    std::vector<Mat> outputs;
    net.forward(outputs, lp.name);
    normAssert(maxValues, outputs[0].reshape(1, 5));
    normAssert(indices, outputs[1].reshape(1, 5));
}

typedef testing::TestWithParam<tuple<Vec4i, int, tuple<Backend, Target> > > Layer_Test_ShuffleChannel;
TEST_P(Layer_Test_ShuffleChannel, Accuracy)
{
    Vec4i inpShapeVec = get<0>(GetParam());
    int group = get<1>(GetParam());
    ASSERT_EQ(inpShapeVec[1] % group, 0);
    const int groupSize = inpShapeVec[1] / group;
    int backendId = get<0>(get<2>(GetParam()));
    int targetId = get<1>(get<2>(GetParam()));

    Net net;
    LayerParams lp;
    lp.set("group", group);
    lp.type = "ShuffleChannel";
    lp.name = "testLayer";
    net.addLayerToPrev(lp.name, lp.type, lp);

    const int inpShape[] = {inpShapeVec[0], inpShapeVec[1], inpShapeVec[2], inpShapeVec[3]};
    Mat inp(4, inpShape, CV_32F);
    randu(inp, 0, 255);

    net.setInput(inp);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    Mat out = net.forward();

    double l1 = 1e-5, lInf = 1e-4;
    if (targetId == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 5e-2;
        lInf = 7e-2;
    }
    else if (targetId == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.06;
        lInf = 0.07;
    }
    for (int n = 0; n < inpShapeVec[0]; ++n)
    {
        for (int c = 0; c < inpShapeVec[1]; ++c)
        {
            Mat outChannel = getPlane(out, n, c);
            Mat inpChannel = getPlane(inp, n, groupSize * (c % group) + c / group);
            normAssert(outChannel, inpChannel, "", l1, lInf);
        }
    }
}
INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_ShuffleChannel, Combine(
/*input shape*/  Values(Vec4i(1, 6, 5, 7), Vec4i(3, 12, 1, 4)),
/*group*/        Values(1, 2, 3, 6), dnnBackendsAndTargets(/*with IE*/ false)
));

TEST(Layer_Test_ReduceMean, accuracy_input_0)
{
    vector<int> szData = { 2, 1, 2, 1 ,2 };
    std::vector<float> initData = { 0, 1, 2, 3, 4, 5, 6, 7 };
    Mat inpInitA(szData, CV_32FC1, Mat(initData).data);
    std::vector<float> resAxes0 = { 2, 3, 4, 5 };
    std::vector<float> resAxes1 = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::vector<float> resAxes2 = { 1, 2, 5, 6 };
    std::vector<float> resAxes3 = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::vector<float> resAxes4 = { 0.5, 2.5, 4.5, 6.5 };
    std::vector < vector<float>> resReduceMean = { resAxes0, resAxes1, resAxes2, resAxes3, resAxes4 };


    for (int i = 0; i < resReduceMean.size(); i++)
    {
        Net net;
        LayerParams lp;
        lp.set("keepdims", 0);
        lp.type = "Reduce";
        lp.set("reduce", "MEAN");
        lp.name = "testReduceMean";
        lp.set("axes", i);
        lp.blobs.push_back(inpInitA);

        net.addLayerToPrev(lp.name, lp.type, lp);
        net.setInput(inpInitA);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);

        Mat output = net.forward();
        MatShape gt_shape;
        for (int j = 0; j < szData.size(); j++)
        {
            if (i == j) continue;
            gt_shape.push_back(szData[j]);
        }

        EXPECT_EQ(gt_shape, shape(output));

        Mat a = output.reshape(1, output.total());
        normAssert(a, Mat(resReduceMean[i]));
    }
}


// Check if relu is not fused to convolution if we requested it's output
TEST(Layer_Test_Convolution, relu_fusion)
{
    Net net;
    {
        LayerParams lp;
        lp.set("kernel_size", 1);
        lp.set("num_output", 1);
        lp.set("bias_term", false);
        lp.type = "Convolution";
        lp.name = "testConv";

        int weightsShape[] = {1, 1, 1, 1};
        Mat weights(4, &weightsShape[0], CV_32F, Scalar(1));
        lp.blobs.push_back(weights);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }
    {
        LayerParams lp;
        lp.type = "ReLU";
        lp.name = "testReLU";
        net.addLayerToPrev(lp.name, lp.type, lp);
    }
    int sz[] = {1, 1, 2, 3};
    Mat input(4, &sz[0], CV_32F);
    randu(input, -1.0, -0.1);
    net.setInput(input);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    Mat output = net.forward("testConv");
    normAssert(input, output);
}

typedef testing::TestWithParam<tuple<bool, tuple<Backend, Target> > > Layer_Test_Eltwise_unequal;
TEST_P(Layer_Test_Eltwise_unequal, accuracy_input_0_truncate)
{
    bool weighted = get<0>(GetParam());
    int backendId = get<0>(get<1>(GetParam()));
    int targetId = get<1>(get<1>(GetParam()));

    if (backendId == DNN_BACKEND_CUDA && weighted)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);

    Net net;
    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = "testLayer";
    lp.set<std::string>("output_channels_mode", "input_0_truncate");

    const int inpShapes[][4] = {{1, 4, 2, 2}, {1, 5, 2, 2}, {1, 3, 2, 2}};
    const int out_channels = inpShapes[0][1];
    std::vector<String> inpNames(3);
    std::vector<Mat> inputs(3);

    std::vector<float> weights(3, 1);
    if (weighted)
    {
        for (int i = 0; i < inputs.size(); ++i)
            weights[i] = -0.125f + i * 0.25f;
        lp.set("coeff", DictValue::arrayReal<float*>(&weights[0], weights.size()));
    }

    int eltwiseId = net.addLayer(lp.name, lp.type, lp);
    for (int i = 0; i < inputs.size(); ++i)
    {
        inputs[i].create(4, inpShapes[i], CV_32F);
        size_t total = inputs[i].total();
        for (size_t j = 0; j < total; j++)
            inputs[i].ptr<float>()[j] = j + i * 100;
        inpNames[i] = format("input_%d", i);
        net.connect(0, i, eltwiseId, i);
    }
    Mat ref(4, inpShapes[0], CV_32F, Scalar(0));

    net.setInputsNames(inpNames);
    for (int i = 0; i < inputs.size(); ++i)
    {
        //std::cout << ref.reshape(1,1) << endl;
        net.setInput(inputs[i], inpNames[i]);
        for (size_t batchId = 0; batchId < ref.size[0]; batchId++)
        {
            int input_channels = inputs[i].size[1];
            Range ranges[4] = { Range(batchId, batchId + 1), Range(0, std::min(out_channels, input_channels)), Range::all(), Range::all() };
            Mat ref_slice = ref(ranges);
            Mat input_slice = inputs[i](ranges);
            ref_slice += weights[i] * input_slice;
        }
    }

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    Mat out = net.forward();
    normAssert(out, ref);
    if (testing::Test::HasFailure())
    {
        std::cout << out.reshape(1,1) << endl;
        std::cout << ref.reshape(1,1) << endl;
    }
}

TEST_P(Layer_Test_Eltwise_unequal, accuracy_input_0)
{
    bool weighted = get<0>(GetParam());
    int backendId = get<0>(get<1>(GetParam()));
    int targetId = get<1>(get<1>(GetParam()));

    Net net;
    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = "testLayer";
    lp.set<std::string>("output_channels_mode", "input_0");

    if (backendId == DNN_BACKEND_CUDA && weighted)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);

    const int inpShapes[][4] = {{1, 4, 2, 2}, {1, 2, 2, 2}, {1, 3, 2, 2}};
    const int out_channels = inpShapes[0][1];
    std::vector<String> inpNames(3);
    std::vector<Mat> inputs(3);

    std::vector<float> weights(3, 1);
    if (weighted)
    {
        for (int i = 0; i < inputs.size(); ++i)
            weights[i] = -0.125f + i * 0.25f;
        lp.set("coeff", DictValue::arrayReal<float*>(&weights[0], weights.size()));
    }

    int eltwiseId = net.addLayer(lp.name, lp.type, lp);
    for (int i = 0; i < inputs.size(); ++i)
    {
        inputs[i].create(4, inpShapes[i], CV_32F);
        size_t total = inputs[i].total();
        for (size_t j = 0; j < total; j++)
            inputs[i].ptr<float>()[j] = j + i * 100;
        inpNames[i] = format("input_%d", i);
        net.connect(0, i, eltwiseId, i);
    }
    Mat ref(4, inpShapes[0], CV_32F, Scalar(0));

    net.setInputsNames(inpNames);
    for (int i = 0; i < inputs.size(); ++i)
    {
        //std::cout << ref.reshape(1,1) << endl;
        net.setInput(inputs[i], inpNames[i]);
        for (size_t batchId = 0; batchId < ref.size[0]; batchId++)
        {
            int input_channels = inputs[i].size[1];
            Range ranges[4] = { Range(batchId, batchId + 1), Range(0, std::min(out_channels, input_channels)), Range::all(), Range::all() };
            Mat ref_slice = ref(ranges);
            Mat input_slice = inputs[i](ranges);
            ref_slice += weights[i] * input_slice;
        }
    }

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    Mat out = net.forward();
    normAssert(out, ref);
    if (testing::Test::HasFailure())
    {
        std::cout << out.reshape(1,1) << endl;
        std::cout << ref.reshape(1,1) << endl;
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_Eltwise_unequal, Combine(
    testing::Bool(),
    dnnBackendsAndTargets()
));


struct Layer_Test_Eltwise_bcast : testing::TestWithParam<tuple<string, int, tuple<Backend, Target>>>
{
public:
    void test_bcast()
    {
        string op = get<0>(GetParam());
        int dim = get<1>(GetParam());
        tuple<Backend, Target> backend_target= get<2>(GetParam());
        int backend = get<0>(backend_target);
        int target = get<1>(backend_target);

        if (backend == DNN_BACKEND_CUDA && dim > 4)
            applyTestTag(CV_TEST_TAG_LONG);

        vector<vector<int>> dim_shape_list;
        get_all_arr(dim_shape_list, dim);
        replace(dim_shape_list, 1, 3);
        // same shape
        for (int i = 0; i < dim_shape_list.size(); i++)
            for (int j = 0; j < dim_shape_list.size(); j++)
                run(dim_shape_list[i], dim_shape_list[j], op, backend, target);

        vector<vector<int>> sub_shape_list;
        vector<vector<int>> tmp;
        for(int i = 1; i < dim; i++){
            get_all_arr(tmp, i);
            replace(tmp, 1, 3);
            sub_shape_list.insert(sub_shape_list.end(), tmp.begin(), tmp.end());
        }

        // diff shape
        for (const auto &shp1: dim_shape_list)
            for (const auto &shp2: sub_shape_list)
                run(shp1, shp2, op, backend, target);

        // diff shape
        for (const auto &shp1: sub_shape_list)
            for (const auto &shp2: dim_shape_list)
                run(shp1, shp2, op, backend, target);
    }

private:
    // give n to generate all n-D arrays with 0 or 1
    static void get_all_arr(vector<vector<int>> &arr, int n)
    {
        int total = 1 << n;
        arr.assign(total, vector<int>(n, -1));
        for (int i = 0; i < total; i++)
            for (int j = 0; j < n; j++)
                arr[i][j] = (i >> (n - j - 1)) & 1;
    }

    // zero will replace all 0, one will replace all 1
    static void replace(vector<vector<int>> &arr, int zero, int one)
    {
        for (int i = 0; i < arr.size(); i++)
            for (int j = 0; j < arr[0].size(); j++)
                arr[i][j] = arr[i][j] ? one : zero;
    }

    static void run(const vector<int> &a_shape, const vector<int> &b_shape, const String &op, const int backend, const int target)
    {
        Mat a = Mat::zeros((int) a_shape.size(), a_shape.data(), CV_32FC1);
        Mat b = Mat::ones((int) b_shape.size(), b_shape.data(), CV_32FC1);

        Net net;
        LayerParams lp;
        lp.type = "NaryEltwise";
        lp.name = "testLayer";
        lp.set("operation", op);
        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 1, id, 1);

        vector<String> inpNames(2);
        inpNames[0] = "a";
        inpNames[1] = "b";
        net.setInputsNames(inpNames);
        net.setInput(a, inpNames[0]);
        net.setInput(b, inpNames[1]);

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        Mat re;
        re = net.forward();
        auto ptr_re = (float *) re.data;
        for (int i = 0; i < re.total(); i++)
            if (op == "sum"){
                ASSERT_EQ(1, ptr_re[i]); // sum result should be 1
            }
    }
};

TEST_P(Layer_Test_Eltwise_bcast, brute_force)
{
    test_bcast();
}

// This test is to verify whether the broadcast operations of unidirectional and bidirectional,
// as well as tensors with same and different shapes, can be forwarded correctly.
// This can ensure that the elementwise layer does not have any errors when forwarding.
//
// To test which cases the backend will fallback to the cpu, replace the fallback command like
// `return Ptr<BackendNode>();` in `initCUDA()` with `throw std::runtime_error("fallback");`
//
// To test more operators, add more ops after "sum".
// Default only "sum" is tested, because for the most cases they have the same implementation.
INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_Eltwise_bcast, Combine(
        Values("sum"),
        Values(1, 2, 3, 4, 5),
        dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<Backend, Target> > Layer_Test_Resize;
TEST_P(Layer_Test_Resize, change_input)
{
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());

    Net net;
    LayerParams lp;
    lp.type = "Resize";
    lp.name = "testLayer";
    lp.set("zoom_factor", 2);
    lp.set("interpolation", "nearest");
    net.addLayerToPrev(lp.name, lp.type, lp);

    for (int i = 0; i < 2; ++i)
    {
        Mat inp(4 + i, 5 + i, CV_8UC3), ref;
        randu(inp, 0, 255);
        resize(inp, ref, Size(0, 0), 2, 2, INTER_NEAREST);
        ref = blobFromImage(ref);

        net.setInput(blobFromImage(inp));
        net.setPreferableBackend(backendId);
        net.setPreferableTarget(targetId);
        Mat out = net.forward();
        normAssert(out, ref);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_Resize, dnnBackendsAndTargets());

struct Layer_Test_Slice : public testing::TestWithParam<tuple<Backend, Target> >
{
    template<int DIMS>
    void test_slice(const int* inputShape, const int* begin, const int* end)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat input(DIMS, inputShape, CV_32FC1, Scalar::all(0));
        for (int i = 0; i < (int)input.total(); ++i)
            input.ptr<float>()[i] = (float)i;

        std::vector<Range> range(DIMS);
        for (int i = 0; i < DIMS; ++i)
            range[i] = Range(begin[i], end[i]);

        Net net;
        LayerParams lp;
        lp.type = "Slice";
        lp.name = "testLayer";
        lp.set("begin", DictValue::arrayInt<int*>((int*)&begin[0], DIMS));
        lp.set("end", DictValue::arrayInt<int*>((int*)&end[0], DIMS));
        net.addLayerToPrev(lp.name, lp.type, lp);

        {
            net.setInput(input);
            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();

            EXPECT_GT(cv::norm(out, NORM_INF), 0);
            normAssert(out, input(range));
#if 0
            cout << input(range).clone().reshape(1, 1) << endl;
            cout << out.reshape(1, 1) << endl;
#endif
        }
    }
};

TEST_P(Layer_Test_Slice, slice_channels_17762)
{
    const int inputShape[4] = {1, 16, 6, 8};
    const int begin[] = {0, 4, 0, 0};
    const int end[] = {1, 8, 6, 8};
    test_slice<4>(inputShape, begin, end);
}

TEST_P(Layer_Test_Slice, slice_channels_with_batch_17762)
{
    const int inputShape[4] = {4, 4, 3, 4};
    const int begin[] = {0, 1, 0, 0};
    const int end[] = {4, 3, 3, 4};
    test_slice<4>(inputShape, begin, end);
}

TEST_P(Layer_Test_Slice, slice_channels_and_batch_17762)
{
    int backend = get<0>(GetParam());
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    const int inputShape[4] = {4, 4, 3, 4};
    const int begin[] = {2, 1, 0, 0};
    const int end[] = {4, 3, 3, 4};
    test_slice<4>(inputShape, begin, end);
}

TEST_P(Layer_Test_Slice, slice_rows)
{
    const int inputShape[4] = {1, 2, 6, 4};
    const int begin[] = {0, 0, 4, 0};
    const int end[] = {1, 2, 6, 4};
    test_slice<4>(inputShape, begin, end);
}

TEST_P(Layer_Test_Slice, slice_cols)
{
    const int inputShape[4] = {1, 2, 3, 8};
    const int begin[] = {0, 0, 0, 4};
    const int end[] = {1, 2, 3, 8};
    test_slice<4>(inputShape, begin, end);
}


TEST_P(Layer_Test_Slice, slice_complex_1_unaligned)
{
    const int inputShape[4] = {1, 4, 2, 3};
    const int begin[] = {0, 2, 1, 0};
    const int end[] = {1, 3, 2, 2};
    test_slice<4>(inputShape, begin, end);
}

TEST_P(Layer_Test_Slice, slice_complex_2_x4)
{
    const int inputShape[4] = {1, 3, 2, 4};
    const int begin[] = {0, 2, 1, 0};
    const int end[] = {1, 3, 2, 2};
    test_slice<4>(inputShape, begin, end);
}

TEST_P(Layer_Test_Slice, slice_complex_3)
{
    const int inputShape[4] = {1, 6, 4, 8};
    const int begin[] = {0, 2, 1, 4};
    const int end[] = {1, 4, 3, 8};
    test_slice<4>(inputShape, begin, end);
}

TEST_P(Layer_Test_Slice, variable_input_shape)
{
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());

    int begin[] = {0, 0, 0, 0};
    int end[] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};

    Net net;
    LayerParams lp;
    lp.type = "Slice";
    lp.name = "testLayer";
    lp.set("begin", DictValue::arrayInt<int*>(&begin[0], 4));
    lp.set("end", DictValue::arrayInt<int*>(&end[0], 4));
    net.addLayerToPrev(lp.name, lp.type, lp);

    for (int i = 0; i < 2; ++i)
    {
        Mat inp(4 + i, 5 + i, CV_8UC1);
        randu(inp, 0, 255);
        inp = blobFromImage(inp);

        net.setInput(inp);
        net.setPreferableBackend(backendId);
        net.setPreferableTarget(targetId);
        Mat out = net.forward();

        normAssert(out, inp);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_Slice, dnnBackendsAndTargets());

typedef testing::TestWithParam<tuple<Backend, Target> > Layer_Test_BatchNorm;
TEST_P(Layer_Test_BatchNorm, fusion)
{
    // This tests reinitializes network by forwarding different batch size input.
    // We check BatchNorm layer weights restoring after fusion.
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());
    const int ch = 4;

    Mat mean(1, ch, CV_32F), var(1, ch, CV_32F), weights(1, ch, CV_32F);
    randu(mean, 0, 1);
    randu(var, 0, 1);
    randu(weights, 0, 1);

    Net net;
    {
        LayerParams lp;
        lp.type = "BatchNorm";
        lp.name = "bn";
        lp.set("has_weight", false);
        lp.set("has_bias", false);
        lp.blobs.push_back(mean);
        lp.blobs.push_back(var);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }
    {
        LayerParams lp;
        lp.type = "Scale";
        lp.name = "scale";
        lp.set("has_bias", false);
        lp.blobs.push_back(weights);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }

    Mat inp(4, 5, CV_32FC(ch));
    randu(inp, 0, 1);

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    net.setInput(blobFromImage(inp));
    Mat ref = net.forward();

    net.setInput(blobFromImages(std::vector<Mat>(2, inp)));
    Mat out = net.forward();

    for (int i = 0; i < 2; ++i)
    {
        std::vector<Range> ranges(4, Range::all());
        ranges[0].start = i;
        ranges[0].end = i + 1;
        normAssert(out(ranges), ref);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Test_BatchNorm, dnnBackendsAndTargets());

class TestLayerFusion : public DNNTestLayer {
public:
    static void makeDefaultTestConvolutionLayer(LayerParams& convParams, int in_channels, int num_filters, bool bias_term)
    {
        const int kernel_h = 3, kernel_w = 3;
        const int pad_h = kernel_h / 2, pad_w = kernel_w / 2;

        convParams.set("kernel_h", kernel_h);
        convParams.set("kernel_w", kernel_w);
        convParams.set("pad_h", pad_h);
        convParams.set("pad_w", pad_w);
        convParams.set("num_output", num_filters);
        convParams.set("bias_term", bias_term);
        convParams.type = "Convolution";
        convParams.name = "convolution";

        float conv_init_magnitude = 1.0f / in_channels / kernel_h / kernel_w;
        int weightsShape[] = {num_filters, in_channels, kernel_h, kernel_w};
        Mat weights(4, &weightsShape[0], CV_32F);
        randu(weights, -conv_init_magnitude, conv_init_magnitude);
        convParams.blobs.push_back(weights);
        if (bias_term)
        {
            Mat bias(1, num_filters, CV_32F);
            randu(bias, -1.0f, 1.0f);
            convParams.blobs.push_back(bias);
        }
    }

    static void makeDefaultTestActivationLayer(LayerParams& activationParams, const std::string& type, int in_channels)
    {
        activationParams.type = type;
        activationParams.name = "activation";
        if (activationParams.type == "ReLU")
            activationParams.set("negative_slope", 0.1f);
        else if (activationParams.type == "Power")
        {
            activationParams.set("power", 2.0f);
            activationParams.set("scale", 0.5f);
            activationParams.set("shift", 0.3f);
        }
        else if (activationParams.type == "ReLU6")
        {
            activationParams.set("min_value", -1.0f);
            activationParams.set("max_value", 1.0f);
        }
        else if (activationParams.type == "ChannelsPReLU")
        {
            Mat scales(1, in_channels, CV_32F);
            randu(scales, -1.0f, 1.0f);
            activationParams.blobs.push_back(scales);
        }
        else if (activationParams.type == "Exp")
        {
            activationParams.set("base", -1.0f);
            activationParams.set("scale", 0.3f);
            activationParams.set("shift", 0.6f);
        }
    }

    static void makeDefaultTestEltwiseLayer(LayerParams& eltwiseParams, const std::string& op, bool withCoefficients)
    {
        eltwiseParams.type = "Eltwise";
        eltwiseParams.name = "eltwise";
        eltwiseParams.set("operation", op);
        if (withCoefficients)
        {
            float coeff[] = {0.3f, 0.5f};
            eltwiseParams.set("coeff", DictValue::arrayReal<float*>(coeff, 2));
        }
    }

    static void test(Mat& input, Net& net, Backend backendId, Target targetId, std::vector<int> expectedFusedLayers = std::vector<int>(), double l1 = 0.0, double lInf = 0.0)
    {
        DNNTestLayer::checkBackend(backendId, targetId);

        net.enableFusion(false);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        net.setInput(input);
        Mat outputReference = net.forward().clone();
        std::vector<double> refTimings;
        net.getPerfProfile(refTimings);
        for (int i = 0; i < refTimings.size(); i++)
        {
            CV_Assert(refTimings[i] != 0.0);
        }

        net.enableFusion(true);
        net.setPreferableBackend(backendId);
        net.setPreferableTarget(targetId);
        net.setInput(input);
        Mat outputTest = net.forward().clone();
        std::vector<double> testTimings;
        net.getPerfProfile(testTimings);
        for (int i = 0; i < testTimings.size(); i++)
        {
            if(std::find(expectedFusedLayers.begin(), expectedFusedLayers.end(), i + 1) != expectedFusedLayers.end())
            {
                EXPECT_EQ(testTimings[i], 0.0);
            }
            else
            {
                EXPECT_NE(testTimings[i], 0.0);
            }
        }

        // double ref_max_value, ref_min_value;
        // minMaxLoc(outputReference.reshape(1, 1), &ref_min_value, &ref_max_value);
        // std::cout << "reference range: " << ref_min_value << ' ' << ref_max_value << std::endl;

        double default_l1, default_lInf;
        DNNTestLayer::getDefaultThresholds(backendId, targetId, &default_l1, &default_lInf);
        if (l1 == 0.0)
            l1 = default_l1;
        if (lInf == 0.0)
            lInf = default_lInf;
        normAssert(outputReference, outputTest, "", l1, lInf);
    }

    static testing::internal::ParamGenerator<std::string> eltwiseOpList()
    {
        // TODO: automate list generation
        return Values("sum", "max", "min", "prod", "div");
    }

    static testing::internal::ParamGenerator<std::string> activationLayersList()
    {
        // TODO: automate list generation
        return Values("ReLU", "ReLU6", "ChannelsPReLU", "TanH", "Swish", "Mish", "Sigmoid", "ELU", "AbsVal", "BNLL", "Power", "Exp");
    }

    static testing::internal::ParamGenerator<tuple<Backend, Target> > dnnBackendsAndTargetsForFusionTests()
    {
        return dnnBackendsAndTargets(false, false, true, false, true, false); // OCV OpenCL + OCV CPU + CUDA
    }
};

typedef TestWithParam<tuple<bool, std::string, tuple<Backend, Target> > > ConvolutionActivationFusion;
TEST_P(ConvolutionActivationFusion, Accuracy)
{
    //          input
    //            |
    // -----------------------
    // |     convolution     |
    // -----------------------
    //            |
    // -----------------------
    // |     activation      |
    // -----------------------
    //            |
    //         output

    const int batch_size = 2, in_channels = 16;
    const int in_height = 16, in_width = 16;
    int inputShape[] = {batch_size, in_channels, in_height, in_width};
    Mat input(4, &inputShape[0], CV_32F);
    randu(input, 1.0f, 2.0f);

    bool bias_term = get<0>(GetParam());
    LayerParams convParams;
    TestLayerFusion::makeDefaultTestConvolutionLayer(convParams, in_channels, in_channels, bias_term);

    std::string actType = get<1>(GetParam());
    LayerParams activationParams;
    TestLayerFusion::makeDefaultTestActivationLayer(activationParams, actType, in_channels);

    Backend backendId = get<0>(get<2>(GetParam()));
    Target targetId = get<1>(get<2>(GetParam()));

    Net net;
    int convId = net.addLayer(convParams.name, convParams.type, convParams);
    int activId = net.addLayerToPrev(activationParams.name, activationParams.type, activationParams);
    net.connect(0, 0, convId, 0);

    std::vector<int> expectedFusedLayers;
    if (backendId == DNN_BACKEND_OPENCV)
    {
        if (targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16)
            expectedFusedLayers.push_back(activId); // all activations are fused
        else if (targetId == DNN_TARGET_OPENCL || targetId == DNN_TARGET_OPENCL_FP16)
        {
            if (actType == "ReLU" || actType == "ChannelsPReLU" || actType == "ReLU6" || actType == "TanH" /*|| actType == "Power"*/)
                expectedFusedLayers.push_back(activId);
        }
    }
    else if (backendId == DNN_BACKEND_CUDA)
    {
        if (actType == "ReLU" || actType == "ReLU6" || actType == "TanH" || actType == "Swish" ||
            actType == "Mish" || actType == "Sigmoid" || actType == "Power")
                expectedFusedLayers.push_back(activId);
    }
    TestLayerFusion::test(input, net, backendId, targetId, expectedFusedLayers);
}
INSTANTIATE_TEST_CASE_P(TestLayerFusion, ConvolutionActivationFusion, Combine(
/* bias */       testing::Bool(),
/* activation */ TestLayerFusion::activationLayersList(),
                 TestLayerFusion::dnnBackendsAndTargetsForFusionTests()
));

typedef TestWithParam<tuple<bool, std::string, bool, tuple<Backend, Target> > > ConvolutionEltwiseFusion;
TEST_P(ConvolutionEltwiseFusion, Accuracy)
{
    //                 input
    //                   |
    //    -------------------------------
    //    |                             |
    //    |                      ---------------
    //    |                      | convolution |
    //    |                      ---------------
    //    |                             |
    //    |       ----------------      |
    //    --------|  eltwise op  |-------
    //            ----------------
    //                   |
    //                 output

    const int batch_size = 2, in_channels = 16;
    const int in_height = 16, in_width = 16;
    int inputShape[] = {batch_size, in_channels, in_height, in_width};
    Mat input(4, &inputShape[0], CV_32F);
    randu(input, 1.0f, 2.0f); // avoid small values to test eltwise div

    bool bias_term = get<0>(GetParam());
    LayerParams convParams;
    TestLayerFusion::makeDefaultTestConvolutionLayer(convParams, in_channels, in_channels, bias_term);

    std::string eltwiseOp = get<1>(GetParam());
    bool weightedEltwise = get<2>(GetParam());
    if (eltwiseOp != "sum" && weightedEltwise)
        throw SkipTestException("weighted eltwise not supported");
    LayerParams eltwiseParams;
    TestLayerFusion::makeDefaultTestEltwiseLayer(eltwiseParams, eltwiseOp, weightedEltwise);

    Net net;
    int convId = net.addLayer(convParams.name, convParams.type, convParams);
    int eltwiseId = net.addLayer(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
    net.connect(0, 0, convId, 0);
    net.connect(convId, 0, eltwiseId, 0);
    net.connect(0, 0, eltwiseId, 1);

    Backend backendId = get<0>(get<3>(GetParam()));
    Target targetId = get<1>(get<3>(GetParam()));

    std::vector<int> expectedFusedLayers;
    if (backendId == DNN_BACKEND_CUDA && eltwiseOp == "sum" && !weightedEltwise)
        expectedFusedLayers.push_back(eltwiseId);
    TestLayerFusion::test(input, net, backendId, targetId, expectedFusedLayers);
}
INSTANTIATE_TEST_CASE_P(TestLayerFusion, ConvolutionEltwiseFusion, Combine(
/* bias */              testing::Bool(),
/* eltwise op */        TestLayerFusion::eltwiseOpList(),
/* eltwise weighted */  testing::Bool(),
                        TestLayerFusion::dnnBackendsAndTargetsForFusionTests()
));

typedef TestWithParam<tuple<bool, std::string, bool, std::string, tuple<Backend, Target> > > ConvolutionEltwiseActivationFusion;
TEST_P(ConvolutionEltwiseActivationFusion, Accuracy)
{
    //                 input
    //                   |
    //    -------------------------------
    //    |                             |
    //    |                      ---------------
    //    |                      | convolution |
    //    |                      ---------------
    //    |                             |
    //    |       ----------------      |
    //    --------|  eltwise op  |-------
    //            ----------------
    //                   |
    //            ----------------
    //            |  activation  |
    //            ----------------
    //                   |
    //                output

    const int batch_size = 2, in_channels = 16;
    const int in_height = 16, in_width = 16;
    int inputShape[] = {batch_size, in_channels, in_height, in_width};
    Mat input(4, &inputShape[0], CV_32F);
    randu(input, 1.0f, 2.0f); // avoid small values to test eltwise div

    bool bias_term = get<0>(GetParam());
    LayerParams convParams;
    TestLayerFusion::makeDefaultTestConvolutionLayer(convParams, in_channels, in_channels, bias_term);

    std::string eltwiseOp = get<1>(GetParam());
    bool weightedEltwise = get<2>(GetParam());
    if (eltwiseOp != "sum" && weightedEltwise)
            throw SkipTestException("weighted eltwise not supported");
    LayerParams eltwiseParams;
    TestLayerFusion::makeDefaultTestEltwiseLayer(eltwiseParams, eltwiseOp, weightedEltwise);

    std::string actType = get<3>(GetParam());
    LayerParams activationParams;
    TestLayerFusion::makeDefaultTestActivationLayer(activationParams, actType, in_channels);

    Backend backendId = get<0>(get<4>(GetParam()));
    Target targetId = get<1>(get<4>(GetParam()));

    Net net;
    int convId = net.addLayer(convParams.name, convParams.type, convParams);
    int eltwiseId = net.addLayer(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
    int activId = net.addLayer(activationParams.name, activationParams.type, activationParams);
    net.connect(0, 0, convId, 0);
    net.connect(convId, 0, eltwiseId, 0);
    net.connect(0, 0, eltwiseId, 1);
    net.connect(eltwiseId, 0, activId, 0);

    std::vector<int> expectedFusedLayers;
    if (backendId == DNN_BACKEND_OPENCV)
    {
        if (targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16)
            expectedFusedLayers.push_back(activId); // activation is fused with eltwise layer
        else if (targetId == DNN_TARGET_OPENCL || targetId == DNN_TARGET_OPENCL_FP16)
        {
            if (eltwiseOp == "sum" && !weightedEltwise &&
                (actType == "ReLU" || actType == "ChannelsPReLU" /*|| actType == "Power"*/)
            )
            {
                expectedFusedLayers.push_back(eltwiseId);
                expectedFusedLayers.push_back(activId);
            }
        }
    }
    else if(backendId == DNN_BACKEND_CUDA)
    {
        if (eltwiseOp == "sum" && !weightedEltwise)
        {
            expectedFusedLayers.push_back(eltwiseId);
            if (actType == "ReLU" || actType == "ReLU6" || actType == "TanH" || actType == "Swish" ||
                actType == "Mish" || actType == "Sigmoid" || actType == "Power")
                expectedFusedLayers.push_back(activId);
        }
    }
    TestLayerFusion::test(input, net, backendId, targetId, expectedFusedLayers);
}
INSTANTIATE_TEST_CASE_P(TestLayerFusion, ConvolutionEltwiseActivationFusion, Combine(
/* bias */              testing::Bool(),
/* eltwise op */        TestLayerFusion::eltwiseOpList(),
/* eltwise weighted */  testing::Bool(),
/* activation */        TestLayerFusion::activationLayersList(),
                        TestLayerFusion::dnnBackendsAndTargetsForFusionTests()
));

typedef TestWithParam<tuple<bool, std::string, std::string, bool, tuple<Backend, Target> > > ConvolutionActivationEltwiseFusion;
TEST_P(ConvolutionActivationEltwiseFusion, Accuracy)
{
    //                 input
    //                   |
    //    -------------------------------
    //    |                             |
    //    |                     ----------------
    //    |                     |  convolution |
    //    |                     ----------------
    //    |                             |
    //    |                     ----------------
    //    |                     |  activation  |
    //    |                     ----------------
    //    |                             |
    //    |       ----------------      |
    //    --------| eltwise sum  |-------
    //            ----------------
    //                   |

    const int batch_size = 2, in_channels = 16;
    const int in_height = 16, in_width = 16;
    int inputShape[] = {batch_size, in_channels, in_height, in_width};
    Mat input(4, &inputShape[0], CV_32F);
    randu(input, 1.0f, 2.0f); // avoid small values to test eltwise div

    bool bias_term = get<0>(GetParam());
    LayerParams convParams;
    TestLayerFusion::makeDefaultTestConvolutionLayer(convParams, in_channels, in_channels, bias_term);

    std::string actType = get<1>(GetParam());
    LayerParams activationParams;
    TestLayerFusion::makeDefaultTestActivationLayer(activationParams, actType, in_channels);

    std::string eltwiseOp = get<2>(GetParam());
    bool weightedEltwise = get<3>(GetParam());
    if (eltwiseOp != "sum" && weightedEltwise)
            throw SkipTestException("weighted eltwise not supported");
    LayerParams eltwiseParams;
    TestLayerFusion::makeDefaultTestEltwiseLayer(eltwiseParams, eltwiseOp, weightedEltwise);

    Backend backendId = get<0>(get<4>(GetParam()));
    Target targetId = get<1>(get<4>(GetParam()));

    Net net;
    int convId = net.addLayer(convParams.name, convParams.type, convParams);
    int activId = net.addLayer(activationParams.name, activationParams.type, activationParams);
    int eltwiseId = net.addLayer(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
    net.connect(0, 0, convId, 0);
    net.connect(convId, 0, activId, 0);
    net.connect(activId, 0, eltwiseId, 0);
    net.connect(0, 0, eltwiseId, 1);

    std::vector<int> expectedFusedLayers;
    if (backendId == DNN_BACKEND_OPENCV)
    {
        if (targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16)
            expectedFusedLayers.push_back(activId); // activation fused with convolution
        else if (targetId == DNN_TARGET_OPENCL || targetId == DNN_TARGET_OPENCL_FP16)
        {
            if (actType == "ReLU" || actType == "ChannelsPReLU" || actType == "ReLU6" || actType == "TanH" /*|| actType == "Power"*/)
                expectedFusedLayers.push_back(activId); // activation fused with convolution
        }
    }
    else if(backendId == DNN_BACKEND_CUDA)
    {
        if (actType == "ReLU" || actType == "ReLU6" || actType == "TanH" || actType == "Swish" ||
            actType == "Mish" || actType == "Sigmoid" || actType == "Power")
        {
                expectedFusedLayers.push_back(activId);
                if (eltwiseOp == "sum" && !weightedEltwise)
                    expectedFusedLayers.push_back(eltwiseId);
        }
    }
    TestLayerFusion::test(input, net, backendId, targetId, expectedFusedLayers);
}
INSTANTIATE_TEST_CASE_P(TestLayerFusion, ConvolutionActivationEltwiseFusion, Combine(
/* bias */              testing::Bool(),
/* activation */        TestLayerFusion::activationLayersList(),
/* eltwise op */        TestLayerFusion::eltwiseOpList(),
/* eltwise weighted */  testing::Bool(),
                        TestLayerFusion::dnnBackendsAndTargetsForFusionTests()
));

}} // namespace
