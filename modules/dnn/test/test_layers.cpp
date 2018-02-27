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
#include <opencv2/ts/ocl_test.hpp>

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
    std::vector<Mat> inp_(ninputs);
    std::vector<Mat*> inp(ninputs);
    std::vector<Mat> outp, intp;
    std::vector<MatShape> inputs, outputs, internals;

    for (size_t i = 0; i < ninputs; i++)
    {
        inp_[i] = inpBlobs[i].clone();
        inp[i] = &inp_[i];
        inputs.push_back(shape(inp_[i]));
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


void testLayerUsingCaffeModels(String basename, int targetId = DNN_TARGET_CPU,
                               bool useCaffeModel = false, bool useCommonInputBlob = true)
{
    String prototxt = _tf(basename + ".prototxt");
    String caffemodel = _tf(basename + ".caffemodel");

    String inpfile = (useCommonInputBlob) ? _tf("blob.npy") : _tf(basename + ".input.npy");
    String outfile = _tf(basename + ".npy");

    Net net = readNetFromCaffe(prototxt, (useCaffeModel) ? caffemodel : String());
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(targetId);

    Mat inp = blobFromNPY(inpfile);
    Mat ref = blobFromNPY(outfile);

    net.setInput(inp, "input");
    Mat out = net.forward("output");

    normAssert(ref, out);
}

TEST(Layer_Test_Softmax, Accuracy)
{
    testLayerUsingCaffeModels("layer_softmax");
}

OCL_TEST(Layer_Test_Softmax, Accuracy)
{
    testLayerUsingCaffeModels("layer_softmax", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_LRN_spatial, Accuracy)
{
    testLayerUsingCaffeModels("layer_lrn_spatial");
}

OCL_TEST(Layer_Test_LRN_spatial, Accuracy)
{
    testLayerUsingCaffeModels("layer_lrn_spatial", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_LRN_channels, Accuracy)
{
    testLayerUsingCaffeModels("layer_lrn_channels");
}

OCL_TEST(Layer_Test_LRN_channels, Accuracy)
{
    testLayerUsingCaffeModels("layer_lrn_channels", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_Convolution, Accuracy)
{
    testLayerUsingCaffeModels("layer_convolution", DNN_TARGET_CPU, true);
}

OCL_TEST(Layer_Test_Convolution, Accuracy)
{
    testLayerUsingCaffeModels("layer_convolution", DNN_TARGET_OPENCL, true);
}

TEST(Layer_Test_DeConvolution, Accuracy)
{
    testLayerUsingCaffeModels("layer_deconvolution", DNN_TARGET_CPU, true, false);
}

OCL_TEST(Layer_Test_DeConvolution, Accuracy)
{
    testLayerUsingCaffeModels("layer_deconvolution", DNN_TARGET_OPENCL, true, false);
}

TEST(Layer_Test_InnerProduct, Accuracy)
{
    testLayerUsingCaffeModels("layer_inner_product", DNN_TARGET_CPU, true);
}

OCL_TEST(Layer_Test_InnerProduct, Accuracy)
{
    testLayerUsingCaffeModels("layer_inner_product", DNN_TARGET_OPENCL, true);
}

TEST(Layer_Test_Pooling_max, Accuracy)
{
    testLayerUsingCaffeModels("layer_pooling_max");
}

OCL_TEST(Layer_Test_Pooling_max, Accuracy)
{
    testLayerUsingCaffeModels("layer_pooling_max", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_Pooling_ave, Accuracy)
{
    testLayerUsingCaffeModels("layer_pooling_ave");
}

OCL_TEST(Layer_Test_Pooling_ave, Accuracy)
{
    testLayerUsingCaffeModels("layer_pooling_ave", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_MVN, Accuracy)
{
    testLayerUsingCaffeModels("layer_mvn");
}

OCL_TEST(Layer_Test_MVN, Accuracy)
{
    testLayerUsingCaffeModels("layer_mvn", DNN_TARGET_OPENCL);
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
}

TEST(Layer_Test_BatchNorm, Accuracy)
{
    testLayerUsingCaffeModels("layer_batch_norm", DNN_TARGET_CPU, true);
}

TEST(Layer_Test_ReLU, Accuracy)
{
    testLayerUsingCaffeModels("layer_relu");
}

OCL_TEST(Layer_Test_ReLU, Accuracy)
{
    testLayerUsingCaffeModels("layer_relu", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_Dropout, Accuracy)
{
    testLayerUsingCaffeModels("layer_dropout");
}

TEST(Layer_Test_Concat, Accuracy)
{
    testLayerUsingCaffeModels("layer_concat");
}

OCL_TEST(Layer_Test_Concat, Accuracy)
{
    testLayerUsingCaffeModels("layer_concat", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_Fused_Concat, Accuracy)
{
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
    Mat out = net.forward();

    normAssert(slice(out, Range::all(), Range(0, 2), Range::all(), Range::all()), input);
    normAssert(slice(out, Range::all(), Range(2, 4), Range::all(), Range::all()), input);

    //

    testLayerUsingCaffeModels("layer_concat_optim", DNN_TARGET_CPU, true, false);
    testLayerUsingCaffeModels("layer_concat_shared_input", DNN_TARGET_CPU, true, false);
}

TEST(Layer_Test_Eltwise, Accuracy)
{
    testLayerUsingCaffeModels("layer_eltwise");
}

OCL_TEST(Layer_Test_Eltwise, Accuracy)
{
    testLayerUsingCaffeModels("layer_eltwise", DNN_TARGET_OPENCL);
}

TEST(Layer_Test_PReLU, Accuracy)
{
    testLayerUsingCaffeModels("layer_prelu", DNN_TARGET_CPU, true);
    testLayerUsingCaffeModels("layer_prelu_fc", DNN_TARGET_CPU, true, false);
}

OCL_TEST(Layer_Test_PReLU, Accuracy)
{
    testLayerUsingCaffeModels("layer_prelu", DNN_TARGET_OPENCL, true);
    testLayerUsingCaffeModels("layer_prelu_fc", DNN_TARGET_OPENCL, true, false);
}

//template<typename XMat>
//static void test_Layer_Concat()
//{
//    Matx21f a(1.f, 1.f), b(2.f, 2.f), c(3.f, 3.f);
//    std::vector<Blob> res(1), src = { Blob(XMat(a)), Blob(XMat(b)), Blob(XMat(c)) };
//    Blob ref(XMat(Matx23f(1.f, 2.f, 3.f, 1.f, 2.f, 3.f)));
//
//    runLayer(ConcatLayer::create(1), src, res);
//    normAssert(ref, res[0]);
//}
//TEST(Layer_Concat, Accuracy)
//{
//    test_Layer_Concat<Mat>());
//}
//OCL_TEST(Layer_Concat, Accuracy)
//{
//    OCL_ON(test_Layer_Concat<Mat>());
//    );
//}

static void test_Reshape_Split_Slice_layers(int targetId)
{
    Net net = readNetFromCaffe(_tf("reshape_and_slice_routines.prototxt"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(targetId);

    Mat input(6, 12, CV_32F);
    RNG rng(0);
    rng.fill(input, RNG::UNIFORM, -1, 1);

    net.setInput(input, "input");
    Mat output = net.forward("output");

    normAssert(input, output);
}

TEST(Layer_Test_Reshape_Split_Slice, Accuracy)
{
    test_Reshape_Split_Slice_layers(DNN_TARGET_CPU);
}

OCL_TEST(Layer_Test_Reshape_Split_Slice, Accuracy)
{
    test_Reshape_Split_Slice_layers(DNN_TARGET_OPENCL);
}

TEST(Layer_Conv_Elu, Accuracy)
{
    Net net = readNetFromTensorflow(_tf("layer_elu_model.pb"));
    ASSERT_FALSE(net.empty());

    Mat inp = blobFromNPY(_tf("layer_elu_in.npy"));
    Mat ref = blobFromNPY(_tf("layer_elu_out.npy"));

    net.setInput(inp, "input");
    Mat out = net.forward();

    normAssert(ref, out);
}

class Layer_LSTM_Test : public ::testing::Test
{
public:
    int numInp, numOut;
    Mat Wh, Wx, b;
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

        LayerParams lp;
        lp.blobs.resize(3);
        lp.blobs[0] = Wh;
        lp.blobs[1] = Wx;
        lp.blobs[2] = b;
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
    lp.blobs.resize(3);
    lp.blobs[0] = blobFromNPY(_tf("lstm.prototxt.w_2.npy"));  // Wh
    lp.blobs[1] = blobFromNPY(_tf("lstm.prototxt.w_0.npy"));  // Wx
    lp.blobs[2] = blobFromNPY(_tf("lstm.prototxt.w_1.npy"));  // bias
    Ptr<LSTMLayer> layer = LSTMLayer::create(lp);

    Mat inp = blobFromNPY(_tf("recurrent.input.npy"));
    std::vector<Mat> inputs(1, inp), outputs;
    runLayer(layer, inputs, outputs);

    Mat h_t_reference = blobFromNPY(_tf("lstm.prototxt.h_1.npy"));
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

void testLayerUsingDarknetModels(String basename, bool useDarknetModel = false, bool useCommonInputBlob = true)
{
    String cfg = _tf(basename + ".cfg");
    String weights = _tf(basename + ".weights");

    String inpfile = (useCommonInputBlob) ? _tf("blob.npy") : _tf(basename + ".input.npy");
    String outfile = _tf(basename + ".npy");

    Net net = readNetFromDarknet(cfg, (useDarknetModel) ? weights : String());
    ASSERT_FALSE(net.empty());

    Mat inp = blobFromNPY(inpfile);
    Mat ref = blobFromNPY(outfile);

    net.setInput(inp, "data");
    Mat out = net.forward();

    normAssert(ref, out);
}

TEST(Layer_Test_Region, Accuracy)
{
    testLayerUsingDarknetModels("region", false, false);
}

TEST(Layer_Test_Reorg, Accuracy)
{
    testLayerUsingDarknetModels("reorg", false, false);
}

TEST(Layer_Test_ROIPooling, Accuracy)
{
    Net net = readNetFromCaffe(_tf("net_roi_pooling.prototxt"));

    Mat inp = blobFromNPY(_tf("net_roi_pooling.input.npy"));
    Mat rois = blobFromNPY(_tf("net_roi_pooling.rois.npy"));
    Mat ref = blobFromNPY(_tf("net_roi_pooling.npy"));

    net.setInput(inp, "input");
    net.setInput(rois, "rois");

    Mat out = net.forward();

    normAssert(out, ref);
}

TEST(Layer_Test_FasterRCNN_Proposal, Accuracy)
{
    Net net = readNetFromCaffe(_tf("net_faster_rcnn_proposal.prototxt"));

    Mat scores = blobFromNPY(_tf("net_faster_rcnn_proposal.scores.npy"));
    Mat deltas = blobFromNPY(_tf("net_faster_rcnn_proposal.deltas.npy"));
    Mat imInfo = (Mat_<float>(1, 3) << 600, 800, 1.6f);
    Mat ref = blobFromNPY(_tf("net_faster_rcnn_proposal.npy"));

    net.setInput(scores, "rpn_cls_prob_reshape");
    net.setInput(deltas, "rpn_bbox_pred");
    net.setInput(imInfo, "im_info");

    Mat out = net.forward();

    const int numDets = ref.size[0];
    EXPECT_LE(numDets, out.size[0]);
    normAssert(out.rowRange(0, numDets), ref);

    if (numDets < out.size[0])
        EXPECT_EQ(countNonZero(out.rowRange(numDets, out.size[0])), 0);
}

OCL_TEST(Layer_Test_FasterRCNN_Proposal, Accuracy)
{
    Net net = readNetFromCaffe(_tf("net_faster_rcnn_proposal.prototxt"));

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    Mat scores = blobFromNPY(_tf("net_faster_rcnn_proposal.scores.npy"));
    Mat deltas = blobFromNPY(_tf("net_faster_rcnn_proposal.deltas.npy"));
    Mat imInfo = (Mat_<float>(1, 3) << 600, 800, 1.6f);
    Mat ref = blobFromNPY(_tf("net_faster_rcnn_proposal.npy"));

    net.setInput(scores, "rpn_cls_prob_reshape");
    net.setInput(deltas, "rpn_bbox_pred");
    net.setInput(imInfo, "im_info");

    Mat out = net.forward();

    const int numDets = ref.size[0];
    EXPECT_LE(numDets, out.size[0]);
    normAssert(out.rowRange(0, numDets), ref);

    if (numDets < out.size[0])
        EXPECT_EQ(countNonZero(out.rowRange(numDets, out.size[0])), 0);
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
TEST(Layer_Test_Average_pooling_kernel_area, Accuracy)
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
    Mat target = (Mat_<float>(2, 2) << (1 + 2 + 4 + 5) / 4.f, (3 + 6) / 2.f, (7 + 8) / 2.f, 9);
    Mat tmp = blobFromImage(inp);
    net.setInput(blobFromImage(inp));
    Mat out = net.forward();
    normAssert(out, blobFromImage(target));
}

}} // namespace
