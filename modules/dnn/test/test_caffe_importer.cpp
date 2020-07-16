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

namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    return findDataFile(std::string("dnn/") + filename);
}

class Test_Caffe_nets : public DNNTestLayer
{
public:
    void testFaster(const std::string& proto, const std::string& model, const Mat& ref,
                    double scoreDiff = 0.0, double iouDiff = 0.0)
    {
        checkBackend();
        Net net = readNetFromCaffe(findDataFile("dnn/" + proto),
                                   findDataFile("dnn/" + model, false));
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        Mat img = imread(findDataFile("dnn/dog416.png"));
        resize(img, img, Size(800, 600));
        Mat blob = blobFromImage(img, 1.0, Size(), Scalar(102.9801, 115.9465, 122.7717), false, false);
        Mat imInfo = (Mat_<float>(1, 3) << img.rows, img.cols, 1.6f);

        net.setInput(blob, "data");
        net.setInput(imInfo, "im_info");
        // Output has shape 1x1xNx7 where N - number of detections.
        // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
        Mat out = net.forward();
        scoreDiff = scoreDiff ? scoreDiff : default_l1;
        iouDiff = iouDiff ? iouDiff : default_lInf;
        normAssertDetections(ref, out, ("model name: " + model).c_str(), 0.8, scoreDiff, iouDiff);
    }
};

TEST(Test_Caffe, memory_read)
{
    const string proto = findDataFile("dnn/bvlc_googlenet.prototxt");
    const string model = findDataFile("dnn/bvlc_googlenet.caffemodel", false);

    std::vector<char> dataProto;
    readFileContent(proto, dataProto);

    std::vector<char> dataModel;
    readFileContent(model, dataModel);

    Net net = readNetFromCaffe(dataProto.data(), dataProto.size());
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    ASSERT_FALSE(net.empty());

    Net net2 = readNetFromCaffe(dataProto.data(), dataProto.size(),
                                dataModel.data(), dataModel.size());
    ASSERT_FALSE(net2.empty());
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

TEST_P(Test_Caffe_nets, Axpy)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    String proto = _tf("axpy.prototxt");
    Net net = readNetFromCaffe(proto);

    checkBackend();
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    int size[] = {1, 2, 3, 4};
    int scale_size[] = {1, 2, 1, 1};
    Mat scale(4, &scale_size[0], CV_32F);
    Mat shift(4, &size[0], CV_32F);
    Mat inp(4, &size[0], CV_32F);
    randu(scale, -1.0f, 1.0f);
    randu(shift, -1.0f, 1.0f);
    randu(inp, -1.0f, 1.0f);

    net.setInput(scale, "scale");
    net.setInput(shift, "shift");
    net.setInput(inp, "data");

    Mat out = net.forward();

    Mat ref(4, &size[0], inp.type());
    for (int i = 0; i < inp.size[1]; i++) {
        for (int h = 0; h < inp.size[2]; h++) {
            for (int w = 0; w < inp.size[3]; w++) {
                int idx[] = {0, i, h, w};
                int scale_idx[] = {0, i, 0, 0};
                ref.at<float>(idx) = inp.at<float>(idx) * scale.at<float>(scale_idx) +
                                     shift.at<float>(idx);
            }
        }
    }
    float l1 = (target == DNN_TARGET_OPENCL_FP16) ? 2e-4 : 1e-5;
    float lInf = (target == DNN_TARGET_OPENCL_FP16) ? 1e-3 : 1e-4;
    normAssert(ref, out, "", l1, lInf);
}

typedef testing::TestWithParam<tuple<bool, Target> > Reproducibility_AlexNet;
TEST_P(Reproducibility_AlexNet, Accuracy)
{
    Target targetId = get<1>(GetParam());
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#else
    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#endif
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU);

    bool readFromMemory = get<0>(GetParam());
    Net net;
    {
        const string proto = findDataFile("dnn/bvlc_alexnet.prototxt");
        const string model = findDataFile("dnn/bvlc_alexnet.caffemodel", false);
        if (readFromMemory)
        {
            std::vector<char> dataProto;
            readFileContent(proto, dataProto);
            std::vector<char> dataModel;
            readFileContent(model, dataModel);

            net = readNetFromCaffe(dataProto.data(), dataProto.size(),
                                   dataModel.data(), dataModel.size());
        }
        else
            net = readNetFromCaffe(proto, model);
        ASSERT_FALSE(net.empty());
    }

    // Test input layer size
    std::vector<MatShape> inLayerShapes;
    std::vector<MatShape> outLayerShapes;
    net.getLayerShapes(MatShape(), 0, inLayerShapes, outLayerShapes);
    ASSERT_FALSE(inLayerShapes.empty());
    ASSERT_EQ(inLayerShapes[0].size(), 4);
    ASSERT_EQ(inLayerShapes[0][0], 1);
    ASSERT_EQ(inLayerShapes[0][1], 3);
    ASSERT_EQ(inLayerShapes[0][2], 227);
    ASSERT_EQ(inLayerShapes[0][3], 227);

    const float l1 = 1e-5;
    const float lInf = (targetId == DNN_TARGET_OPENCL_FP16) ? 3e-3 : 1e-4;

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());

    net.setInput(blobFromImage(sample, 1.0f, Size(227, 227), Scalar(), false), "data");
    Mat out = net.forward("prob");
    Mat ref = blobFromNPY(_tf("caffe_alexnet_prob.npy"));
    normAssert(ref, out, "", l1, lInf);
}

INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_AlexNet, Combine(testing::Bool(),
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV))));

TEST(Reproducibility_FCN, Accuracy)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_VERYLONG, CV_TEST_TAG_MEMORY_2GB);

    Net net;
    {
        const string proto = findDataFile("dnn/fcn8s-heavy-pascal.prototxt");
        const string model = findDataFile("dnn/fcn8s-heavy-pascal.caffemodel", false);
        net = readNetFromCaffe(proto, model);
        ASSERT_FALSE(net.empty());
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

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

TEST(Reproducibility_SSD, Accuracy)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB, CV_TEST_TAG_DEBUG_LONG);
    Net net;
    {
        const string proto = findDataFile("dnn/ssd_vgg16.prototxt");
        const string model = findDataFile("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", false);
        net = readNetFromCaffe(proto, model);
        ASSERT_FALSE(net.empty());
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    if (sample.channels() == 4)
        cvtColor(sample, sample, COLOR_BGRA2BGR);

    Mat in_blob = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    net.setInput(in_blob, "data");
    Mat out = net.forward("detection_out");

    Mat ref = blobFromNPY(_tf("ssd_out.npy"));
    normAssertDetections(ref, out, "", FLT_MIN);
}

typedef testing::TestWithParam<tuple<Backend, Target> > Reproducibility_MobileNet_SSD;
TEST_P(Reproducibility_MobileNet_SSD, Accuracy)
{
    const string proto = findDataFile("dnn/MobileNetSSD_deploy.prototxt", false);
    const string model = findDataFile("dnn/MobileNetSSD_deploy.caffemodel", false);
    Net net = readNetFromCaffe(proto, model);
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    Mat sample = imread(_tf("street.png"));

    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inp);
    Mat out = net.forward().clone();

    ASSERT_EQ(out.size[2], 100);

    const float scores_diff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 1.5e-2 : 1e-5;
    const float boxes_iou_diff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 6.3e-2 : 1e-4;
    Mat ref = blobFromNPY(_tf("mobilenet_ssd_caffe_out.npy"));
    normAssertDetections(ref, out, "", FLT_MIN, scores_diff, boxes_iou_diff);

    // Check that detections aren't preserved.
    inp.setTo(0.0f);
    net.setInput(inp);
    Mat zerosOut = net.forward();
    zerosOut = zerosOut.reshape(1, zerosOut.total() / 7);

    const int numDetections = zerosOut.rows;
    // TODO: fix it
    if (targetId != DNN_TARGET_MYRIAD ||
        getInferenceEngineVPUType() != CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
    {
        ASSERT_NE(numDetections, 0);
        for (int i = 0; i < numDetections; ++i)
        {
            float confidence = zerosOut.ptr<float>(i)[2];
            ASSERT_EQ(confidence, 0);
        }
    }

    // There is something wrong with Reshape layer in Myriad plugin.
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019
        || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH
    )
    {
        if (targetId == DNN_TARGET_MYRIAD || targetId == DNN_TARGET_OPENCL_FP16)
            return;
    }

    // Check batching mode.
    inp = blobFromImages(std::vector<Mat>(2, sample), 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inp);
    Mat outBatch = net.forward();

    // Output blob has a shape 1x1x2Nx7 where N is a number of detection for
    // a single sample in batch. The first numbers of detection vectors are batch id.
    // For Inference Engine backend there is -1 delimiter which points the end of detections.
    const int numRealDetections = ref.size[2];
    EXPECT_EQ(outBatch.size[2], 2 * numDetections);
    out = out.reshape(1, numDetections).rowRange(0, numRealDetections);
    outBatch = outBatch.reshape(1, 2 * numDetections);
    for (int i = 0; i < 2; ++i)
    {
        Mat pred = outBatch.rowRange(i * numRealDetections, (i + 1) * numRealDetections);
        EXPECT_EQ(countNonZero(pred.col(0) != i), 0);
        normAssert(pred.colRange(1, 7), out.colRange(1, 7));
    }
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_MobileNet_SSD, dnnBackendsAndTargets());

typedef testing::TestWithParam<Target> Reproducibility_ResNet50;
TEST_P(Reproducibility_ResNet50, Accuracy)
{
    Target targetId = GetParam();
    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU);

    Net net = readNetFromCaffe(findDataFile("dnn/ResNet-50-deploy.prototxt"),
                               findDataFile("dnn/ResNet-50-model.caffemodel", false));

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    float l1 = (targetId == DNN_TARGET_OPENCL_FP16) ? 3e-5 : 1e-5;
    float lInf = (targetId == DNN_TARGET_OPENCL_FP16) ? 6e-3 : 1e-4;

    Mat input = blobFromImage(imread(_tf("googlenet_0.png")), 1.0f, Size(224,224), Scalar(), false);
    ASSERT_TRUE(!input.empty());

    net.setInput(input);
    Mat out = net.forward();

    Mat ref = blobFromNPY(_tf("resnet50_prob.npy"));
    normAssert(ref, out, "", l1, lInf);

    if (targetId == DNN_TARGET_OPENCL || targetId == DNN_TARGET_OPENCL_FP16)
    {
        UMat out_umat;
        net.forward(out_umat);
        normAssert(ref, out_umat, "out_umat", l1, lInf);

        std::vector<UMat> out_umats;
        net.forward(out_umats);
        normAssert(ref, out_umats[0], "out_umat_vector", l1, lInf);
    }
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_ResNet50,
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));

typedef testing::TestWithParam<Target> Reproducibility_SqueezeNet_v1_1;
TEST_P(Reproducibility_SqueezeNet_v1_1, Accuracy)
{
    int targetId = GetParam();
    if(targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    Net net = readNetFromCaffe(findDataFile("dnn/squeezenet_v1.1.prototxt"),
                               findDataFile("dnn/squeezenet_v1.1.caffemodel", false));
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    Mat input = blobFromImage(imread(_tf("googlenet_0.png")), 1.0f, Size(227,227), Scalar(), false, true);
    ASSERT_TRUE(!input.empty());

    Mat out;
    if (targetId == DNN_TARGET_OPENCL)
    {
        // Firstly set a wrong input blob and run the model to receive a wrong output.
        // Then set a correct input blob to check CPU->GPU synchronization is working well.
        net.setInput(input * 2.0f);
        out = net.forward();
    }
    net.setInput(input);
    out = net.forward();

    Mat ref = blobFromNPY(_tf("squeezenet_v1.1_prob.npy"));
    normAssert(ref, out);
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_SqueezeNet_v1_1,
    testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));

TEST(Reproducibility_AlexNet_fp16, Accuracy)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    const float l1 = 1e-5;
    const float lInf = 3e-3;

    const string proto = findDataFile("dnn/bvlc_alexnet.prototxt");
    const string model = findDataFile("dnn/bvlc_alexnet.caffemodel", false);

    shrinkCaffeModel(model, "bvlc_alexnet.caffemodel_fp16");
    Net net = readNetFromCaffe(proto, "bvlc_alexnet.caffemodel_fp16");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat sample = imread(findDataFile("dnn/grace_hopper_227.png"));

    net.setInput(blobFromImage(sample, 1.0f, Size(227, 227), Scalar()));
    Mat out = net.forward();
    Mat ref = blobFromNPY(findDataFile("dnn/caffe_alexnet_prob.npy"));
    normAssert(ref, out, "", l1, lInf);
}

TEST(Reproducibility_GoogLeNet_fp16, Accuracy)
{
    const float l1 = 1e-5;
    const float lInf = 3e-3;

    const string proto = findDataFile("dnn/bvlc_googlenet.prototxt");
    const string model = findDataFile("dnn/bvlc_googlenet.caffemodel", false);

    shrinkCaffeModel(model, "bvlc_googlenet.caffemodel_fp16");
    Net net = readNetFromCaffe(proto, "bvlc_googlenet.caffemodel_fp16");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

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
TEST_P(Test_Caffe_nets, Colorization)
{
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    checkBackend();

    Mat inp = blobFromNPY(_tf("colorization_inp.npy"));
    Mat ref = blobFromNPY(_tf("colorization_out.npy"));
    Mat kernel = blobFromNPY(_tf("colorization_pts_in_hull.npy"));

    const string proto = findDataFile("dnn/colorization_deploy_v2.prototxt", false);
    const string model = findDataFile("dnn/colorization_release_v2.caffemodel", false);
    Net net = readNetFromCaffe(proto, model);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.getLayer(net.getLayerId("class8_ab"))->blobs.push_back(kernel);
    net.getLayer(net.getLayerId("conv8_313_rh"))->blobs.push_back(Mat(1, 313, CV_32F, 2.606));

    net.setInput(inp);
    Mat out = net.forward();

    // Reference output values are in range [-29.1, 69.5]
    double l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.25 : 4e-4;
    double lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 5.3 : 3e-3;
    if (target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
    {
        l1 = 0.5; lInf = 11;
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.26; lInf = 6.5;
    }

    normAssert(out, ref, "", l1, lInf);
    expectNoFallbacksFromIE(net);
}

TEST_P(Test_Caffe_nets, DenseNet_121)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    checkBackend();
    const string proto = findDataFile("dnn/DenseNet_121.prototxt", false);
    const string model = findDataFile("dnn/DenseNet_121.caffemodel", false);

    Mat inp = imread(_tf("dog416.png"));
    inp = blobFromImage(inp, 1.0 / 255, Size(224, 224), Scalar(), true, true);
    Mat ref = blobFromNPY(_tf("densenet_121_output.npy"));

    Net net = readNetFromCaffe(proto, model);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.setInput(inp);
    Mat out = net.forward();

    // Reference is an array of 1000 values from a range [-6.16, 7.9]
    float l1 = default_l1, lInf = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16)
    {
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019020000)
        l1 = 0.045; lInf = 0.21;
#else
        l1 = 0.017; lInf = 0.0795;
#endif
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        l1 = 0.11; lInf = 0.5;
    }
    normAssert(out, ref, "", l1, lInf);
    if (target != DNN_TARGET_MYRIAD || getInferenceEngineVPUType() != CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        expectNoFallbacksFromIE(net);
}

TEST(Test_Caffe, multiple_inputs)
{
    const string proto = findDataFile("dnn/layers/net_input.prototxt");
    Net net = readNetFromCaffe(proto);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat first_image(10, 11, CV_32FC3);
    Mat second_image(10, 11, CV_32FC3);
    randu(first_image, -1, 1);
    randu(second_image, -1, 1);

    first_image = blobFromImage(first_image);
    second_image = blobFromImage(second_image);

    Mat first_image_blue_green = slice(first_image, Range::all(), Range(0, 2), Range::all(), Range::all());
    Mat first_image_red = slice(first_image, Range::all(), Range(2, 3), Range::all(), Range::all());
    Mat second_image_blue_green = slice(second_image, Range::all(), Range(0, 2), Range::all(), Range::all());
    Mat second_image_red = slice(second_image, Range::all(), Range(2, 3), Range::all(), Range::all());

    net.setInput(first_image_blue_green, "old_style_input_blue_green");
    net.setInput(first_image_red, "different_name_for_red");
    net.setInput(second_image_blue_green, "input_layer_blue_green");
    net.setInput(second_image_red, "old_style_input_red");
    Mat out = net.forward();

    normAssert(out, first_image + second_image);
}

TEST(Test_Caffe, shared_weights)
{
  const string proto = findDataFile("dnn/layers/shared_weights.prototxt");
  const string model = findDataFile("dnn/layers/shared_weights.caffemodel");

  Net net = readNetFromCaffe(proto, model);

  Mat input_1 = (Mat_<float>(2, 2) << 0., 2., 4., 6.);
  Mat input_2 = (Mat_<float>(2, 2) << 1., 3., 5., 7.);

  Mat blob_1 = blobFromImage(input_1);
  Mat blob_2 = blobFromImage(input_2);

  net.setInput(blob_1, "input_1");
  net.setInput(blob_2, "input_2");
  net.setPreferableBackend(DNN_BACKEND_OPENCV);

  Mat sum = net.forward();

  EXPECT_EQ(sum.at<float>(0,0), 12.);
  EXPECT_EQ(sum.at<float>(0,1), 16.);
}

typedef testing::TestWithParam<tuple<std::string, Target> > opencv_face_detector;
TEST_P(opencv_face_detector, Accuracy)
{
    std::string proto = findDataFile("dnn/opencv_face_detector.prototxt");
    std::string model = findDataFile(get<0>(GetParam()), false);
    dnn::Target targetId = (dnn::Target)(int)get<1>(GetParam());

    Net net = readNetFromCaffe(proto, model);
    Mat img = imread(findDataFile("gpu/lbpcascade/er.png"));
    Mat blob = blobFromImage(img, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    net.setInput(blob);
    // Output has shape 1x1xNx7 where N - number of detections.
    // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
    Mat out = net.forward();
    Mat ref = (Mat_<float>(6, 7) << 0, 1, 0.99520785, 0.80997437, 0.16379407, 0.87996572, 0.26685631,
                                    0, 1, 0.9934696, 0.2831718, 0.50738752, 0.345781, 0.5985168,
                                    0, 1, 0.99096733, 0.13629119, 0.24892329, 0.19756334, 0.3310290,
                                    0, 1, 0.98977017, 0.23901358, 0.09084064, 0.29902688, 0.1769477,
                                    0, 1, 0.97203469, 0.67965847, 0.06876482, 0.73999709, 0.1513494,
                                    0, 1, 0.95097077, 0.51901293, 0.45863652, 0.5777427, 0.5347801);
    normAssertDetections(ref, out, "", 0.5, 1e-5, 2e-4);
}

// False positives bug for large faces: https://github.com/opencv/opencv/issues/15106
TEST_P(opencv_face_detector, issue_15106)
{
    std::string proto = findDataFile("dnn/opencv_face_detector.prototxt");
    std::string model = findDataFile(get<0>(GetParam()), false);
    dnn::Target targetId = (dnn::Target)(int)get<1>(GetParam());

    Net net = readNetFromCaffe(proto, model);
    Mat img = imread(findDataFile("cv/shared/lena.png"));
    img = img.rowRange(img.rows / 4, 3 * img.rows / 4).colRange(img.cols / 4, 3 * img.cols / 4);
    Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    net.setInput(blob);
    // Output has shape 1x1xNx7 where N - number of detections.
    // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
    Mat out = net.forward();
    Mat ref = (Mat_<float>(1, 7) << 0, 1, 0.9149431, 0.30424616, 0.26964942, 0.88733053, 0.99815309);
    normAssertDetections(ref, out, "", 0.2, 6e-5, 1e-4);
}
INSTANTIATE_TEST_CASE_P(Test_Caffe, opencv_face_detector,
    Combine(
        Values("dnn/opencv_face_detector.caffemodel",
               "dnn/opencv_face_detector_fp16.caffemodel"),
        Values(DNN_TARGET_CPU, DNN_TARGET_OPENCL)
    )
);

TEST_P(Test_Caffe_nets, FasterRCNN_vgg16)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,  // utilizes ~1Gb, but huge blobs may not be allocated on 32-bit systems due memory fragmentation
#else
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
#endif
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

#if defined(INF_ENGINE_RELEASE)
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

    static Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.949398, 99.2454, 210.141, 601.205, 462.849,
                                           0, 7, 0.997022, 481.841, 92.3218, 722.685, 175.953,
                                           0, 12, 0.993028, 133.221, 189.377, 350.994, 563.166);
    testFaster("faster_rcnn_vgg16.prototxt", "VGG16_faster_rcnn_final.caffemodel", ref);
}

TEST_P(Test_Caffe_nets, FasterRCNN_zf)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,
#else
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
#endif
        CV_TEST_TAG_DEBUG_LONG
    );
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
    static Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.90121, 120.407, 115.83, 570.586, 528.395,
                                           0, 7, 0.988779, 469.849, 75.1756, 718.64, 186.762,
                                           0, 12, 0.967198, 138.588, 206.843, 329.766, 553.176);
    testFaster("faster_rcnn_zf.prototxt", "ZF_faster_rcnn_final.caffemodel", ref);
}

TEST_P(Test_Caffe_nets, RFCN)
{
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_2GB),
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
    double scoreDiff = (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) ? 4e-3 : default_l1;
    double iouDiff = (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) ? 8e-2 : default_lInf;
    static Mat ref = (Mat_<float>(2, 7) << 0, 7, 0.991359, 491.822, 81.1668, 702.573, 178.234,
                                           0, 12, 0.94786, 132.093, 223.903, 338.077, 566.16);
    testFaster("rfcn_pascal_voc_resnet50.prototxt", "resnet50_rfcn_final.caffemodel", ref, scoreDiff, iouDiff);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Caffe_nets, dnnBackendsAndTargets());

}} // namespace
