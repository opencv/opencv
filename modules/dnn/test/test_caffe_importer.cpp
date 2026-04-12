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
#include <set>

namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    return findDataFile(std::string("dnn/") + filename);
}

static std::string getCurrentTestNameNoParams()
{
    const testing::TestInfo* info = testing::UnitTest::GetInstance()->current_test_info();
    if (!info) return std::string();

#if defined(GTEST_VERSION_MAJOR) && (GTEST_VERSION_MAJOR > 1 || (GTEST_VERSION_MAJOR == 1 && GTEST_VERSION_MINOR >= 10))
    const char* suite = info->test_suite_name();
#else
    const char* suite = info->test_case_name();
#endif

    std::string name = std::string(suite ? suite : "") + "." + info->name();
    const size_t pos = name.find('/');
    if (pos != std::string::npos)
        name.resize(pos);
    return name;
}

static const std::set<std::string>& getCaffeNewEngineDenylist()
{
    static std::set<std::string> deny;
    static bool isInitialized = false;
    if (!isInitialized)
    {
        const std::vector<std::string> items = {
            #include "test_caffe_importer_new_engine_denylist.inl.hpp"
        };
        deny.insert(items.begin(), items.end());
        isInitialized = true;
    }
    return deny;
}

static void skipIfInCaffeNewEngineDenylist()
{
    const std::string name = getCurrentTestNameNoParams();
    if (!name.empty() && getCaffeNewEngineDenylist().count(name))
        throw SkipTestException("Test is in the new engine denylist: " + name);
}

class Test_Caffe_nets : public DNNTestLayer
{
public:
    void SetUp() CV_OVERRIDE
    {
        skipIfInCaffeNewEngineDenylist();
        DNNTestLayer::SetUp();
    }

    void testFaster(const std::string& proto, const std::string& model, const Mat& ref,
                    double scoreDiff = 0.0, double iouDiff = 0.0)
    {
        checkBackend();
        Net net = readNet(findDataFile("dnn/" + proto),
                          findDataFile("dnn/" + model, false));
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        if (target == DNN_TARGET_CPU_FP16)
            net.enableWinograd(false);

        Mat img = imread(findDataFile("dnn/dog416.png"));
        resize(img, img, Size(800, 600));
        Mat blob = blobFromImage(img, 1.0, Size(), Scalar(102.9801, 115.9465, 122.7717), false, false);
        Mat imInfo = (Mat_<float>(1, 3) << img.rows, img.cols, 1.6f);

        net.setInput(blob);
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
    skipIfInCaffeNewEngineDenylist();
    const string proto = findDataFile("dnn/bvlc_googlenet.prototxt");
    const string model = findDataFile("dnn/bvlc_googlenet.caffemodel", false);

    std::vector<char> dataProto;
    readFileContent(proto, dataProto);
    std::vector<uchar> vecProto(dataProto.begin(), dataProto.end());

    std::vector<char> dataModel;
    readFileContent(model, dataModel);
    std::vector<uchar> vecModel(dataModel.begin(), dataModel.end());

    Net net = readNet("caffe", std::vector<uchar>(), vecProto);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    ASSERT_FALSE(net.empty());

    Net net2 = readNet("caffe", vecModel, vecProto);
    ASSERT_FALSE(net2.empty());
}

TEST(Test_Caffe, read_gtsrb)
{
    skipIfInCaffeNewEngineDenylist();
    Net net = readNet(_tf("gtsrb.prototxt"), "", "caffe");
    ASSERT_FALSE(net.empty());
}

TEST(Test_Caffe, read_googlenet)
{
    skipIfInCaffeNewEngineDenylist();
    Net net = readNet(_tf("bvlc_googlenet.prototxt"), "", "caffe");
    ASSERT_FALSE(net.empty());
}

TEST_P(Test_Caffe_nets, Axpy)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    String proto = _tf("axpy.prototxt");
    Net net = readNet(proto);

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
    net.setInput(inp);

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
    float l1 = 1e-5, lInf = 1e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 2e-4;
        lInf = 1e-3;
    }
    if (target == DNN_TARGET_MYRIAD)
    {
        l1 = 0.001;
        lInf = 0.001;
    }
    if(target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.0002;
        lInf = 0.0007;
    }
    normAssert(ref, out, "", l1, lInf);
}

TEST(Reproducibility_FCN, Accuracy)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_VERYLONG, CV_TEST_TAG_MEMORY_2GB);

    Net net = readNetFromONNX(findDataFile("dnn/fcn8s.onnx", false));
    ASSERT_FALSE(net.empty());
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    std::vector<int> layerIds;
    std::vector<size_t> weights, blobs;
    net.getMemoryConsumption(shape(1,3,227,227), CV_32F, layerIds, weights, blobs);

    net.setInput(blobFromImage(sample, 1.0f, Size(500, 500), Scalar(), false));

    Mat out = net.forward();

    Mat refData = imread(_tf("caffe_fcn8s_prob.png"), IMREAD_ANYDEPTH);
    int shape[] = {1, 21, 500, 500};
    Mat ref(4, shape, CV_32FC1, refData.data);

    normAssert(ref, out);
}

TEST(Reproducibility_SSD, Accuracy)
{
    skipIfInCaffeNewEngineDenylist();
    applyTestTag(
        CV_TEST_TAG_MEMORY_512MB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    Net net = readNetFromONNX(findDataFile("dnn/ssd_vgg16.onnx", false));
    ASSERT_FALSE(net.empty());
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    if (sample.channels() == 4)
        cvtColor(sample, sample, COLOR_BGRA2BGR);

    Mat in_blob = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    net.setInput(in_blob);

    Mat out = net.forward();

    Mat ref = blobFromNPY(_tf("ssd_out.npy"));
    normAssertDetections(ref, out, "", 0.06);
}

typedef testing::TestWithParam<tuple<Backend, Target> > Reproducibility_MobileNet_SSD;
TEST_P(Reproducibility_MobileNet_SSD, Accuracy)
{
    skipIfInCaffeNewEngineDenylist();
    const string model = findDataFile("dnn/ssd_mobilenet_v1_12.onnx", false);
    Net net = readNetFromONNX(model);
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    Mat sample = imread(_tf("street.png"));

    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inp);
    Mat out = net.forward().clone();

    ASSERT_EQ(out.size[2], 100);

    float scores_diff = 1e-5, boxes_iou_diff = 1e-4;
    if (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD || targetId == DNN_TARGET_CPU_FP16)
    {
        scores_diff = 1.5e-2;
        boxes_iou_diff = 6.3e-2;
    }
    else if (targetId == DNN_TARGET_CUDA_FP16)
    {
        scores_diff = 0.015;
        boxes_iou_diff = 0.07;
    }
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

// https://github.com/richzhang/colorization
TEST_P(Test_Caffe_nets, Colorization)
{
    applyTestTag(
        target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );
    checkBackend();

    Mat inp = blobFromNPY(_tf("colorization_inp.npy"));
    Mat ref = blobFromNPY(_tf("colorization_out.npy"));
    Mat kernel = blobFromNPY(_tf("colorization_pts_in_hull.npy"));

    const string model = findDataFile("dnn/colorization_deploy_v2.onnx", false);
    Net net = readNetFromONNX(model);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    // This model has bad accuracy when the FP16 and Winograd are enable at same time.
    if (target == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

#ifdef ENGINE_CLASSIC
    net.getLayer(net.getLayerId("class8_ab"))->blobs.push_back(kernel);
    net.getLayer(net.getLayerId("conv8_313_rh"))->blobs.push_back(Mat(1, 313, CV_32F, 2.606));
#endif

    net.setInput(inp);
    Mat out = net.forward();

    // Reference output values are in range [-29.1, 69.5]
    double l1 = 4e-4, lInf = 3e-3;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 0.25;
        lInf = 5.3;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        l1 = (getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X) ? 0.5 : 0.25;
        lInf = (getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X) ? 11 : 5.3;
    }
    else if(target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.21;
        lInf = 4.5;
    }
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.3; lInf = 10;
    }
#endif

    normAssert(out, ref, "", l1, lInf);
    expectNoFallbacksFromIE(net);
}

TEST(Test_Caffe, multiple_inputs)
{
    skipIfInCaffeNewEngineDenylist();
    const string proto = findDataFile("dnn/layers/net_input.prototxt");
    Net net = readNet(proto);
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
  skipIfInCaffeNewEngineDenylist();
  const string proto = findDataFile("dnn/layers/shared_weights.prototxt");
  const string model = findDataFile("dnn/layers/shared_weights.caffemodel");

  Net net = readNet(proto, model);

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

    if (targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (targetId == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

    Net net = readNet(proto, model);
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
    normAssertDetections(ref, out, "", 0.5, 1e-4, 2e-4);
}

// False positives bug for large faces: https://github.com/opencv/opencv/issues/15106
TEST_P(opencv_face_detector, issue_15106)
{
    std::string proto = findDataFile("dnn/opencv_face_detector.prototxt");
    std::string model = findDataFile(get<0>(GetParam()), false);
    dnn::Target targetId = (dnn::Target)(int)get<1>(GetParam());

    if (targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (targetId == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

    Net net = readNet(proto, model);
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
    normAssertDetections(ref, out, "", 0.89, 6e-5, 1e-4);
}
INSTANTIATE_TEST_CASE_P(Test_Caffe, opencv_face_detector,
    Combine(
        Values("dnn/opencv_face_detector.caffemodel",
               "dnn/opencv_face_detector_fp16.caffemodel"),
        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV))
    )
);

TEST_P(Test_Caffe_nets, FasterRCNN_vgg16)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,  // utilizes ~1Gb, but huge blobs may not be allocated on 32-bit systems due memory fragmentation
#else
        CV_TEST_TAG_MEMORY_2GB,
#endif
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Ngraph operation Reshape with name rpn_cls_score_reshape has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:390:
    // While validating node 'v1::Reshape bbox_pred_reshape (bbox_pred[0]:f32{1,84}, Constant_241202[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
    // Requested output shape {1,6300,4,1} is incompatible with input shape Shape{1, 84}
    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    double scoreDiff = 0.0012, iouDiff = 0.03;
#if defined(INF_ENGINE_RELEASE)
    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
        iouDiff = 0.02;
        if (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16) {
            scoreDiff = 0.04;
            iouDiff = 0.06;
        }
    }
#endif

    static Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.949398, 99.2454, 210.141, 601.205, 462.849,
                                           0, 7, 0.997022, 481.841, 92.3218, 722.685, 175.953,
                                           0, 12, 0.993028, 133.221, 189.377, 350.994, 563.166);
    testFaster("faster_rcnn_vgg16.prototxt", "VGG16_faster_rcnn_final.caffemodel", ref, scoreDiff, iouDiff);
}

TEST_P(Test_Caffe_nets, FasterRCNN_zf)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,
#else
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
#endif
        CV_TEST_TAG_DEBUG_VERYLONG
    );
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Ngraph operation Reshape with name rpn_cls_score_reshape has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif

    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
    if (target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
    static Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.90121, 120.407, 115.83, 570.586, 528.395,
                                           0, 7, 0.988779, 469.849, 75.1756, 718.64, 186.762,
                                           0, 12, 0.967198, 138.588, 206.843, 329.766, 553.176);

    double scoreDiff = 0.003, iouDiff = 0.07;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
        scoreDiff = 0.02;
        iouDiff = 0.13;
    }

    testFaster("faster_rcnn_zf.prototxt", "ZF_faster_rcnn_final.caffemodel", ref, scoreDiff, iouDiff);
}

TEST_P(Test_Caffe_nets, RFCN)
{
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_2GB),
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    float scoreDiff = default_l1, iouDiff = default_lInf;
    if (backend == DNN_BACKEND_OPENCV && (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16))
    {
        scoreDiff = 4e-3;
        iouDiff = 8e-2;
    }
    if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.0034;
        iouDiff = 0.12;
    }

#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        scoreDiff = 0.1f;
        iouDiff = 0.2f;
    }

    // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:427:
    // While validating node 'v1::Reshape bbox_pred_reshape (ave_bbox_pred_rois[0]:f32{1,8,1,1}, Constant_388[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
    // Requested output shape {1,300,8,1} is incompatible with input shape {1, 8, 1, 1}
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Exception: Function contains several inputs and outputs with one friendly name! (HETERO bug?)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

    static Mat ref = (Mat_<float>(2, 7) << 0, 7, 0.991359, 491.822, 81.1668, 702.573, 178.234,
                                           0, 12, 0.94786, 132.093, 223.903, 338.077, 566.16);
    testFaster("rfcn_pascal_voc_resnet50.prototxt", "resnet50_rfcn_final.caffemodel", ref, scoreDiff, iouDiff);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Caffe_nets, dnnBackendsAndTargets());

}} // namespace
