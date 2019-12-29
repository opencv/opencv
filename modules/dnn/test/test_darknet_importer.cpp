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
//                        (3-clause BSD License)
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
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
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_Darknet, read_tiny_yolo_voc)
{
    Net net = readNetFromDarknet(_tf("tiny-yolo-voc.cfg"));
    ASSERT_FALSE(net.empty());
}

TEST(Test_Darknet, read_yolo_voc)
{
    Net net = readNetFromDarknet(_tf("yolo-voc.cfg"));
    ASSERT_FALSE(net.empty());
}

TEST(Test_Darknet, read_yolo_voc_stream)
{
    applyTestTag(CV_TEST_TAG_MEMORY_1GB);
    Mat ref;
    Mat sample = imread(_tf("dog416.png"));
    Mat inp = blobFromImage(sample, 1.0/255, Size(416, 416), Scalar(), true, false);
    const std::string cfgFile = findDataFile("dnn/yolo-voc.cfg");
    const std::string weightsFile = findDataFile("dnn/yolo-voc.weights", false);
    // Import by paths.
    {
        Net net = readNetFromDarknet(cfgFile, weightsFile);
        net.setInput(inp);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        ref = net.forward();
    }
    // Import from bytes array.
    {
        std::vector<char> cfg, weights;
        readFileContent(cfgFile, cfg);
        readFileContent(weightsFile, weights);

        Net net = readNetFromDarknet(cfg.data(), cfg.size(), weights.data(), weights.size());
        net.setInput(inp);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        Mat out = net.forward();
        normAssert(ref, out);
    }
}

class Test_Darknet_layers : public DNNTestLayer
{
public:
    void testDarknetLayer(const std::string& name, bool hasWeights = false)
    {
        SCOPED_TRACE(name);
        Mat inp = blobFromNPY(findDataFile("dnn/darknet/" + name + "_in.npy"));
        Mat ref = blobFromNPY(findDataFile("dnn/darknet/" + name + "_out.npy"));

        std::string cfg = findDataFile("dnn/darknet/" + name + ".cfg");
        std::string model = "";
        if (hasWeights)
            model = findDataFile("dnn/darknet/" + name + ".weights", false);

        checkBackend(&inp, &ref);

        Net net = readNet(cfg, model);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(inp);
        Mat out = net.forward();
        normAssert(out, ref, "", default_l1, default_lInf);

        if (inp.size[0] == 1)  // test handling of batch size
        {
            SCOPED_TRACE("batch size 2");

#if defined(INF_ENGINE_RELEASE)
            if (target == DNN_TARGET_MYRIAD && name == "shortcut")
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

            std::vector<int> sz2 = shape(inp);
            sz2[0] = 2;

            Net net2 = readNet(cfg, model);
            net2.setPreferableBackend(backend);
            net2.setPreferableTarget(target);
            Range ranges0[4] = { Range(0, 1), Range::all(), Range::all(), Range::all() };
            Range ranges1[4] = { Range(1, 2), Range::all(), Range::all(), Range::all() };
            Mat inp2(sz2, inp.type(), Scalar::all(0));
            inp.copyTo(inp2(ranges0));
            inp.copyTo(inp2(ranges1));
            net2.setInput(inp2);
            Mat out2 = net2.forward();
            EXPECT_EQ(0, cv::norm(out2(ranges0), out2(ranges1), NORM_INF)) << "Batch result is not equal: " << name;

            Mat ref2 = ref;
            if (ref.dims == 2 && out2.dims == 3)
            {
                int ref_3d_sizes[3] = {1, ref.rows, ref.cols};
                ref2 = Mat(3, ref_3d_sizes, ref.type(), (void*)ref.data);
            }
            /*else if (ref.dims == 3 && out2.dims == 4)
            {
                int ref_4d_sizes[4] = {1, ref.size[0], ref.size[1], ref.size[2]};
                ref2 = Mat(4, ref_4d_sizes, ref.type(), (void*)ref.data);
            }*/
            ASSERT_EQ(out2.dims, ref2.dims) << ref.dims;

            normAssert(out2(ranges0), ref2, "", default_l1, default_lInf);
            normAssert(out2(ranges1), ref2, "", default_l1, default_lInf);
        }
    }
};

class Test_Darknet_nets : public DNNTestLayer
{
public:
    // Test object detection network from Darknet framework.
    void testDarknetModel(const std::string& cfg, const std::string& weights,
                          const std::vector<std::vector<int> >& refClassIds,
                          const std::vector<std::vector<float> >& refConfidences,
                          const std::vector<std::vector<Rect2d> >& refBoxes,
                          double scoreDiff, double iouDiff, float confThreshold = 0.24, float nmsThreshold = 0.4)
    {
        checkBackend();

        Mat img1 = imread(_tf("dog416.png"));
        Mat img2 = imread(_tf("street.png"));
        std::vector<Mat> samples(2);
        samples[0] = img1; samples[1] = img2;

        // determine test type, whether batch or single img
        int batch_size = refClassIds.size();
        CV_Assert(batch_size == 1 || batch_size == 2);
        samples.resize(batch_size);

        Mat inp = blobFromImages(samples, 1.0/255, Size(416, 416), Scalar(), true, false);

        Net net = readNet(findDataFile("dnn/" + cfg),
                          findDataFile("dnn/" + weights, false));
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(inp);
        std::vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        for (int b = 0; b < batch_size; ++b)
        {
            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect2d> boxes;
            for (int i = 0; i < outs.size(); ++i)
            {
                Mat out;
                if (batch_size > 1){
                    // get the sample slice from 3D matrix (batch, box, classes+5)
                    Range ranges[3] = {Range(b, b+1), Range::all(), Range::all()};
                    out = outs[i](ranges).reshape(1, outs[i].size[1]);
                }else{
                    out = outs[i];
                }
                for (int j = 0; j < out.rows; ++j)
                {
                    Mat scores = out.row(j).colRange(5, out.cols);
                    double confidence;
                    Point maxLoc;
                    minMaxLoc(scores, 0, &confidence, 0, &maxLoc);

                    if (confidence > confThreshold) {
                        float* detection = out.ptr<float>(j);
                        double centerX = detection[0];
                        double centerY = detection[1];
                        double width = detection[2];
                        double height = detection[3];
                        boxes.push_back(Rect2d(centerX - 0.5 * width, centerY - 0.5 * height,
                                            width, height));
                        confidences.push_back(confidence);
                        classIds.push_back(maxLoc.x);
                    }
                }
            }

            // here we need NMS of boxes
            std::vector<int> indices;
            NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

            std::vector<int> nms_classIds;
            std::vector<float> nms_confidences;
            std::vector<Rect2d> nms_boxes;

            for (size_t i = 0; i < indices.size(); ++i)
            {
                int idx = indices[i];
                Rect2d box = boxes[idx];
                float conf = confidences[idx];
                int class_id = classIds[idx];
                nms_boxes.push_back(box);
                nms_confidences.push_back(conf);
                nms_classIds.push_back(class_id);
            }

            normAssertDetections(refClassIds[b], refConfidences[b], refBoxes[b], nms_classIds,
                             nms_confidences, nms_boxes, format("batch size %d, sample %d\n", batch_size, b).c_str(), confThreshold, scoreDiff, iouDiff);
        }
    }

    void testDarknetModel(const std::string& cfg, const std::string& weights,
                          const std::vector<int>& refClassIds,
                          const std::vector<float>& refConfidences,
                          const std::vector<Rect2d>& refBoxes,
                          double scoreDiff, double iouDiff, float confThreshold = 0.24, float nmsThreshold = 0.4)
    {
        testDarknetModel(cfg, weights,
                         std::vector<std::vector<int> >(1, refClassIds),
                         std::vector<std::vector<float> >(1, refConfidences),
                         std::vector<std::vector<Rect2d> >(1, refBoxes),
                         scoreDiff, iouDiff, confThreshold, nmsThreshold);
    }

    void testDarknetModel(const std::string& cfg, const std::string& weights,
                          const cv::Mat& ref, double scoreDiff, double iouDiff,
                          float confThreshold = 0.24, float nmsThreshold = 0.4)
    {
        CV_Assert(ref.cols == 7);
        std::vector<std::vector<int> > refClassIds;
        std::vector<std::vector<float> > refScores;
        std::vector<std::vector<Rect2d> > refBoxes;
        for (int i = 0; i < ref.rows; ++i)
        {
            int batchId = static_cast<int>(ref.at<float>(i, 0));
            int classId = static_cast<int>(ref.at<float>(i, 1));
            float score = ref.at<float>(i, 2);
            float left  = ref.at<float>(i, 3);
            float top   = ref.at<float>(i, 4);
            float right  = ref.at<float>(i, 5);
            float bottom = ref.at<float>(i, 6);
            Rect2d box(left, top, right - left, bottom - top);
            if (batchId >= refClassIds.size())
            {
                refClassIds.resize(batchId + 1);
                refScores.resize(batchId + 1);
                refBoxes.resize(batchId + 1);
            }
            refClassIds[batchId].push_back(classId);
            refScores[batchId].push_back(score);
            refBoxes[batchId].push_back(box);
        }
        testDarknetModel(cfg, weights, refClassIds, refScores, refBoxes,
                         scoreDiff, iouDiff, confThreshold, nmsThreshold);
    }
};

TEST_P(Test_Darknet_nets, YoloVoc)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_MEMORY_1GB);

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
#endif
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);  // need to update check function
#endif

    // batchId, classId, confidence, left, top, right, bottom
    Mat ref = (Mat_<float>(6, 7) << 0, 6,  0.750469f, 0.577374f, 0.127391f, 0.902949f, 0.300809f,  // a car
                                    0, 1,  0.780879f, 0.270762f, 0.264102f, 0.732475f, 0.745412f,  // a bicycle
                                    0, 11, 0.901615f, 0.1386f,   0.338509f, 0.421337f, 0.938789f,  // a dog
                                    1, 14, 0.623813f, 0.183179f, 0.381921f, 0.247726f, 0.625847f,  // a person
                                    1, 6,  0.667770f, 0.446555f, 0.453578f, 0.499986f, 0.519167f,  // a car
                                    1, 6,  0.844947f, 0.637058f, 0.460398f, 0.828508f, 0.66427f);  // a car

    double nmsThreshold = (target == DNN_TARGET_MYRIAD) ? 0.397 : 0.4;
    double scoreDiff = 8e-5, iouDiff = 3e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 1e-2;
        iouDiff = 0.018;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.03;
        iouDiff = 0.018;
    }

    std::string config_file = "yolo-voc.cfg";
    std::string weights_file = "yolo-voc.weights";

    {
    SCOPED_TRACE("batch size 1");
    testDarknetModel(config_file, weights_file, ref.rowRange(0, 3), scoreDiff, iouDiff);
    }

    {
    SCOPED_TRACE("batch size 2");
    testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff, 0.24, nmsThreshold);
    }
}

TEST_P(Test_Darknet_nets, TinyYoloVoc)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);  // need to update check function
#endif
    // batchId, classId, confidence, left, top, right, bottom
    Mat ref = (Mat_<float>(4, 7) << 0, 6,  0.761967f, 0.579042f, 0.159161f, 0.894482f, 0.31994f,   // a car
                                    0, 11, 0.780595f, 0.129696f, 0.386467f, 0.445275f, 0.920994f,  // a dog
                                    1, 6,  0.651450f, 0.460526f, 0.458019f, 0.522527f, 0.5341f,    // a car
                                    1, 6,  0.928758f, 0.651024f, 0.463539f, 0.823784f, 0.654998f); // a car

    double scoreDiff = 8e-5, iouDiff = 3e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 8e-3;
        iouDiff = 0.018;
    }
    else if(target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.008;
        iouDiff = 0.02;
    }

    std::string config_file = "tiny-yolo-voc.cfg";
    std::string weights_file = "tiny-yolo-voc.weights";

    {
    SCOPED_TRACE("batch size 1");
    testDarknetModel(config_file, weights_file, ref.rowRange(0, 2), scoreDiff, iouDiff);
    }

    {
    SCOPED_TRACE("batch size 2");
    testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff);
    }
}

#ifdef HAVE_INF_ENGINE
static const std::chrono::milliseconds async_timeout(10000);

typedef testing::TestWithParam<tuple<std::string, tuple<Backend, Target> > > Test_Darknet_nets_async;
TEST_P(Test_Darknet_nets_async, Accuracy)
{
    Backend backendId = get<0>(get<1>(GetParam()));
    Target targetId = get<1>(get<1>(GetParam()));

    if (INF_ENGINE_VER_MAJOR_LT(2019020000) && backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    std::string prefix = get<0>(GetParam());

    if (backendId != DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && backendId != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        throw SkipTestException("No support for async forward");

    const int numInputs = 2;
    std::vector<Mat> inputs(numInputs);
    int blobSize[] = {1, 3, 416, 416};
    for (int i = 0; i < numInputs; ++i)
    {
        inputs[i].create(4, &blobSize[0], CV_32F);
        randu(inputs[i], 0, 1);
    }

    Net netSync = readNet(findDataFile("dnn/" + prefix + ".cfg"),
                          findDataFile("dnn/" + prefix + ".weights", false));
    netSync.setPreferableBackend(backendId);
    netSync.setPreferableTarget(targetId);

    // Run synchronously.
    std::vector<Mat> refs(numInputs);
    for (int i = 0; i < numInputs; ++i)
    {
        netSync.setInput(inputs[i]);
        refs[i] = netSync.forward().clone();
    }

    Net netAsync = readNet(findDataFile("dnn/" + prefix + ".cfg"),
                           findDataFile("dnn/" + prefix + ".weights", false));
    netAsync.setPreferableBackend(backendId);
    netAsync.setPreferableTarget(targetId);

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

INSTANTIATE_TEST_CASE_P(/**/, Test_Darknet_nets_async, Combine(
    Values("yolo-voc", "tiny-yolo-voc", "yolov3"),
    dnnBackendsAndTargets()
));

#endif

TEST_P(Test_Darknet_nets, YOLOv3)
{
    applyTestTag(CV_TEST_TAG_LONG, (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB));

    // batchId, classId, confidence, left, top, right, bottom
    Mat ref = (Mat_<float>(9, 7) << 0, 7,  0.952983f, 0.614622f, 0.150257f, 0.901369f, 0.289251f,  // a truck
                                    0, 1,  0.987908f, 0.150913f, 0.221933f, 0.742255f, 0.74626f,   // a bicycle
                                    0, 16, 0.998836f, 0.160024f, 0.389964f, 0.417885f, 0.943716f,  // a dog (COCO)
                                    1, 9,  0.384801f, 0.659824f, 0.372389f, 0.673926f, 0.429412f,  // a traffic light
                                    1, 9,  0.733283f, 0.376029f, 0.315694f, 0.401776f, 0.395165f,  // a traffic light
                                    1, 9,  0.785352f, 0.665503f, 0.373543f, 0.688893f, 0.439245f,  // a traffic light
                                    1, 0,  0.980052f, 0.195856f, 0.378454f, 0.258626f, 0.629258f,  // a person
                                    1, 2,  0.989633f, 0.450719f, 0.463353f, 0.496305f, 0.522258f,  // a car
                                    1, 2,  0.997412f, 0.647584f, 0.459939f, 0.821038f, 0.663947f); // a car

    double scoreDiff = 8e-5, iouDiff = 3e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.006;
        iouDiff = 0.018;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.04;
        iouDiff = 0.03;
    }
    std::string config_file = "yolov3.cfg";
    std::string weights_file = "yolov3.weights";

#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD &&
        getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
    {
        scoreDiff = 0.04;
        iouDiff = 0.2;
    }
#endif

    {
    SCOPED_TRACE("batch size 1");
    testDarknetModel(config_file, weights_file, ref.rowRange(0, 3), scoreDiff, iouDiff);
    }

#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (INF_ENGINE_VER_MAJOR_LE(2018050000) && target == DNN_TARGET_OPENCL)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
        else if (INF_ENGINE_VER_MAJOR_EQ(2019020000))
        {
            if (target == DNN_TARGET_OPENCL)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
            if (target == DNN_TARGET_OPENCL_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
        }
        else if (target == DNN_TARGET_MYRIAD &&
                 getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
    }
#endif

    {
        SCOPED_TRACE("batch size 2");
        testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Darknet_nets, dnnBackendsAndTargets());

TEST_P(Test_Darknet_layers, shortcut)
{
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testDarknetLayer("shortcut");
    testDarknetLayer("shortcut_leaky");
    testDarknetLayer("shortcut_unequal");
    testDarknetLayer("shortcut_unequal_2");
}

TEST_P(Test_Darknet_layers, upsample)
{
    testDarknetLayer("upsample");
}

TEST_P(Test_Darknet_layers, avgpool_softmax)
{
    testDarknetLayer("avgpool_softmax");
}

TEST_P(Test_Darknet_layers, region)
{
    testDarknetLayer("region");
}

TEST_P(Test_Darknet_layers, reorg)
{
    testDarknetLayer("reorg");
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Darknet_layers, dnnBackendsAndTargets());

}} // namespace
