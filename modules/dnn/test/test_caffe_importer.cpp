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

static const std::set<std::string>& getCaffeNewEngineDenylist()
{
    static const std::set<std::string> denyList = {
        #include "test_caffe_importer_new_engine_denylist.inl.hpp"
    };
    return denyList;
}

static void skipIfInCaffeNewEngineDenylist()
{
    const std::string name = opencv_test::getCurrentTestNameNoParams();
    if (!name.empty() && getCaffeNewEngineDenylist().count(name))
        throw SkipTestException("Test is in the new engine denylist: " + name);
}

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

TEST(Reproducibility_SSD, Accuracy)
{
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
    normAssertDetections(ref, out, "", 0.06, 1e-4, 0.18);
}

TEST(Test_Caffe, multiple_inputs)
{
    const string model = findDataFile("dnn/layers/net_input.onnx");
    Net net = readNetFromONNX(model);
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

typedef testing::TestWithParam<tuple<std::string, Target> > opencv_face_detector;
TEST_P(opencv_face_detector, Accuracy)
{
    std::string model = findDataFile(get<0>(GetParam()), false);
    dnn::Target targetId = (dnn::Target)(int)get<1>(GetParam());

    if (targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (targetId == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

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
    std::string model = findDataFile(get<0>(GetParam()), false);
    dnn::Target targetId = (dnn::Target)(int)get<1>(GetParam());

    if (targetId == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (targetId == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

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
        Values("dnn/onnx/models/opencv_face_detector.onnx",
               "dnn/onnx/models/opencv_face_detector_fp16.onnx"),
        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV))
    )
);

INSTANTIATE_TEST_CASE_P(/**/, Test_Caffe_nets, dnnBackendsAndTargets());

}} // namespace
