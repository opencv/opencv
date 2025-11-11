// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "npy_blob.hpp"

namespace opencv_test { namespace {

TEST(DNN_CUDA, AlexNet_ONNX_GroundTruth)
{
#ifndef HAVE_CUDA
    throw SkipTestException("OpenCV is built without CUDA support");
#else
    const std::string modelname = findDataFile("dnn/onnx/models/alexnet_trimmed_conv.onnx", true);
    const std::string image_path = findDataFile("dnn/googlenet_0.png", true);
    const std::string output_path = findDataFile("dnn/alexnet_gt.npy", true);

    int input_size = 224;
    Mat output_gt = blobFromNPY(output_path);

    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
        throw SkipTestException("Failed to read image: " + image_path);

    const Scalar mean_scaled(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0);
    Mat blob = dnn::blobFromImage(
        img, 1.0/255.0, Size(input_size, input_size), mean_scaled, true, false
    );
    const float std_vals[3]  = {0.229f, 0.224f, 0.225f};
    CV_Assert(blob.dims == 4 && blob.size[0] == 1 && blob.size[1] == 3);
    for (int c = 0; c < 3; ++c)
    {
        Range ranges[4] = { Range(0,1), Range(c, c+1), Range::all(), Range::all() };
        Mat ch = blob(ranges);
        ch /= std_vals[c];
    }

    dnn::Net net = dnn::readNetFromONNX(modelname, ENGINE_NEW);
    net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
    net.setInput(blob);
    net.setTracingMode(cv::dnn::DNN_TRACE_OP);  // or cv.dnn.DNN_TRACE_ALL

    Mat out = net.forward().clone();
    out = out.reshape(1, 1);

    if (out.type() != CV_32F)
        out.convertTo(out, CV_32F);
    output_gt = output_gt.reshape(1, 1);
    if (output_gt.type() != CV_32F)
        output_gt.convertTo(output_gt, CV_32F);

    ASSERT_EQ(out.total(), output_gt.total()) << "Mismatched output sizes";

    const double l1_thr = 1e-4;
    const double lInf_thr = 1e-3;
    normAssert(output_gt, out, "AlexNet_ONNX_GroundTruth", l1_thr, lInf_thr);
#endif
}

}} // namespace
