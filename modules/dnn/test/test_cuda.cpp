// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <numeric>
#include <numeric>
namespace opencv_test { namespace {

static cv::Mat softmaxRow(const cv::Mat& logitsRow)
{
    CV_Assert(logitsRow.rows == 1 && logitsRow.type() == CV_32F);
    double maxVal;
    cv::minMaxLoc(logitsRow, nullptr, &maxVal);
    cv::Mat shifted;
    logitsRow.convertTo(shifted, CV_32F);
    shifted -= (float)maxVal;
    cv::Mat exps;
    cv::exp(shifted, exps);
    double sumVal = cv::sum(exps)[0];
    exps /= (float)sumVal;
    return exps;
}

static std::vector<int> topkIdx(const cv::Mat& row, int k)
{
    CV_Assert(row.rows == 1 && row.type() == CV_32F);
    const int n = (int)row.cols;
    k = std::min(k, n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    const float* p = row.ptr<float>(0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b){ return p[a] > p[b]; });
    idx.resize(k);
    return idx;
}

TEST(DNN_CUDA, AlexNet_ONNX_GroundTruth)
{
#ifndef HAVE_CUDA
    throw SkipTestException("OpenCV is built without CUDA support");
#else
    const char* yaml_env = std::getenv("ALEXNET_YAML");
    if (!yaml_env || !*yaml_env)
        throw SkipTestException("ALEXNET_YAML environment variable is not set");

    std::string yaml_path(yaml_env);

    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw SkipTestException("Failed to open YAML: " + yaml_path);

    std::string model_path;
    std::string image_path;
    int input_size = 224;
    cv::Mat output_gt; // (1, N) float32

    fs["model"] >> model_path;
    fs["image"] >> image_path;
    fs["size"] >> input_size;
    fs["output"] >> output_gt;
    fs.release();

    if (model_path.empty() || image_path.empty() || output_gt.empty())
        throw SkipTestException("YAML must contain keys: model, image, size, output");

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
        throw SkipTestException("Failed to read image: " + image_path);

    // Preprocess: scale 1/255, resize, swapRB, no mean subtraction here
    cv::Mat blob = cv::dnn::blobFromImage(
        img, 1.0/255.0, cv::Size(input_size, input_size), cv::Scalar(0,0,0), true, false
    ); // NCHW float32 (1,3,H,W)

    // ImageNet normalization per-channel: (x - mean) / std
    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    const float std_vals[3]  = {0.229f, 0.224f, 0.225f};
    CV_Assert(blob.dims == 4 && blob.size[0] == 1 && blob.size[1] == 3);
    const int H = blob.size[2];
    const int W = blob.size[3];
    float* blob_ptr = reinterpret_cast<float*>(blob.data);
    const int stride = H * W;
    for (int c = 0; c < 3; ++c)
    {
        float m = mean_vals[c];
        float s = std_vals[c];
        float* p = blob_ptr + c * stride;
        for (int i = 0; i < stride; ++i)
        {
            p[i] = (p[i] - m) / s;
        }
    }

    cv::dnn::Net net = cv::dnn::readNet(model_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net.setInput(blob);

    // Warm-up
    (void)net.forward();

    cv::Mat out = net.forward().clone();
    out = out.reshape(1, 1); // 1 x N

    // Ensure same type and shape as ground-truth
    if (out.type() != CV_32F)
        out.convertTo(out, CV_32F);
    output_gt = output_gt.reshape(1, 1);
    if (output_gt.type() != CV_32F)
        output_gt.convertTo(output_gt, CV_32F);

    ASSERT_EQ(out.total(), output_gt.total()) << "Mismatched output sizes";

    // Compare probabilities to be robust to presence/absence of a final Softmax layer
    cv::Mat probs_gt = softmaxRow(output_gt);
    cv::Mat probs_cv = softmaxRow(out);

    const double l1_thr = 5e-4;
    const double lInf_thr = 5e-3;
    cv::Mat diff = probs_gt - probs_cv;
    double l1 = cv::norm(diff, cv::NORM_L1) / (double)diff.total();
    double lInf = cv::norm(diff, cv::NORM_INF);
    if (l1 <= l1_thr && lInf <= lInf_thr)
    {
        SUCCEED();
    }
    else
    {
        // Fallback: compare Top-5 classes
        std::vector<int> gt_top5 = topkIdx(probs_gt, 5);
        std::vector<int> cv_top5 = topkIdx(probs_cv, 5);
        bool top5_eq = gt_top5.size() == cv_top5.size() && std::equal(gt_top5.begin(), gt_top5.end(), cv_top5.begin());
        if (!top5_eq)
        {
            // Final assertion with details to fail the test
            ADD_FAILURE() << "Probability diff too large (l1=" << l1 << ", lInf=" << lInf
                          << "), and Top-5 mismatch.";
        }
    }
#endif
}

}} // namespace
