// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <cstdlib>

namespace opencv_test { namespace {

// Sets an environment variable for the current scope and restores it afterwards.
class ScopedEnv
{
public:
    ScopedEnv(const char* name, const char* value) : name_(name)
    {
        const char* old = getenv(name);
        hadOld_ = old != nullptr;
        if (hadOld_)
            old_ = old;
        set(value);
    }
    ~ScopedEnv()
    {
        if (hadOld_)
            set(old_.c_str());
        else
            unset();
    }
private:
    void set(const char* value)
    {
#ifdef _WIN32
        _putenv_s(name_.c_str(), value);
#else
        setenv(name_.c_str(), value, 1);
#endif
    }
    void unset()
    {
#ifdef _WIN32
        _putenv_s(name_.c_str(), "");
#else
        unsetenv(name_.c_str());
#endif
    }
    std::string name_, old_;
    bool hadOld_;
};

// "1" forces the im2col + GEMM conv path on any AArch64 CPU, "0" turns it off.
static const char* const kConvGemmSwitch = "OPENCV_DNN_CONV_GEMM";

static void skipIfNotArm()
{
#if !(defined(__aarch64__) || defined(_M_ARM64))
    throw SkipTestException("the im2col + GEMM conv path is built for AArch64 only");
#endif
}

static std::vector<Mat> runModel(const std::string& path, const Mat& input, const char* convGemm, int nthreads = -1)
{
    ScopedEnv env(kConvGemmSwitch, convGemm);
    const int savedThreads = getNumThreads();
    setNumThreads(nthreads);
    Net net = readNetFromONNX(path, ENGINE_OPENCV);
    CV_Assert(!net.empty());
    net.setInput(input);
    std::vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    for (Mat& m : outs)
        m = m.clone();
    setNumThreads(savedThreads);
    return outs;
}

// Conv models with reference outputs, run with the path forced on. Their odd sizes
// (Cin 2/3, Cout 3-6, asymmetric pads, strides) cover the channel tails of NCHWc8.
typedef testing::TestWithParam<std::string> Test_Conv2Gemm_Reference;
TEST_P(Test_Conv2Gemm_Reference, Accuracy)
{
    skipIfNotArm();
    const std::string name = GetParam();
    Mat input = blobFromNPY(findDataFile("dnn/onnx/data/input_" + name + ".npy"));
    Mat ref = blobFromNPY(findDataFile("dnn/onnx/data/output_" + name + ".npy"));
    std::vector<Mat> outs = runModel(findDataFile("dnn/onnx/models/" + name + ".onnx"), input, "1");
    ASSERT_EQ(outs.size(), 1u);
    normAssert(ref, outs[0], name.c_str());
}
INSTANTIATE_TEST_CASE_P(/**/, Test_Conv2Gemm_Reference, testing::Values(
    "convolution", "conv_asymmetric_pads", "two_convolution",
    "batchnorm_conv_dense", "batchnorm_conv_1x1_mlas"));

// Models whose dense convs take a fused ReLU, Sigmoid or residual Add: the path forced
// on must match the NCHWc kernels (path off) on the same random input. One thread and a
// batch of 2 also cover the few-pixels branch and the per-image offsets.
typedef testing::TestWithParam<tuple<std::string, std::vector<int>, int> > Test_Conv2Gemm_OnOff;
TEST_P(Test_Conv2Gemm_OnOff, Accuracy)
{
    skipIfNotArm();
    const std::string name = get<0>(GetParam());
    const std::vector<int> shape = get<1>(GetParam());
    const int nthreads = get<2>(GetParam());
    const std::string path = findDataFile("dnn/onnx/models/" + name + ".onnx");
    Mat input(shape, CV_32F);
    RNG& rng = theRNG();
    rng.fill(input, RNG::UNIFORM, 0.f, 1.f);

    std::vector<Mat> on = runModel(path, input, "1", nthreads);
    std::vector<Mat> off = runModel(path, input, "0", nthreads);
    ASSERT_EQ(on.size(), off.size());
    for (size_t i = 0; i < on.size(); i++) {
        const double scale = std::max(1.0, cvtest::norm(off[i], NORM_INF));
        EXPECT_LE(cvtest::norm(on[i], off[i], NORM_INF), 1e-4 * scale) << name << " output " << i;
    }
}
INSTANTIATE_TEST_CASE_P(/**/, Test_Conv2Gemm_OnOff, testing::Values(
    make_tuple(std::string("constant"), std::vector<int>{1, 3, 32, 32}, -1),
    make_tuple(std::string("constant"), std::vector<int>{2, 3, 32, 32}, -1),
    make_tuple(std::string("mobilenetv4"), std::vector<int>{1, 3, 224, 224}, -1),
    make_tuple(std::string("mobilenetv4"), std::vector<int>{1, 3, 224, 224}, 1),
    make_tuple(std::string("xfeat"), std::vector<int>{1, 1, 640, 640}, -1)));

}}  // namespace
