// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#include <cstring>

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

struct OclGemmErrorCapture
{
    bool cTypeMismatch;

    OclGemmErrorCapture() : cTypeMismatch(false) {}
};

static int captureOclGemmError(int, const char* funcName, const char* errorMessage,
                               const char*, int, void* userdata)
{
    OclGemmErrorCapture* capture = static_cast<OclGemmErrorCapture*>(userdata);
    if (capture && funcName && errorMessage &&
        std::strstr(funcName, "ocl_gemm") &&
        std::strstr(errorMessage, "type == matC.type()"))
    {
        capture->cTypeMismatch = true;
    }
    return 0;
}

class ScopedErrorCallback
{
public:
    ScopedErrorCallback(cv::ErrorCallback callback, void* userdata)
        : previousCallback(cv::redirectError(callback, userdata, &previousUserdata))
    {
    }

    ~ScopedErrorCallback()
    {
        cv::redirectError(previousCallback, previousUserdata);
    }

private:
    cv::ErrorCallback previousCallback;
    void* previousUserdata;
};

class ScopedOpenCLState
{
public:
    explicit ScopedOpenCLState(bool enabled) : previousState(cv::ocl::useOpenCL())
    {
        cv::ocl::setUseOpenCL(enabled);
    }

    ~ScopedOpenCLState()
    {
        cv::ocl::setUseOpenCL(previousState);
    }

private:
    bool previousState;
};

OCL_TEST(PCA, project_empty_C_beta_zero)
{
    cv::ocl::Context context = cv::ocl::Context::getDefault();
    if (!context.ptr())
        throw cvtest::SkipTestException("OpenCL is not available");

    cv::ocl::Device device = cv::ocl::Device::getDefault();
    if (!device.compilerAvailable())
        throw cvtest::SkipTestException("OpenCL compiler is not available");

    cv::Mat samples(32, 8, CV_32FC1);
    cv::RNG rng(12345);
    rng.fill(samples, cv::RNG::UNIFORM, -1.0f, 1.0f);

    ScopedOpenCLState openclState(false);
    const int flags[] = { cv::PCA::DATA_AS_ROW, cv::PCA::DATA_AS_COL };
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); ++i)
    {
        SCOPED_TRACE(flags[i] == cv::PCA::DATA_AS_ROW ? "DATA_AS_ROW" : "DATA_AS_COL");
        cv::Mat pcaData = samples;
        if (flags[i] == cv::PCA::DATA_AS_COL)
            pcaData = samples.t();
        cv::PCA pca(pcaData, cv::Mat(), flags[i], 8);

        cv::Mat expected;
        pca.project(pcaData, expected);

        cv::UMat actual;
        OclGemmErrorCapture errorCapture;
        {
            ScopedOpenCLState useOpenCL(true);
            ScopedErrorCallback errorCallback(captureOclGemmError, &errorCapture);
            pca.project(pcaData, actual);
        }

        EXPECT_FALSE(errorCapture.cTypeMismatch)
            << "OpenCL GEMM treated the absent C input as CV_8UC1";
        EXPECT_LE(cvtest::norm(expected, actual, cv::NORM_INF), 1e-5);
    }
}

} // namespace ocl
} // namespace opencv_test

#endif // HAVE_OPENCL
