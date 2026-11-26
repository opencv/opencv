// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../test_precomp.hpp"
#include "opencv2/core/utils/instrumentation.hpp"
#include <cstring>

#ifdef HAVE_OPENCL

namespace opencv_test { namespace ocl {
namespace {

// CV_OCL_RUN catches cv::Exception and falls back to CPU. A numerical comparison
// alone therefore passes even when ocl_gemm fails. Error callbacks observe
// cv::error(), not logger messages; only negative GEMM error codes are counted.
class GemmErrorScope
{
public:
    GemmErrorScope() : errors(0), previousData(NULL)
    {
        previousCallback = cv::redirectError(onError, this, &previousData);
    }
    ~GemmErrorScope() { cv::redirectError(previousCallback, previousData); }
    int errors;
private:
    static int onError(int code, const char* function, const char*, const char*, int, void* data)
    {
        if (code < 0 && function && std::strstr(function, "ocl_gemm"))
            ++static_cast<GemmErrorScope*>(data)->errors;
        return 0;
    }
    cv::ErrorCallback previousCallback;
    void* previousData;
    GemmErrorScope(const GemmErrorScope&);
    GemmErrorScope& operator=(const GemmErrorScope&);
};

class OpenCLScope
{
public:
    OpenCLScope() : previous(cv::ocl::useOpenCL())
    {
#ifdef CV_COLLECT_IMPL_DATA
        previousCollection = cv::useCollection();
#endif
    }
    ~OpenCLScope()
    {
        cv::ocl::setUseOpenCL(previous);
#ifdef CV_COLLECT_IMPL_DATA
        cv::setUseCollection(previousCollection);
#endif
    }
private:
    bool previous;
#ifdef CV_COLLECT_IMPL_DATA
    bool previousCollection;
#endif
};

static void requireDevice(int depth)
{
    cv::ocl::setUseOpenCL(true);
    if (!cv::ocl::useOpenCL())
        throw cvtest::SkipTestException("OpenCL is unavailable");
    const cv::ocl::Device& device = cv::ocl::Device::getDefault();
    if (!device.compilerAvailable())
        throw cvtest::SkipTestException("OpenCL compiler is unavailable");
    if (depth == CV_64F && device.doubleFPConfig() == 0)
        throw cvtest::SkipTestException("OpenCL device does not support double precision");
}

// Optional built-in implementation collection independently verifies that gemm,
// specifically, succeeded on OpenCL (not just a copy or a transpose kernel).
static void beginGemmCollection()
{
#ifdef CV_COLLECT_IMPL_DATA
    cv::setUseCollection(true);
    cv::setImpl(0);
#endif
}

static void expectOpenCLGemm()
{
#ifdef CV_COLLECT_IMPL_DATA
    std::vector<int> implementations;
    std::vector<cv::String> functions;
    cv::getImpl(implementations, functions);
    bool found = false;
    for (size_t i = 0; i < functions.size(); ++i)
        if ((implementations[i] & CV_IMPL_OCL) && functions[i].find("gemm") != cv::String::npos)
            found = true;
    EXPECT_TRUE(found) << "GEMM fell back to CPU; numerical equality is insufficient";
#endif
}

} // namespace

PARAM_TEST_CASE(GemmAbsentC, MatType, bool) {};

OCL_TEST_P(GemmAbsentC, zeroBetaIgnoresC)
{
    OpenCLScope state;
    const int type = GET_PARAM(0);
    const bool transposeB = GET_PARAM(1);
    requireDevice(CV_MAT_DEPTH(type));
    Mat a(16, 32, type), b(transposeB ? 24 : 32, transposeB ? 32 : 24, type);
    cv::RNG dataRng(29007);
    dataRng.fill(a, cv::RNG::UNIFORM, -1.0, 1.0);
    dataRng.fill(b, cv::RNG::UNIFORM, -1.0, 1.0);
    const int flags = transposeB ? GEMM_2_T : 0;
    Mat expected;
    OCL_OFF(cv::gemm(a, b, 1.0, noArray(), 0.0, expected, flags));
    requireDevice(CV_MAT_DEPTH(type));
    UMat ua, ub, actual, emptyUMat;
    a.copyTo(ua);
    b.copyTo(ub);
    Mat emptyMat, ignoredC(3, 5, CV_8U, Scalar::all(17));
    for (int variant = 0; variant < 4; ++variant)
    {
        SCOPED_TRACE(cv::format("C variant=%d", variant));
        GemmErrorScope errors;
        beginGemmCollection();
        if (variant == 0)
            cv::gemm(ua, ub, 1.0, emptyMat, 0.0, actual, flags);
        else if (variant == 1)
            cv::gemm(ua, ub, 1.0, emptyUMat, 0.0, actual, flags);
        else if (variant == 2)
            cv::gemm(ua, ub, 1.0, noArray(), 0.0, actual, flags);
        else
            cv::gemm(ua, ub, 1.0, ignoredC, 0.0, actual, flags);
        cv::ocl::finish();
        EXPECT_EQ(0, errors.errors) << "OpenCL GEMM must ignore C when beta is zero";
        expectOpenCLGemm();
        EXPECT_LE(cvtest::norm(expected, actual, cv::NORM_INF),
                  CV_MAT_DEPTH(type) == CV_32F ? 1e-4 : 1e-10);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Core, GemmAbsentC, ::testing::Combine(
    testing::Values(CV_32FC1, CV_32FC2, CV_64FC1, CV_64FC2), Bool()));

PARAM_TEST_CASE(PCAProject, MatDepth, bool) {};

OCL_TEST_P(PCAProject, umatOutput)
{
    OpenCLScope state;
    const int depth = GET_PARAM(0);
    const bool columns = GET_PARAM(1);
    requireDevice(depth);
    Mat training(64, 16, depth), samples(12, 16, depth);
    cv::RNG dataRng(29007);
    dataRng.fill(training, cv::RNG::UNIFORM, -1.0, 1.0);
    dataRng.fill(samples, cv::RNG::UNIFORM, -1.0, 1.0);
    if (columns)
    {
        training = training.t();
        samples = samples.t();
    }
    cv::ocl::setUseOpenCL(false);
    cv::PCA pca(training, noArray(), columns ? cv::PCA::DATA_AS_COL : cv::PCA::DATA_AS_ROW, 8);
    Mat expected;
    pca.project(samples, expected);
    requireDevice(depth);
    UMat actual;
    GemmErrorScope errors;
    beginGemmCollection();
    pca.project(samples, actual);
    cv::ocl::finish();
    EXPECT_EQ(0, errors.errors) << "PCA::project must not fail inside OpenCL GEMM";
    expectOpenCLGemm();
    EXPECT_LE(cvtest::norm(expected, actual, cv::NORM_INF), depth == CV_32F ? 1e-4 : 1e-10);
}

OCL_INSTANTIATE_TEST_CASE_P(Core, PCAProject, ::testing::Combine(
    testing::Values(CV_32F, CV_64F), Bool()));

}} // namespace opencv_test::ocl
#endif // HAVE_OPENCL
