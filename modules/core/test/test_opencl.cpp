// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

namespace opencv_test {
namespace ocl {

static
testing::internal::ParamGenerator<std::string> getOpenCLTestConfigurations()
{
    if (!cv::ocl::useOpenCL())
    {
        return testing::ValuesIn(std::vector<std::string>());
    }

    std::vector<std::string> configurations = {
        ":GPU:0",
        ":GPU:1",
        ":CPU:0",
    };
    return testing::ValuesIn(configurations);
}


static void executeUMatCall(bool requireOpenCL = true)
{
    UMat a(100, 100, CV_8UC1, Scalar::all(0));
    UMat b;
    cv::add(a, Scalar::all(1), b);
    Mat b_cpu = b.getMat(ACCESS_READ);
    EXPECT_EQ(0, cv::norm(b_cpu - 1, NORM_INF));

    if (requireOpenCL)
    {
        EXPECT_TRUE(cv::ocl::useOpenCL());
    }
}

TEST(OCL_Context, createFromDevice)
{
    bool useOCL = cv::ocl::useOpenCL();

    OpenCLExecutionContext ctx = OpenCLExecutionContext::getCurrent();

    if (!useOCL)
    {
        ASSERT_TRUE(ctx.empty());  // Other tests should not broke global state
        throw SkipTestException("OpenCL is not available / disabled");
    }

    ASSERT_FALSE(ctx.empty());

    ocl::Device device = ctx.getDevice();
    ASSERT_FALSE(device.empty());

    ocl::Context context = ocl::Context::fromDevice(device);
    ocl::Context context2 = ocl::Context::fromDevice(device);

    EXPECT_TRUE(context.getImpl() == context2.getImpl()) << "Broken cache for OpenCL context (device)";
}

TEST(OCL_OpenCLExecutionContextDefault, basic)
{
    bool useOCL = cv::ocl::useOpenCL();

    OpenCLExecutionContext ctx = OpenCLExecutionContext::getCurrent();

    if (!useOCL)
    {
        ASSERT_TRUE(ctx.empty());  // Other tests should not broke global state
        throw SkipTestException("OpenCL is not available / disabled");
    }

    ASSERT_FALSE(ctx.empty());

    ocl::Context context = ctx.getContext();
    ocl::Context context2 = ocl::Context::getDefault();
    EXPECT_TRUE(context.getImpl() == context2.getImpl());

    ocl::Device device = ctx.getDevice();
    ocl::Device device2 = ocl::Device::getDefault();
    EXPECT_TRUE(device.getImpl() == device2.getImpl());

    ocl::Queue queue = ctx.getQueue();
    ocl::Queue queue2 = ocl::Queue::getDefault();
    EXPECT_TRUE(queue.getImpl() == queue2.getImpl());
}

TEST(OCL_OpenCLExecutionContextDefault, createAndBind)
{
    bool useOCL = cv::ocl::useOpenCL();

    OpenCLExecutionContext ctx = OpenCLExecutionContext::getCurrent();

    if (!useOCL)
    {
        ASSERT_TRUE(ctx.empty());  // Other tests should not broke global state
        throw SkipTestException("OpenCL is not available / disabled");
    }

    ASSERT_FALSE(ctx.empty());

    ocl::Context context = ctx.getContext();
    ocl::Device device = ctx.getDevice();

    OpenCLExecutionContext ctx2 = OpenCLExecutionContext::create(context, device);
    ASSERT_FALSE(ctx2.empty());

    try
    {
        ctx2.bind();
        executeUMatCall();
        ctx.bind();
        executeUMatCall();
    }
    catch (...)
    {
        ctx.bind();  // restore
        throw;
    }
}

typedef testing::TestWithParam<std::string> OCL_OpenCLExecutionContext_P;

TEST_P(OCL_OpenCLExecutionContext_P, multipleBindAndExecute)
{
    bool useOCL = cv::ocl::useOpenCL();

    OpenCLExecutionContext ctx = OpenCLExecutionContext::getCurrent();

    if (!useOCL)
    {
        ASSERT_TRUE(ctx.empty());  // Other tests should not broke global state
        throw SkipTestException("OpenCL is not available / disabled");
    }

    ASSERT_FALSE(ctx.empty());

    std::string opencl_device = GetParam();
    ocl::Context context = ocl::Context::create(opencl_device);
    if (context.empty())
    {
        throw SkipTestException(std::string("OpenCL device is not available: '") + opencl_device + "'");
    }

    ocl::Device device = context.device(0);

    OpenCLExecutionContext ctx2 = OpenCLExecutionContext::create(context, device);
    ASSERT_FALSE(ctx2.empty());

    try
    {
        std::cout << "ctx2..." << std::endl;
        ctx2.bind();
        executeUMatCall();
        std::cout << "ctx..." << std::endl;
        ctx.bind();
        executeUMatCall();
    }
    catch (...)
    {
        ctx.bind();  // restore
        throw;
    }
}

TEST_P(OCL_OpenCLExecutionContext_P, ScopeTest)
{
    bool useOCL = cv::ocl::useOpenCL();

    OpenCLExecutionContext ctx = OpenCLExecutionContext::getCurrent();

    if (!useOCL)
    {
        ASSERT_TRUE(ctx.empty());  // Other tests should not broke global state
        throw SkipTestException("OpenCL is not available / disabled");
    }

    ASSERT_FALSE(ctx.empty());

    std::string opencl_device = GetParam();
    ocl::Context context = ocl::Context::create(opencl_device);
    if (context.empty())
    {
        throw SkipTestException(std::string("OpenCL device is not available: '") + opencl_device + "'");
    }

    ocl::Device device = context.device(0);

    OpenCLExecutionContext ctx2 = OpenCLExecutionContext::create(context, device);
    ASSERT_FALSE(ctx2.empty());

    try
    {
        OpenCLExecutionContextScope ctx_scope(ctx2);
        executeUMatCall();
    }
    catch (...)
    {
        ctx.bind();  // restore
        throw;
    }

    executeUMatCall();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, OCL_OpenCLExecutionContext_P, getOpenCLTestConfigurations());


typedef testing::TestWithParam<UMatUsageFlags> UsageFlagsFixture;
OCL_TEST_P(UsageFlagsFixture, UsageFlagsRetained)
{
    if (!cv::ocl::useOpenCL())
    {
        throw SkipTestException("OpenCL is not available / disabled");
    }

    const UMatUsageFlags usage = GetParam();
    cv::UMat flip_in(10, 10, CV_32F, usage);
    cv::UMat flip_out(usage);
    cv::flip(flip_in, flip_out, 1);
    cv::ocl::finish();

    ASSERT_EQ(usage, flip_in.usageFlags);
    ASSERT_EQ(usage, flip_out.usageFlags);
}

INSTANTIATE_TEST_CASE_P(
    /*nothing*/,
    UsageFlagsFixture,
    testing::Values(USAGE_DEFAULT, USAGE_ALLOCATE_HOST_MEMORY, USAGE_ALLOCATE_DEVICE_MEMORY)
);


} } // namespace opencv_test::ocl
