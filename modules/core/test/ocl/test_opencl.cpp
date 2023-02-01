// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../test_precomp.hpp"

#include <opencv2/core/ocl.hpp>

namespace opencv_test { namespace {

static void testOpenCLKernel(cv::ocl::Kernel& k)
{
    ASSERT_FALSE(k.empty());
    cv::UMat src(cv::Size(4096, 2048), CV_8UC1, cv::Scalar::all(100));
    cv::UMat dst(src.size(), CV_8UC1);
    size_t globalSize[2] = {(size_t)src.cols, (size_t)src.rows};
    size_t localSize[2] = {8, 8};
    int64 kernel_time = k.args(
            cv::ocl::KernelArg::ReadOnlyNoSize(src), // size is not used (similar to 'dst' size)
            cv::ocl::KernelArg::WriteOnly(dst),
            (int)5
        ).runProfiling(2, globalSize, localSize);
    ASSERT_GE(kernel_time, (int64)0);
    std::cout << "Kernel time: " << (kernel_time * 1e-6) << " ms" << std::endl;
    cv::Mat res, reference(src.size(), CV_8UC1, cv::Scalar::all(105));
    dst.copyTo(res);
    EXPECT_EQ(0, cvtest::norm(reference, res, cv::NORM_INF));
}

TEST(OpenCL, support_binary_programs)
{
    cv::ocl::Context ctx = cv::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        throw cvtest::SkipTestException("OpenCL is not available");
    }
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    if (!device.compilerAvailable())
    {
        throw cvtest::SkipTestException("OpenCL compiler is not available");
    }
    std::vector<char> program_binary_code;

    cv::String module_name; // empty to disable OpenCL cache

    { // Generate program binary from OpenCL C source
        static const char* opencl_kernel_src =
"__kernel void test_kernel(__global const uchar* src, int src_step, int src_offset,\n"
"                          __global uchar* dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,\n"
"                          int c)\n"
"{\n"
"   int x = get_global_id(0);\n"
"   int y = get_global_id(1);\n"
"   if (x < dst_cols && y < dst_rows)\n"
"   {\n"
"       int src_idx = y * src_step + x + src_offset;\n"
"       int dst_idx = y * dst_step + x + dst_offset;\n"
"       dst[dst_idx] = src[src_idx] + c;\n"
"   }\n"
"}\n";
        cv::ocl::ProgramSource src(module_name, "simple", opencl_kernel_src, "");
        cv::String errmsg;
        cv::ocl::Program program(src, "", errmsg);
        ASSERT_TRUE(program.ptr() != NULL);
        cv::ocl::Kernel k("test_kernel", program);
        EXPECT_FALSE(k.empty());
        program.getBinary(program_binary_code);
        std::cout << "Program binary size: " << program_binary_code.size() << " bytes" << std::endl;
    }

    cv::ocl::Kernel k;

    { // Load program from binary (without sources)
        ASSERT_FALSE(program_binary_code.empty());
        cv::ocl::ProgramSource src = cv::ocl::ProgramSource::fromBinary(module_name, "simple_binary", (uchar*)&program_binary_code[0], program_binary_code.size(), "");
        cv::String errmsg;
        cv::ocl::Program program(src, "", errmsg);
        ASSERT_TRUE(program.ptr() != NULL);
        k.create("test_kernel", program);
    }

    testOpenCLKernel(k);
}


TEST(OpenCL, support_SPIR_programs)
{
    cv::ocl::Context ctx = cv::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        throw cvtest::SkipTestException("OpenCL is not available");
    }
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    if (!device.isExtensionSupported("cl_khr_spir"))
    {
        throw cvtest::SkipTestException("'cl_khr_spir' extension is not supported by OpenCL device");
    }
    std::vector<char> program_binary_code;
    cv::String fname = cv::format("test_kernel.spir%d", device.addressBits());
    std::string full_path = cvtest::findDataFile(std::string("opencl/") + fname);

    {
        std::fstream f(full_path.c_str(), std::ios::in|std::ios::binary);
        ASSERT_TRUE(f.is_open());
        size_t pos = (size_t)f.tellg();
        f.seekg(0, std::fstream::end);
        size_t fileSize = (size_t)f.tellg();
        std::cout << "Program SPIR size: " << fileSize << " bytes" << std::endl;
        f.seekg(pos, std::fstream::beg);
        program_binary_code.resize(fileSize);
        f.read(&program_binary_code[0], fileSize);
        ASSERT_FALSE(f.fail());
    }

    cv::String module_name; // empty to disable OpenCL cache

    cv::ocl::Kernel k;

    { // Load program from SPIR format
        ASSERT_FALSE(program_binary_code.empty());
        cv::ocl::ProgramSource src = cv::ocl::ProgramSource::fromSPIR(module_name, "simple_spir", (uchar*)&program_binary_code[0], program_binary_code.size(), "");
        cv::String errmsg;
        cv::ocl::Program program(src, "", errmsg);
        if (program.ptr() == NULL && device.isAMD())
        {
            // https://community.amd.com/t5/opencl/spir-support-in-new-drivers-lost/td-p/170165
            throw cvtest::SkipTestException("Bypass AMD OpenCL runtime bug: 'cl_khr_spir' extension is declared, but it doesn't really work");
        }
        ASSERT_TRUE(program.ptr() != NULL);
        k.create("test_kernel", program);
    }

    testOpenCLKernel(k);
}

TEST(OpenCL, image2Dcount_regression_19334)
{
    cv::ocl::Context ctx = cv::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        throw cvtest::SkipTestException("OpenCL is not available");
    }
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    if (!device.compilerAvailable())
    {
        throw cvtest::SkipTestException("OpenCL compiler is not available");
    }

    std::string module_name; // empty to disable OpenCL cache

    static const char* opencl_kernel_src =
"__kernel void test_kernel(int a,\n"
"                          __global const uchar* src0, int src0_step, int src0_offset, int src0_rows, int src0_cols,\n"
"                          __global const uchar* src1, int src1_step, int src1_offset, int src1_rows, int src1_cols,\n"
"                          __global const uchar* src2, int src2_step, int src2_offset, int src2_rows, int src2_cols,\n"
"                          __read_only image2d_t image)\n"
"{\n"
"}";
    cv::ocl::ProgramSource src(module_name, "test_opencl_image_arg", opencl_kernel_src, "");
    cv::String errmsg;
    cv::ocl::Program program(src, "", errmsg);
    ASSERT_TRUE(program.ptr() != NULL);
    cv::ocl::Kernel k("test_kernel", program);
    ASSERT_FALSE(k.empty());

    std::vector<UMat> images(4);
    for (size_t i = 0; i < images.size(); ++i)
        images[i] = UMat(10, 10, CV_8UC1);
    cv::ocl::Image2D image;
    try
    {
        cv::ocl::Image2D image_(images.back());
        image = image_;
    }
    catch (const cv::Exception&)
    {
        throw cvtest::SkipTestException("OpenCL images are not supported");
    }

    int nargs = 0;
    int a = 0;
    nargs = k.set(nargs, a);
    ASSERT_EQ(1, nargs);
    nargs = k.set(nargs, images[0]);
    ASSERT_EQ(6, nargs);
    nargs = k.set(nargs, images[1]);
    ASSERT_EQ(11, nargs);
    nargs = k.set(nargs, images[2]);
    ASSERT_EQ(16, nargs);

    // do not throw (issue of #19334)
    ASSERT_NO_THROW(nargs = k.set(nargs, image));
    ASSERT_EQ(17, nargs);

    // allow to replace image argument if kernel is not running
    UMat image2(10, 10, CV_8UC1);
    ASSERT_NO_THROW(nargs = k.set(16, cv::ocl::Image2D(image2)));
    ASSERT_EQ(17, nargs);
}

}} // namespace
