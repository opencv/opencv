// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2022 Intel Corporation


#include "test_precomp.hpp"


#include "logger.hpp"
#include "common/gapi_tests_common.hpp"
#include <opencv2/gapi/gpu/ggpukernel.hpp>
#include "opencl_kernels_test_gapi.hpp"

#ifdef HAVE_DIRECTX
#include <d3d11.h>
#include "opencv2/core/directx.hpp"
#endif // HAVE_DIRECTX
#ifdef HAVE_OPENCL
#  include "opencv2/core/opencl/runtime/opencl_core.hpp"
#endif // HAVE_OPENCL

namespace cv
{

#ifdef HAVE_OPENCL

    static void reference_symm7x7_CPU(const cv::Mat& in, const cv::Mat& kernel_coeff, int shift, cv::Mat &out)
    {
        cv::Point anchor = { -1, -1 };
        double delta = 0;

        const int* ci = kernel_coeff.ptr<int>();

        float c_float[10];
        float divisor = (float)(1 << shift);
        for (int i = 0; i < 10; i++)
        {
            c_float[i] = ci[i] / divisor;
        }
        // J & I & H & G & H & I & J
        // I & F & E & D & E & F & I
        // H & E & C & B & C & E & H
        // G & D & B & A & B & D & G
        // H & E & C & B & C & E & H
        // I & F & E & D & E & F & I
        // J & I & H & G & H & I & J

        // A & B & C & D & E & F & G & H & I & J

        // 9 & 8 & 7 & 6 & 7 & 8 & 9
        // 8 & 5 & 4 & 3 & 4 & 5 & 8
        // 7 & 4 & 2 & 1 & 2 & 4 & 7
        // 6 & 3 & 1 & 0 & 1 & 3 & 6
        // 7 & 4 & 2 & 1 & 2 & 4 & 7
        // 8 & 5 & 4 & 3 & 4 & 5 & 8
        // 9 & 8 & 7 & 6 & 7 & 8 & 9

        float coefficients[49] =
        {
            c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9],
            c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
            c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
            c_float[6], c_float[3], c_float[1], c_float[0], c_float[1], c_float[3], c_float[6],
            c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
            c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
            c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9]
        };

        cv::Mat kernel = cv::Mat(7, 7, CV_32FC1);
        float* cf = kernel.ptr<float>();
        for (int i = 0; i < 49; i++)
        {
            cf[i] = coefficients[i];
        }

        cv::filter2D(in, out, CV_8UC1, kernel, anchor, delta, cv::BORDER_REPLICATE);
    }

    namespace gapi_test_kernels
    {
        G_TYPED_KERNEL(TSymm7x7_test, <GMat(GMat, Mat, int)>, "org.opencv.imgproc.symm7x7_test") {
            static GMatDesc outMeta(GMatDesc in, Mat, int) {
                return in.withType(CV_8U, 1);
            }
        };


        GAPI_GPU_KERNEL(GGPUSymm7x7_test, TSymm7x7_test)
        {
            static void run(const cv::UMat& in, const cv::Mat& kernel_coeff, int shift, cv::UMat &out)
            {
                if (cv::ocl::isOpenCLActivated())
                {
                    cv::Size size = in.size();
                    size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };

                    const cv::String moduleName = "gapi";
                    cv::ocl::ProgramSource source(moduleName, "symm7x7", opencl_symm7x7_src, "");

                    static const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_UNDEFINED" };
                    std::string build_options = " -D BORDER_CONSTANT_VALUE=" + std::to_string(0) +
                        " -D " + borderMap[1] +
                        " -D SCALE=1.f/" + std::to_string(1 << shift) + ".f";

                    cv::String errmsg;
                    cv::ocl::Program program(source, build_options, errmsg);
                    if (program.ptr() == NULL)
                    {
                        CV_Error_(cv::Error::OpenCLInitError, ("symm_7x7_test Can't compile OpenCL program: = %s with build_options = %s\n", errmsg.c_str(), build_options.c_str()));
                    }
                    if (!errmsg.empty())
                    {
                        std::cout << "OpenCL program build log:" << std::endl << errmsg << std::endl;
                    }

                    cv::ocl::Kernel kernel("symm_7x7_test", program);
                    if (kernel.empty())
                    {
                        CV_Error(cv::Error::OpenCLInitError, "symm_7x7_test Can't get OpenCL kernel\n");
                    }

                    cv::UMat gKer;
                    kernel_coeff.copyTo(gKer);

                    int tile_y = 0;

                    int idxArg = kernel.set(0, cv::ocl::KernelArg::PtrReadOnly(in));
                    idxArg = kernel.set(idxArg, (int)in.step);
                    idxArg = kernel.set(idxArg, (int)size.width);
                    idxArg = kernel.set(idxArg, (int)size.height);
                    idxArg = kernel.set(idxArg, cv::ocl::KernelArg::PtrWriteOnly(out));
                    idxArg = kernel.set(idxArg, (int)out.step);
                    idxArg = kernel.set(idxArg, (int)size.height);
                    idxArg = kernel.set(idxArg, (int)size.width);
                    idxArg = kernel.set(idxArg, (int)tile_y);
                    idxArg = kernel.set(idxArg, cv::ocl::KernelArg::PtrReadOnly(gKer));

                    if (!kernel.run(2, globalsize, NULL, false))
                    {
                        CV_Error(cv::Error::OpenCLApiCallError, "symm_7x7_test OpenCL kernel run failed\n");
                    }
                }
                else
                {
                    //CPU fallback
                    cv::Mat in_Mat, out_Mat;
                    in_Mat = in.getMat(ACCESS_READ);
                    out_Mat = out.getMat(ACCESS_WRITE);
                    reference_symm7x7_CPU(in_Mat, kernel_coeff, shift, out_Mat);
                }
            }
        };

        cv::GKernelPackage gpuTestPackage = cv::gapi::kernels
            <GGPUSymm7x7_test
            >();

    } // namespace gapi_test_kernels
#endif //HAVE_OPENCL

} // namespace cv


namespace opencv_test
{

#ifdef HAVE_OPENCL

using namespace cv::gapi_test_kernels;

TEST(GPU, Symm7x7_test)
{
    const auto sz = cv::Size(1280, 720);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat_gapi(sz, CV_8UC1);
    cv::Mat out_mat_ocv(sz, CV_8UC1);
    cv::Scalar mean = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);
    cv::randn(in_mat, mean, stddev);

    //Symm7x7 coefficients and shift
    int coefficients_symm7x7[10] = { 1140, -118, 526, 290, -236, 64, -128, -5, -87, -7 };
    int shift = 10;
    cv::Mat kernel_coeff(10, 1, CV_32S);
    int* ci = kernel_coeff.ptr<int>();
    for (int i = 0; i < 10; i++)
    {
        ci[i] = coefficients_symm7x7[i];
    }

    // Run G-API
    cv::GMat in;
    auto out = TSymm7x7_test::on(in, kernel_coeff, shift);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    auto cc = comp.compile(cv::descr_of(in_mat), cv::compile_args(gpuTestPackage));
    cc(cv::gin(in_mat), cv::gout(out_mat_gapi));

    // Run OpenCV
    reference_symm7x7_CPU(in_mat, kernel_coeff, shift, out_mat_ocv);

    compare_f cmpF = AbsSimilarPoints(1, 0.05).to_compare_f();

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
void prepareD3D11Texture2D(const UINT width, const UINT height, std::vector<uint8_t>& test_data, ID3D11Texture2D** texture) {
    std::iota(test_data.begin(), test_data.end(), uint8_t(0));
    // Associate subresource with test data
    D3D11_SUBRESOURCE_DATA initData = { test_data.data(),
                                        sizeof(uint8_t) * width,
                                        height };

    // Create texture description
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_A8_UNORM; // CV_8U
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    desc.CPUAccessFlags = 0; // 0 if CPU access is not required

    HRESULT hr;
    // Create device
    ID3D11Device* mDevice = nullptr;
    hr = D3D11CreateDevice(nullptr,
                           D3D_DRIVER_TYPE_HARDWARE,
                           nullptr,
                           0,
                           nullptr,
                           0,
                           D3D11_SDK_VERSION,
                           &mDevice,
                           nullptr,
                           nullptr);

    if(!SUCCEEDED(hr)) {
        CV_Error(cv::Error::BadCallBack, "D3D11CreateDevice was failed: " +
                    std::to_string(HRESULT_CODE(hr)));
    }

    // Create texture2D by test data
    hr = mDevice->CreateTexture2D(&desc, &initData, &(*texture));

    if(!SUCCEEDED(hr)) {
        CV_Error(cv::Error::BadCallBack, "CreateTexture2D was failed: " +
                    std::to_string(HRESULT_CODE(hr)));
    }

    if(texture == nullptr) {
        CV_Error(cv::Error::BadDataPtr, "CreateTexture2D was failed: texture is NULL");
    }
}

TEST(GPU_D3D11, ConvTexture2DtoCLmem)
{
    // Create test data ///////////////////////////////////////////////////////
    const UINT width = 100;
    const UINT height = 150;
    ID3D11Texture2D* texture = nullptr;
    std::vector<uint8_t> test_data(width * height, 0);

    prepareD3D11Texture2D(width, height, test_data, &texture);

    // Convert texture2D to cl_mem_image //////////////////////////////////////
    cl_mem       image_mem = nullptr;
    cl_context   context = nullptr;
    cl_device_id device = nullptr;

    cv::directx::convertFromD3D11Texture2DtoCLMem(texture, &image_mem, &context, &device);

    if (context == nullptr) {
        CV_Error(cv::Error::BadDataPtr, "OpenCL: convertFromD3D11Texture2DtoCLMem returns empty cl_mem\n");
    }
    if (image_mem == nullptr) {
        CV_Error(cv::Error::BadDataPtr, "OpenCL: convertFromD3D11Texture2DtoCLMem returns empty cl_context\n");
    }
    if (device == nullptr) {
        CV_Error(cv::Error::BadDataPtr, "OpenCL: convertFromD3D11Texture2DtoCLMem returns empty cl_device_id\n");
    }

    // Move data to host
    std::vector<uint8_t> out_test_data(width * height, 0);
    // Target data
    const size_t sz0[3] = {0, 0, 0};
    const size_t sz1[3] = {width, height, 1};

    cl_command_queue q = (cl_command_queue)cv::ocl::Queue::getDefault().ptr();
    cl_int status = clEnqueueReadImage(q, image_mem, CL_FALSE, sz0, sz1,
                                       width * sizeof(uint8_t),
                                       height * sizeof(uint8_t),
                                       &out_test_data.front(), 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReadImage failed");
    }
    clFinish(q);
    clReleaseMemObject(image_mem);
    clReleaseContext(context);
    clReleaseDevice(device);
    // Comparison /////////////////////////////////////////////////////////////
    EXPECT_EQ(out_test_data, test_data);
}

TEST(GPU_D3D11, ConvTexture2DtoUMat)
{
    // Create test data ///////////////////////////////////////////////////////
    const UINT width = 150;
    const UINT height = 100;
    ID3D11Texture2D* texture = nullptr;
    std::vector<uint8_t> test_data(width * height, 0);

    prepareD3D11Texture2D(width, height, test_data, &texture);

    // Convert texture2D to cl_mem_image //////////////////////////////////////
    cl_mem       image_mem = nullptr;
    cl_context   context = nullptr;
    cl_device_id device = nullptr;
    cv::directx::convertFromD3D11Texture2DtoCLMem(texture, &image_mem, &context, &device);

    if (context == nullptr) {
        CV_Error(cv::Error::BadDataPtr, "OpenCL: convertFromD3D11Texture2DtoCLImage returns empty cl_mem\n");
    }
    if (image_mem == nullptr) {
        CV_Error(cv::Error::BadDataPtr, "OpenCL: convertFromD3D11Texture2DtoCLImage returns empty cl_context\n");
    }
    if (device == nullptr) {
        CV_Error(cv::Error::BadDataPtr, "OpenCL: convertFromD3D11Texture2DtoCLImage returns empty cl_device_id\n");
    }

    // Convert to UMat function ///////////////////////////////////////////////
    // Create cv::Umat
    cv::UMat umt_test(USAGE_ALLOCATE_DEVICE_MEMORY);
    // Copy data from cl_mem_image to cv::Umat buffer
    cv::ocl::convertFromImage(image_mem, umt_test);
    clReleaseMemObject(image_mem);
    clReleaseContext(context);
    clReleaseDevice(device);

    // Comparison /////////////////////////////////////////////////////////////
    EXPECT_EQ(umt_test.total(), test_data.size());
    EXPECT_EQ(umt_test.size(), cv::Size(width, height));
    EXPECT_EQ(umt_test.type(), CV_8U);

    uint8_t* umt_data = umt_test.getMat(ACCESS_READ).ptr<uint8_t>();
    for (int i = 0; i < umt_test.total(); ++i) {
        EXPECT_EQ(umt_data[i], test_data[i]);
    }
}

#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_OPENCL

} // namespace opencv_test
