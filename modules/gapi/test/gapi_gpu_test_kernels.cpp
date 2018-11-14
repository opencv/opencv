// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "test_precomp.hpp"

#include <iomanip>
#include "gapi_gpu_test_kernels.hpp"
#include <opencv2/gapi/core.hpp>
#include "opencl_kernels_test_gapi.hpp"


namespace cv
{
namespace gapi_test_kernels
{

#ifdef HAVE_OPENCL
GAPI_GPU_KERNEL(GGPUSymm7x7_test, TSymm7x7_test)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        if (cv::ocl::isOpenCLActivated())
        {
            cv::Size size = in.size();
            size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };

            //size_t maxWorkItemSizes[32];
            //cv::ocl::Device::getDefault().maxWorkItemSizes(maxWorkItemSizes);

            cv::ocl::Kernel kernel;

            int coefficients[10] = { 1140, -118, 526, 290, -236, 64, -128, -5, -87, -7 };
            int shift = 10;


            static const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_UNDEFINED" };
            std::string build_options = " -D BORDER_CONSTANT_VALUE=" + std::to_string(0) +
                " -D " + borderMap[1] +
                " -D SCALE=1.f/" + std::to_string(1 << shift) + ".f";


            if (!kernel.create("symm_7x7_test", cv::ocl::gapi::symm7x7_test_oclsrc, build_options))
            {
                printf("symm_7x7_test OpenCL kernel creation failed with build_options = %s\n", build_options.c_str());
            }

            //prepare coefficients for device
            cv::Mat kernel_coeff(10, 1, CV_32S);
            int* ci = kernel_coeff.ptr<int>();
            for (int i = 0; i < 10; i++)
            {
                ci[i] = coefficients[i];
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
                printf("symm_7x7_test OpenCL kernel run failed\n");
            }
        }
        else
        {
            //printf("symm_7x7 OpenCL kernel run failed - OpenCL is not activated\n");

            //CPU fallback
            cv::Point anchor = { -1, -1 };
            double delta = 0;

            int c_int[10] = { 1140, -118, 526, 290, -236, 64, -128, -5, -87, -7 };
            float c_float[10];
            for (int i = 0; i < 10; i++)
            {
                c_float[i] = c_int[i] / 1024.0f;
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
    }
};
#endif


cv::gapi::GKernelPackage gpuTestPackage = cv::gapi::kernels
        <GGPUSymm7x7_test
        >();
} // namespace gapi_test_kernels
} // namespace cv
