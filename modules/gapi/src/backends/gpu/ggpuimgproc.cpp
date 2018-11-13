// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/gpu/imgproc.hpp"
#include "backends/gpu/ggpuimgproc.hpp"
#include "opencl_kernels_gapi.hpp"


GAPI_GPU_KERNEL(GGPUSepFilter, cv::gapi::imgproc::GSepFilter)
{
    static void run(const cv::UMat& in, int ddepth, const cv::Mat& kernX, const cv::Mat& kernY, const cv::Point& anchor, const cv::Scalar& delta,
                    int border, const cv::Scalar& bordVal, cv::UMat &out)
    {
        if( border == cv::BORDER_CONSTANT )
        {
            cv::UMat temp_in;
            int width_add = (kernY.cols - 1) / 2;
            int height_add =  (kernX.rows - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, border, bordVal);
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::sepFilter2D(temp_in(rect), out, ddepth, kernX, kernY, anchor, delta.val[0], border);
        }
        else
            cv::sepFilter2D(in, out, ddepth, kernX, kernY, anchor, delta.val[0], border);
    }
};

GAPI_GPU_KERNEL(GGPUBoxFilter, cv::gapi::imgproc::GBoxFilter)
{
    static void run(const cv::UMat& in, int ddepth, const cv::Size& ksize, const cv::Point& anchor, bool normalize, int borderType, const cv::Scalar& bordVal, cv::UMat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::UMat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::boxFilter(temp_in(rect), out, ddepth, ksize, anchor, normalize, borderType);
        }
        else
            cv::boxFilter(in, out, ddepth, ksize, anchor, normalize, borderType);
    }
};

GAPI_GPU_KERNEL(GGPUBlur, cv::gapi::imgproc::GBlur)
{
    static void run(const cv::UMat& in, const cv::Size& ksize, const cv::Point& anchor, int borderType, const cv::Scalar& bordVal, cv::UMat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::UMat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::blur(temp_in(rect), out, ksize, anchor, borderType);
        }
        else
            cv::blur(in, out, ksize, anchor, borderType);
    }
};


GAPI_GPU_KERNEL(GGPUFilter2D, cv::gapi::imgproc::GFilter2D)
{
    static void run(const cv::UMat& in, int ddepth, const cv::Mat& k, const cv::Point& anchor, const cv::Scalar& delta, int border,
                    const cv::Scalar& bordVal, cv::UMat &out)
    {
        if( border == cv::BORDER_CONSTANT )
        {
            cv::UMat temp_in;
            int width_add = (k.cols - 1) / 2;
            int height_add =  (k.rows - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, border, bordVal );
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::filter2D(temp_in(rect), out, ddepth, k, anchor, delta.val[0], border);
        }
        else
            cv::filter2D(in, out, ddepth, k, anchor, delta.val[0], border);
    }
};

GAPI_GPU_KERNEL(GGPUGaussBlur, cv::gapi::imgproc::GGaussBlur)
{
    static void run(const cv::UMat& in, const cv::Size& ksize, double sigmaX, double sigmaY, int borderType, const cv::Scalar& bordVal, cv::UMat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::UMat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal );
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::GaussianBlur(temp_in(rect), out, ksize, sigmaX, sigmaY, borderType);
        }
        else
            cv::GaussianBlur(in, out, ksize, sigmaX, sigmaY, borderType);
    }
};

GAPI_GPU_KERNEL(GGPUMedianBlur, cv::gapi::imgproc::GMedianBlur)
{
    static void run(const cv::UMat& in, int ksize, cv::UMat &out)
    {
        cv::medianBlur(in, out, ksize);
    }
};

GAPI_GPU_KERNEL(GGPUErode, cv::gapi::imgproc::GErode)
{
    static void run(const cv::UMat& in, const cv::Mat& kernel, const cv::Point& anchor, int iterations, int borderType, const cv::Scalar& borderValue, cv::UMat &out)
    {
        cv::erode(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_GPU_KERNEL(GGPUDilate, cv::gapi::imgproc::GDilate)
{
    static void run(const cv::UMat& in, const cv::Mat& kernel, const cv::Point& anchor, int iterations, int borderType, const cv::Scalar& borderValue, cv::UMat &out)
    {
        cv::dilate(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_GPU_KERNEL(GGPUSobel, cv::gapi::imgproc::GSobel)
{
    static void run(const cv::UMat& in, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType,
                    const cv::Scalar& bordVal, cv::UMat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::UMat temp_in;
            int add = (ksize - 1) / 2;
            cv::copyMakeBorder(in, temp_in, add, add, add, add, borderType, bordVal );
            cv::Rect rect = cv::Rect(add, add, in.cols, in.rows);
            cv::Sobel(temp_in(rect), out, ddepth, dx, dy, ksize, scale, delta, borderType);
        }
        else
        cv::Sobel(in, out, ddepth, dx, dy, ksize, scale, delta, borderType);
    }
};

GAPI_GPU_KERNEL(GGPUEqualizeHist, cv::gapi::imgproc::GEqHist)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::equalizeHist(in, out);
    }
};

GAPI_GPU_KERNEL(GGPUCanny, cv::gapi::imgproc::GCanny)
{
    static void run(const cv::UMat& in, double thr1, double thr2, int apSize, bool l2gradient, cv::UMat &out)
    {
        cv::Canny(in, out, thr1, thr2, apSize, l2gradient);
    }
};

GAPI_GPU_KERNEL(GGPURGB2YUV, cv::gapi::imgproc::GRGB2YUV)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2YUV);
    }
};

GAPI_GPU_KERNEL(GGPUYUV2RGB, cv::gapi::imgproc::GYUV2RGB)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2RGB);
    }
};

GAPI_GPU_KERNEL(GGPURGB2Lab, cv::gapi::imgproc::GRGB2Lab)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2Lab);
    }
};

GAPI_GPU_KERNEL(GGPUBGR2LUV, cv::gapi::imgproc::GBGR2LUV)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2Luv);
    }
};

GAPI_GPU_KERNEL(GGPUBGR2YUV, cv::gapi::imgproc::GBGR2YUV)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2YUV);
    }
};

GAPI_GPU_KERNEL(GGPULUV2BGR, cv::gapi::imgproc::GLUV2BGR)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_Luv2BGR);
    }
};

GAPI_GPU_KERNEL(GGPUYUV2BGR, cv::gapi::imgproc::GYUV2BGR)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2BGR);
    }
};

GAPI_GPU_KERNEL(GGPURGB2Gray, cv::gapi::imgproc::GRGB2Gray)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2GRAY);
    }
};

GAPI_GPU_KERNEL(GGPUBGR2Gray, cv::gapi::imgproc::GBGR2Gray)
{
    static void run(const cv::UMat& in, cv::UMat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    }
};

GAPI_GPU_KERNEL(GGPURGB2GrayCustom, cv::gapi::imgproc::GRGB2GrayCustom)
{
    //TODO: avoid copy
    static void run(const cv::UMat& in, float rY, float bY, float gY, cv::UMat &out)
    {
        cv::Mat planes[3];
        cv::split(in.getMat(cv::ACCESS_READ), planes);
        cv::Mat tmp_out = (planes[0]*rY + planes[1]*bY + planes[2]*gY);
        tmp_out.copyTo(out);
    }
};

#ifdef HAVE_OPENCL
GAPI_GPU_KERNEL(GGPUSymm7x7, cv::gapi::imgproc::GSymm7x7)
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


            if (!kernel.create("symm_7x7", cv::ocl::gapi::symm7x7_oclsrc, build_options))
            {
                printf("symm_7x7 OpenCL kernel creation failed with build_options = %s\n", build_options.c_str());
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
                printf("symm_7x7 OpenCL kernel run failed\n");
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

cv::gapi::GKernelPackage cv::gapi::imgproc::gpu::kernels()
{
    static auto pkg = cv::gapi::kernels
        < GGPUFilter2D
        , GGPUSepFilter
        , GGPUBoxFilter
        , GGPUBlur
        , GGPUGaussBlur
        , GGPUMedianBlur
        , GGPUErode
        , GGPUDilate
        , GGPUSobel
        , GGPUCanny
        , GGPUEqualizeHist
        , GGPURGB2YUV
        , GGPUYUV2RGB
        , GGPURGB2Lab
        , GGPUBGR2LUV
        , GGPUBGR2YUV
        , GGPUYUV2BGR
        , GGPULUV2BGR
        , GGPUBGR2Gray
        , GGPURGB2Gray
        , GGPURGB2GrayCustom
#ifdef HAVE_OPENCL
        , GGPUSymm7x7
#endif
        >();
    return pkg;
}
