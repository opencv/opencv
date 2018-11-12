// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/cpu/imgproc.hpp"
#include "backends/cpu/gcpuimgproc.hpp"

GAPI_OCV_KERNEL(GCPUSepFilter, cv::gapi::imgproc::GSepFilter)
{
    static void run(const cv::Mat& in, int ddepth, const cv::Mat& kernX, const cv::Mat& kernY, const cv::Point& anchor, const cv::Scalar& delta,
                    int border, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( border == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
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

GAPI_OCV_KERNEL(GCPUBoxFilter, cv::gapi::imgproc::GBoxFilter)
{
    static void run(const cv::Mat& in, int ddepth, const cv::Size& ksize, const cv::Point& anchor, bool normalize, int borderType, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
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

GAPI_OCV_KERNEL(GCPUBlur, cv::gapi::imgproc::GBlur)
{
    static void run(const cv::Mat& in, const cv::Size& ksize, const cv::Point& anchor, int borderType, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
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


GAPI_OCV_KERNEL(GCPUFilter2D, cv::gapi::imgproc::GFilter2D)
{
    static void run(const cv::Mat& in, int ddepth, const cv::Mat& k, const cv::Point& anchor, const cv::Scalar& delta, int border,
                    const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( border == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
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

GAPI_OCV_KERNEL(GCPUGaussBlur, cv::gapi::imgproc::GGaussBlur)
{
    static void run(const cv::Mat& in, const cv::Size& ksize, double sigmaX, double sigmaY, int borderType, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
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

GAPI_OCV_KERNEL(GCPUMedianBlur, cv::gapi::imgproc::GMedianBlur)
{
    static void run(const cv::Mat& in, int ksize, cv::Mat &out)
    {
        cv::medianBlur(in, out, ksize);
    }
};

GAPI_OCV_KERNEL(GCPUErode, cv::gapi::imgproc::GErode)
{
    static void run(const cv::Mat& in, const cv::Mat& kernel, const cv::Point& anchor, int iterations, int borderType, const cv::Scalar& borderValue, cv::Mat &out)
    {
        cv::erode(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUDilate, cv::gapi::imgproc::GDilate)
{
    static void run(const cv::Mat& in, const cv::Mat& kernel, const cv::Point& anchor, int iterations, int borderType, const cv::Scalar& borderValue, cv::Mat &out)
    {
        cv::dilate(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUSobel, cv::gapi::imgproc::GSobel)
{
    static void run(const cv::Mat& in, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType,
                    const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
            int add = (ksize - 1) / 2;
            cv::copyMakeBorder(in, temp_in, add, add, add, add, borderType, bordVal );
            cv::Rect rect = cv::Rect(add, add, in.cols, in.rows);
            cv::Sobel(temp_in(rect), out, ddepth, dx, dy, ksize, scale, delta, borderType);
        }
        else
        cv::Sobel(in, out, ddepth, dx, dy, ksize, scale, delta, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUEqualizeHist, cv::gapi::imgproc::GEqHist)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::equalizeHist(in, out);
    }
};

GAPI_OCV_KERNEL(GCPUCanny, cv::gapi::imgproc::GCanny)
{
    static void run(const cv::Mat& in, double thr1, double thr2, int apSize, bool l2gradient, cv::Mat &out)
    {
        cv::Canny(in, out, thr1, thr2, apSize, l2gradient);
    }
};

GAPI_OCV_KERNEL(GCPURGB2YUV, cv::gapi::imgproc::GRGB2YUV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2YUV);
    }
};

GAPI_OCV_KERNEL(GCPUYUV2RGB, cv::gapi::imgproc::GYUV2RGB)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2RGB);
    }
};

GAPI_OCV_KERNEL(GCPURGB2Lab, cv::gapi::imgproc::GRGB2Lab)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2Lab);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2LUV, cv::gapi::imgproc::GBGR2LUV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2Luv);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2YUV, cv::gapi::imgproc::GBGR2YUV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2YUV);
    }
};

GAPI_OCV_KERNEL(GCPULUV2BGR, cv::gapi::imgproc::GLUV2BGR)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_Luv2BGR);
    }
};

GAPI_OCV_KERNEL(GCPUYUV2BGR, cv::gapi::imgproc::GYUV2BGR)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2BGR);
    }
};

GAPI_OCV_KERNEL(GCPURGB2Gray, cv::gapi::imgproc::GRGB2Gray)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2GRAY);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2Gray, cv::gapi::imgproc::GBGR2Gray)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    }
};

GAPI_OCV_KERNEL(GCPURGB2GrayCustom, cv::gapi::imgproc::GRGB2GrayCustom)
{
    static void run(const cv::Mat& in, float rY, float bY, float gY, cv::Mat &out)
    {
        cv::Mat planes[3];
        cv::split(in, planes);
        out = planes[0]*rY + planes[1]*bY + planes[2]*gY;
    }
};

//custom kernel example
GAPI_OCV_KERNEL(GCPUSymm7x7, cv::gapi::imgproc::GSymm7x7)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
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
};

cv::gapi::GKernelPackage cv::gapi::imgproc::cpu::kernels()
{
    static auto pkg = cv::gapi::kernels
        < GCPUFilter2D
        , GCPUSepFilter
        , GCPUBoxFilter
        , GCPUBlur
        , GCPUGaussBlur
        , GCPUMedianBlur
        , GCPUErode
        , GCPUDilate
        , GCPUSobel
        , GCPUCanny
        , GCPUEqualizeHist
        , GCPURGB2YUV
        , GCPUYUV2RGB
        , GCPURGB2Lab
        , GCPUBGR2LUV
        , GCPUBGR2YUV
        , GCPUYUV2BGR
        , GCPULUV2BGR
        , GCPUBGR2Gray
        , GCPURGB2Gray
        , GCPURGB2GrayCustom
        , GCPUSymm7x7
        >();
    return pkg;
}
