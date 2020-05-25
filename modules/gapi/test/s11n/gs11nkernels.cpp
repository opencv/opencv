// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include "gs11nkernels.hpp"
#include "gs11nkernel.hpp"
#include <opencv2/gapi/gcompoundkernel.hpp>

#include "backends/fluid/gfluidimgproc_func.hpp"

//core kernels
GAPI_S11N_KERNEL(GS11NAdd, cv::gapi::core::GAdd)
{
    static void run(const cv::Mat& a, const cv::Mat& b, int dtype, cv::Mat& out)
    {
        cv::add(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_S11N_KERNEL(GS11NSub, cv::gapi::core::GSub)
{
    static void run(const cv::Mat& a, const cv::Mat& b, int dtype, cv::Mat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_S11N_KERNEL(GS11NMul, cv::gapi::core::GMul)
{
    static void run(const cv::Mat& a, const cv::Mat& b, double scale, int dtype, cv::Mat& out)
    {
        cv::multiply(a, b, out, scale, dtype);
    }
};

GAPI_S11N_KERNEL(GS11NMulCOld, cv::gapi::core::GMulCOld)
{
    static void run(const cv::Mat& a, double b, int dtype, cv::Mat& out)
    {
        cv::multiply(a, b, out, 1, dtype);
    }
};

GAPI_S11N_KERNEL(GS11NNot, cv::gapi::core::GNot)
{
    static void run(const cv::Mat& a, cv::Mat& out)
    {
        cv::bitwise_not(a, out);
    }
};

GAPI_S11N_KERNEL(GS11NSum, cv::gapi::core::GSum)
{
    static void run(const cv::Mat& in, cv::Scalar& out)
    {
        out = cv::sum(in);
    }
};

GAPI_S11N_KERNEL(GS11NSplit3, cv::gapi::core::GSplit3)
{
    static void run(const cv::Mat& in, cv::Mat &m1, cv::Mat &m2, cv::Mat &m3)
    {
        std::vector<cv::Mat> outMats = { m1, m2, m3 };
        cv::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsense completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
    }
};

GAPI_S11N_KERNEL(GS11NMerge3, cv::gapi::core::GMerge3)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, const cv::Mat& in3, cv::Mat &out)
    {
        std::vector<cv::Mat> inMats = { in1, in2, in3 };
        cv::merge(inMats, out);
    }
};

GAPI_S11N_KERNEL(GS11NResize, cv::gapi::core::GResize)
{
    static void run(const cv::Mat& in, cv::Size sz, double fx, double fy, int interp, cv::Mat &out)
    {
        cv::resize(in, out, sz, fx, fy, interp);
    }
};

GAPI_S11N_KERNEL(GS11NCrop, cv::gapi::core::GCrop)
{
    static void run(const cv::Mat& in, cv::Rect rect, cv::Mat& out)
    {
        cv::Mat(in, rect).copyTo(out);
    }
};

//imgproc kernels

GAPI_S11N_KERNEL(GS11NFilter2D, cv::gapi::imgproc::GFilter2D)
{
    static void run(const cv::Mat& in, int ddepth, const cv::Mat& k, const cv::Point& anchor, const cv::Scalar& delta, int border,
        const cv::Scalar& bordVal, cv::Mat &out)
    {
        if (border == cv::BORDER_CONSTANT)
        {
            cv::Mat temp_in;
            int width_add = (k.cols - 1) / 2;
            int height_add = (k.rows - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, border, bordVal);
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::filter2D(temp_in(rect), out, ddepth, k, anchor, delta.val[0], border);
        }
        else
            cv::filter2D(in, out, ddepth, k, anchor, delta.val[0], border);
    }
};

GAPI_S11N_KERNEL(GS11NCanny, cv::gapi::imgproc::GCanny)
{
    static void run(const cv::Mat& in, double thr1, double thr2, int apSize, bool l2gradient, cv::Mat &out)
    {
        cv::Canny(in, out, thr1, thr2, apSize, l2gradient);
    }
};

GAPI_S11N_KERNEL(GS11NRGB2YUV, cv::gapi::imgproc::GRGB2YUV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2YUV);
    }
};

GAPI_S11N_KERNEL(GS11NYUV2RGB, cv::gapi::imgproc::GYUV2RGB)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2RGB);
    }
};

cv::gapi::GKernelPackage opencv_test::s11n::kernels()
{
    static auto pkg = cv::gapi::kernels
        <
        //core
        GS11NAdd
        , GS11NSub
        , GS11NMul
        , GS11NMulCOld
        , GS11NNot
        , GS11NSum
        , GS11NSplit3
        , GS11NResize
        , GS11NMerge3
        , GS11NCrop
        //imgproc
        , GS11NFilter2D
        , GS11NCanny
        , GS11NRGB2YUV
        , GS11NYUV2RGB
        >();
    return pkg;
}
