// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/cpu/core.hpp"
#include "backends/cpu/gcpucore.hpp"

GAPI_OCV_KERNEL(GCPUAdd, cv::gapi::core::GAdd)
{
    static void run(const cv::Mat& a, const cv::Mat& b, int dtype, cv::Mat& out)
    {
        cv::add(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUAddC, cv::gapi::core::GAddC)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, int dtype, cv::Mat& out)
    {
        cv::add(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUSub, cv::gapi::core::GSub)
{
    static void run(const cv::Mat& a, const cv::Mat& b, int dtype, cv::Mat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUSubC, cv::gapi::core::GSubC)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, int dtype, cv::Mat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUSubRC, cv::gapi::core::GSubRC)
{
    static void run(const cv::Scalar& a, const cv::Mat& b, int dtype, cv::Mat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMul, cv::gapi::core::GMul)
{
    static void run(const cv::Mat& a, const cv::Mat& b, double scale, int dtype, cv::Mat& out)
    {
        cv::multiply(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMulCOld, cv::gapi::core::GMulCOld)
{
    static void run(const cv::Mat& a, double b, int dtype, cv::Mat& out)
    {
        cv::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMulC, cv::gapi::core::GMulC)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, int dtype, cv::Mat& out)
    {
        cv::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUDiv, cv::gapi::core::GDiv)
{
    static void run(const cv::Mat& a, const cv::Mat& b, double scale, int dtype, cv::Mat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUDivC, cv::gapi::core::GDivC)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, double scale, int dtype, cv::Mat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUDivRC, cv::gapi::core::GDivRC)
{
    static void run(const cv::Scalar& a, const cv::Mat& b, double scale, int dtype, cv::Mat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMask, cv::gapi::core::GMask)
{
    static void run(const cv::Mat& in, const cv::Mat& mask, cv::Mat& out)
    {
        out = cv::Mat::zeros(in.size(), in.type());
        in.copyTo(out, mask);
    }
};

GAPI_OCV_KERNEL(GCPUMean, cv::gapi::core::GMean)
{
    static void run(const cv::Mat& in, cv::Scalar& out)
    {
        out = cv::mean(in);
    }
};

GAPI_OCV_KERNEL(GCPUPolarToCart, cv::gapi::core::GPolarToCart)
{
    static void run(const cv::Mat& magn, const cv::Mat& angle, bool angleInDegrees, cv::Mat& outx, cv::Mat& outy)
    {
        cv::polarToCart(magn, angle, outx, outy, angleInDegrees);
    }
};

GAPI_OCV_KERNEL(GCPUCartToPolar, cv::gapi::core::GCartToPolar)
{
    static void run(const cv::Mat& x, const cv::Mat& y, bool angleInDegrees, cv::Mat& outmagn, cv::Mat& outangle)
    {
        cv::cartToPolar(x, y, outmagn, outangle, angleInDegrees);
    }
};

GAPI_OCV_KERNEL(GCPUPhase, cv::gapi::core::GPhase)
{
    static void run(const cv::Mat &x, const cv::Mat &y, bool angleInDegrees, cv::Mat &out)
    {
        cv::phase(x, y, out, angleInDegrees);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGT, cv::gapi::core::GCmpGT)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_GT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGE, cv::gapi::core::GCmpGE)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_GE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLE, cv::gapi::core::GCmpLE)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_LE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLT, cv::gapi::core::GCmpLT)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_LT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpEQ, cv::gapi::core::GCmpEQ)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_EQ);
    }
};

GAPI_OCV_KERNEL(GCPUCmpNE, cv::gapi::core::GCmpNE)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_NE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGTScalar, cv::gapi::core::GCmpGTScalar)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_GT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGEScalar, cv::gapi::core::GCmpGEScalar)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_GE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLEScalar, cv::gapi::core::GCmpLEScalar)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_LE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLTScalar, cv::gapi::core::GCmpLTScalar)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_LT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpEQScalar, cv::gapi::core::GCmpEQScalar)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_EQ);
    }
};

GAPI_OCV_KERNEL(GCPUCmpNEScalar, cv::gapi::core::GCmpNEScalar)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::compare(a, b, out, cv::CMP_NE);
    }
};

GAPI_OCV_KERNEL(GCPUAnd, cv::gapi::core::GAnd)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::bitwise_and(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUAndS, cv::gapi::core::GAndS)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::bitwise_and(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUOr, cv::gapi::core::GOr)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::bitwise_or(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUOrS, cv::gapi::core::GOrS)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::bitwise_or(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUXor, cv::gapi::core::GXor)
{
    static void run(const cv::Mat& a, const cv::Mat& b, cv::Mat& out)
    {
        cv::bitwise_xor(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUXorS, cv::gapi::core::GXorS)
{
    static void run(const cv::Mat& a, const cv::Scalar& b, cv::Mat& out)
    {
        cv::bitwise_xor(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUNot, cv::gapi::core::GNot)
{
    static void run(const cv::Mat& a, cv::Mat& out)
    {
        cv::bitwise_not(a, out);
    }
};

GAPI_OCV_KERNEL(GCPUSelect, cv::gapi::core::GSelect)
{
    static void run(const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& mask, cv::Mat& out)
    {
        src2.copyTo(out);
        src1.copyTo(out, mask);
    }
};

GAPI_OCV_KERNEL(GCPUMin, cv::gapi::core::GMin)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out)
    {
        out = cv::min(in1, in2);
    }
};

GAPI_OCV_KERNEL(GCPUMax, cv::gapi::core::GMax)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out)
    {
        out = cv::max(in1, in2);
    }
};

GAPI_OCV_KERNEL(GCPUAbsDiff, cv::gapi::core::GAbsDiff)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out)
    {
        cv::absdiff(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPUAbsDiffC, cv::gapi::core::GAbsDiffC)
{
    static void run(const cv::Mat& in1, const cv::Scalar& in2, cv::Mat& out)
    {
        cv::absdiff(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPUSum, cv::gapi::core::GSum)
{
    static void run(const cv::Mat& in, cv::Scalar& out)
    {
        out = cv::sum(in);
    }
};

GAPI_OCV_KERNEL(GCPUAddW, cv::gapi::core::GAddW)
{
    static void run(const cv::Mat& in1, double alpha, const cv::Mat& in2, double beta, double gamma, int dtype, cv::Mat& out)
    {
        cv::addWeighted(in1, alpha, in2, beta, gamma, out, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUNormL1, cv::gapi::core::GNormL1)
{
    static void run(const cv::Mat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_L1);
    }
};

GAPI_OCV_KERNEL(GCPUNormL2, cv::gapi::core::GNormL2)
{
    static void run(const cv::Mat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_L2);
    }
};

GAPI_OCV_KERNEL(GCPUNormInf, cv::gapi::core::GNormInf)
{
    static void run(const cv::Mat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_INF);
    }
};

GAPI_OCV_KERNEL(GCPUIntegral, cv::gapi::core::GIntegral)
{
    static void run(const cv::Mat& in, int sdepth, int sqdepth, cv::Mat& out, cv::Mat& outSq)
    {
        cv::integral(in, out, outSq, sdepth, sqdepth);
    }
};

GAPI_OCV_KERNEL(GCPUThreshold, cv::gapi::core::GThreshold)
{
    static void run(const cv::Mat& in, const cv::Scalar& a, const cv::Scalar& b, int type, cv::Mat& out)
    {
        cv::threshold(in, out, a.val[0], b.val[0], type);
    }
};

GAPI_OCV_KERNEL(GCPUThresholdOT, cv::gapi::core::GThresholdOT)
{
    static void run(const cv::Mat& in, const cv::Scalar& b, int type, cv::Mat& out, cv::Scalar& outScalar)
    {
        outScalar = cv::threshold(in, out, b.val[0], b.val[0], type);
    }
};


GAPI_OCV_KERNEL(GCPUInRange, cv::gapi::core::GInRange)
{
    static void run(const cv::Mat& in, const cv::Scalar& low, const cv::Scalar& up, cv::Mat& out)
    {
        cv::inRange(in, low, up, out);
    }
};

GAPI_OCV_KERNEL(GCPUSplit3, cv::gapi::core::GSplit3)
{
    static void run(const cv::Mat& in, cv::Mat &m1, cv::Mat &m2, cv::Mat &m3)
    {
        std::vector<cv::Mat> outMats = {m1, m2, m3};
        cv::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsence completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
    }
};

GAPI_OCV_KERNEL(GCPUSplit4, cv::gapi::core::GSplit4)
{
    static void run(const cv::Mat& in, cv::Mat &m1, cv::Mat &m2, cv::Mat &m3, cv::Mat &m4)
    {
        std::vector<cv::Mat> outMats = {m1, m2, m3, m4};
        cv::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsence completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
        m4 = outMats[3];
    }
};

GAPI_OCV_KERNEL(GCPUMerge3, cv::gapi::core::GMerge3)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, const cv::Mat& in3, cv::Mat &out)
    {
        std::vector<cv::Mat> inMats = {in1, in2, in3};
        cv::merge(inMats, out);
    }
};

GAPI_OCV_KERNEL(GCPUMerge4, cv::gapi::core::GMerge4)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, const cv::Mat& in3, const cv::Mat& in4, cv::Mat &out)
    {
        std::vector<cv::Mat> inMats = {in1, in2, in3, in4};
        cv::merge(inMats, out);
    }
};

GAPI_OCV_KERNEL(GCPUResize, cv::gapi::core::GResize)
{
    static void run(const cv::Mat& in, cv::Size sz, double fx, double fy, int interp, cv::Mat &out)
    {
        cv::resize(in, out, sz, fx, fy, interp);
    }
};

GAPI_OCV_KERNEL(GCPURemap, cv::gapi::core::GRemap)
{
    static void run(const cv::Mat& in, const cv::Mat& x, const cv::Mat& y, int a, int b, cv::Scalar s, cv::Mat& out)
    {
        cv::remap(in, out, x, y, a, b, s);
    }
};

GAPI_OCV_KERNEL(GCPUFlip, cv::gapi::core::GFlip)
{
    static void run(const cv::Mat& in, int code, cv::Mat& out)
    {
        cv::flip(in, out, code);
    }
};

GAPI_OCV_KERNEL(GCPUCrop, cv::gapi::core::GCrop)
{
    static void run(const cv::Mat& in, cv::Rect rect, cv::Mat& out)
    {
        cv::Mat(in, rect).copyTo(out);
    }
};

GAPI_OCV_KERNEL(GCPUConcatHor, cv::gapi::core::GConcatHor)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out)
    {
        cv::hconcat(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPUConcatVert, cv::gapi::core::GConcatVert)
{
    static void run(const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out)
    {
        cv::vconcat(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPULUT, cv::gapi::core::GLUT)
{
    static void run(const cv::Mat& in, const cv::Mat& lut, cv::Mat& out)
    {
        cv::LUT(in, lut, out);
    }
};

GAPI_OCV_KERNEL(GCPUConvertTo, cv::gapi::core::GConvertTo)
{
    static void run(const cv::Mat& in, int rtype, double alpha, double beta, cv::Mat& out)
    {
        in.convertTo(out, rtype, alpha, beta);
    }
};

GAPI_OCV_KERNEL(GCPUSqrt, cv::gapi::core::GSqrt)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::sqrt(in, out);
    }
};

GAPI_OCV_KERNEL(GCPUNormalize, cv::gapi::core::GNormalize)
{
    static void run(const cv::Mat& src, double a, double b,
                    int norm_type, int ddepth, cv::Mat& out)
    {
        cv::normalize(src, out, a, b, norm_type, ddepth);
    }
};

cv::gapi::GKernelPackage cv::gapi::core::cpu::kernels()
{
    static auto pkg = cv::gapi::kernels
        <  GCPUAdd
         , GCPUAddC
         , GCPUSub
         , GCPUSubC
         , GCPUSubRC
         , GCPUMul
         , GCPUMulC
         , GCPUMulCOld
         , GCPUDiv
         , GCPUDivC
         , GCPUDivRC
         , GCPUMean
         , GCPUMask
         , GCPUPolarToCart
         , GCPUCartToPolar
         , GCPUPhase
         , GCPUCmpGT
         , GCPUCmpGE
         , GCPUCmpLE
         , GCPUCmpLT
         , GCPUCmpEQ
         , GCPUCmpNE
         , GCPUCmpGTScalar
         , GCPUCmpGEScalar
         , GCPUCmpLEScalar
         , GCPUCmpLTScalar
         , GCPUCmpEQScalar
         , GCPUCmpNEScalar
         , GCPUAnd
         , GCPUAndS
         , GCPUOr
         , GCPUOrS
         , GCPUXor
         , GCPUXorS
         , GCPUNot
         , GCPUSelect
         , GCPUMin
         , GCPUMax
         , GCPUAbsDiff
         , GCPUAbsDiffC
         , GCPUSum
         , GCPUAddW
         , GCPUNormL1
         , GCPUNormL2
         , GCPUNormInf
         , GCPUIntegral
         , GCPUThreshold
         , GCPUThresholdOT
         , GCPUInRange
         , GCPUSplit3
         , GCPUSplit4
         , GCPUResize
         , GCPUMerge3
         , GCPUMerge4
         , GCPURemap
         , GCPUFlip
         , GCPUCrop
         , GCPUConcatHor
         , GCPUConcatVert
         , GCPULUT
         , GCPUConvertTo
         , GCPUSqrt
         , GCPUNormalize
         >();
    return pkg;
}
