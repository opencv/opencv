// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/gpu/core.hpp"
#include "backends/gpu/ggpucore.hpp"

GAPI_GPU_KERNEL(GGPUAdd, cv::gapi::core::GAdd)
{
    static void run(const cv::UMat& a, const cv::UMat& b, int dtype, cv::UMat& out)
    {
        cv::add(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_GPU_KERNEL(GGPUAddC, cv::gapi::core::GAddC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, int dtype, cv::UMat& out)
    {
        cv::add(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_GPU_KERNEL(GGPUSub, cv::gapi::core::GSub)
{
    static void run(const cv::UMat& a, const cv::UMat& b, int dtype, cv::UMat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_GPU_KERNEL(GGPUSubC, cv::gapi::core::GSubC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, int dtype, cv::UMat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_GPU_KERNEL(GGPUSubRC, cv::gapi::core::GSubRC)
{
    static void run(const cv::Scalar& a, const cv::UMat& b, int dtype, cv::UMat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_GPU_KERNEL(GGPUMul, cv::gapi::core::GMul)
{
    static void run(const cv::UMat& a, const cv::UMat& b, double scale, int dtype, cv::UMat& out)
    {
        cv::multiply(a, b, out, scale, dtype);
    }
};

GAPI_GPU_KERNEL(GGPUMulCOld, cv::gapi::core::GMulCOld)
{
    static void run(const cv::UMat& a, double b, int dtype, cv::UMat& out)
    {
        cv::multiply(a, b, out, 1, dtype);
    }
};

GAPI_GPU_KERNEL(GGPUMulC, cv::gapi::core::GMulC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, int dtype, cv::UMat& out)
    {
        cv::multiply(a, b, out, 1, dtype);
    }
};

GAPI_GPU_KERNEL(GGPUDiv, cv::gapi::core::GDiv)
{
    static void run(const cv::UMat& a, const cv::UMat& b, double scale, int dtype, cv::UMat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_GPU_KERNEL(GGPUDivC, cv::gapi::core::GDivC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, double scale, int dtype, cv::UMat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_GPU_KERNEL(GGPUDivRC, cv::gapi::core::GDivRC)
{
    static void run(const cv::Scalar& a, const cv::UMat& b, double scale, int dtype, cv::UMat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_GPU_KERNEL(GGPUMask, cv::gapi::core::GMask)
{
    static void run(const cv::UMat& in, const cv::UMat& mask, cv::UMat& out)
    {
        out = cv::UMat::zeros(in.size(), in.type());
        in.copyTo(out, mask);
    }
};


GAPI_GPU_KERNEL(GGPUMean, cv::gapi::core::GMean)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::mean(in);
    }
};

GAPI_GPU_KERNEL(GGPUPolarToCart, cv::gapi::core::GPolarToCart)
{
    static void run(const cv::UMat& magn, const cv::UMat& angle, bool angleInDegrees, cv::UMat& outx, cv::UMat& outy)
    {
        cv::polarToCart(magn, angle, outx, outy, angleInDegrees);
    }
};

GAPI_GPU_KERNEL(GGPUCartToPolar, cv::gapi::core::GCartToPolar)
{
    static void run(const cv::UMat& x, const cv::UMat& y, bool angleInDegrees, cv::UMat& outmagn, cv::UMat& outangle)
    {
        cv::cartToPolar(x, y, outmagn, outangle, angleInDegrees);
    }
};

GAPI_GPU_KERNEL(GGPUCmpGT, cv::gapi::core::GCmpGT)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GT);
    }
};

GAPI_GPU_KERNEL(GGPUCmpGE, cv::gapi::core::GCmpGE)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GE);
    }
};

GAPI_GPU_KERNEL(GGPUCmpLE, cv::gapi::core::GCmpLE)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LE);
    }
};

GAPI_GPU_KERNEL(GGPUCmpLT, cv::gapi::core::GCmpLT)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LT);
    }
};

GAPI_GPU_KERNEL(GGPUCmpEQ, cv::gapi::core::GCmpEQ)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_EQ);
    }
};

GAPI_GPU_KERNEL(GGPUCmpNE, cv::gapi::core::GCmpNE)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_NE);
    }
};

GAPI_GPU_KERNEL(GGPUCmpGTScalar, cv::gapi::core::GCmpGTScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GT);
    }
};

GAPI_GPU_KERNEL(GGPUCmpGEScalar, cv::gapi::core::GCmpGEScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GE);
    }
};

GAPI_GPU_KERNEL(GGPUCmpLEScalar, cv::gapi::core::GCmpLEScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LE);
    }
};

GAPI_GPU_KERNEL(GGPUCmpLTScalar, cv::gapi::core::GCmpLTScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LT);
    }
};

GAPI_GPU_KERNEL(GGPUCmpEQScalar, cv::gapi::core::GCmpEQScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_EQ);
    }
};

GAPI_GPU_KERNEL(GGPUCmpNEScalar, cv::gapi::core::GCmpNEScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_NE);
    }
};

GAPI_GPU_KERNEL(GGPUAnd, cv::gapi::core::GAnd)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::bitwise_and(a, b, out);
    }
};

GAPI_GPU_KERNEL(GGPUAndS, cv::gapi::core::GAndS)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::bitwise_and(a, b, out);
    }
};

GAPI_GPU_KERNEL(GGPUOr, cv::gapi::core::GOr)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::bitwise_or(a, b, out);
    }
};

GAPI_GPU_KERNEL(GGPUOrS, cv::gapi::core::GOrS)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::bitwise_or(a, b, out);
    }
};

GAPI_GPU_KERNEL(GGPUXor, cv::gapi::core::GXor)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::bitwise_xor(a, b, out);
    }
};

GAPI_GPU_KERNEL(GGPUXorS, cv::gapi::core::GXorS)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::bitwise_xor(a, b, out);
    }
};

GAPI_GPU_KERNEL(GGPUNot, cv::gapi::core::GNot)
{
    static void run(const cv::UMat& a, cv::UMat& out)
    {
        cv::bitwise_not(a, out);
    }
};

GAPI_GPU_KERNEL(GGPUSelect, cv::gapi::core::GSelect)
{
    static void run(const cv::UMat& src1, const cv::UMat& src2, const cv::UMat& mask, cv::UMat& out)
    {
        src2.copyTo(out);
        src1.copyTo(out, mask);
    }
};

////TODO: doesn't compiled with UMat
//GAPI_GPU_KERNEL(GGPUMin, cv::gapi::core::GMin)
//{
//    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
//    {
//        out = cv::min(in1, in2);
//    }
//};
//
////TODO: doesn't compiled with UMat
//GAPI_GPU_KERNEL(GGPUMax, cv::gapi::core::GMax)
//{
//    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
//    {
//        out = cv::max(in1, in2);
//    }
//};


GAPI_GPU_KERNEL(GGPUAbsDiff, cv::gapi::core::GAbsDiff)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
    {
        cv::absdiff(in1, in2, out);
    }
};

GAPI_GPU_KERNEL(GGPUAbsDiffC, cv::gapi::core::GAbsDiffC)
{
    static void run(const cv::UMat& in1, const cv::Scalar& in2, cv::UMat& out)
    {
        cv::absdiff(in1, in2, out);
    }
};

GAPI_GPU_KERNEL(GGPUSum, cv::gapi::core::GSum)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::sum(in);
    }
};

GAPI_GPU_KERNEL(GGPUAddW, cv::gapi::core::GAddW)
{
    static void run(const cv::UMat& in1, double alpha, const cv::UMat& in2, double beta, double gamma, int dtype, cv::UMat& out)
    {
        cv::addWeighted(in1, alpha, in2, beta, gamma, out, dtype);
    }
};


GAPI_GPU_KERNEL(GGPUNormL1, cv::gapi::core::GNormL1)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_L1);
    }
};

GAPI_GPU_KERNEL(GGPUNormL2, cv::gapi::core::GNormL2)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_L2);
    }
};

GAPI_GPU_KERNEL(GGPUNormInf, cv::gapi::core::GNormInf)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_INF);
    }
};

GAPI_GPU_KERNEL(GGPUIntegral, cv::gapi::core::GIntegral)
{
    static void run(const cv::UMat& in, int sdepth, int sqdepth, cv::UMat& out, cv::UMat& outSq)
    {
        cv::integral(in, out, outSq, sdepth, sqdepth);
    }
};

GAPI_GPU_KERNEL(GGPUThreshold, cv::gapi::core::GThreshold)
{
    static void run(const cv::UMat& in, const cv::Scalar& a, const cv::Scalar& b, int type, cv::UMat& out)
    {
        cv::threshold(in, out, a.val[0], b.val[0], type);
    }
};

GAPI_GPU_KERNEL(GGPUThresholdOT, cv::gapi::core::GThresholdOT)
{
    static void run(const cv::UMat& in, const cv::Scalar& b, int type, cv::UMat& out, cv::Scalar& outScalar)
    {
        outScalar = cv::threshold(in, out, b.val[0], b.val[0], type);
    }
};


GAPI_GPU_KERNEL(GGPUInRange, cv::gapi::core::GInRange)
{
    static void run(const cv::UMat& in, const cv::Scalar& low, const cv::Scalar& up, cv::UMat& out)
    {
        cv::inRange(in, low, up, out);
    }
};

GAPI_GPU_KERNEL(GGPUSplit3, cv::gapi::core::GSplit3)
{
    static void run(const cv::UMat& in, cv::UMat &m1, cv::UMat &m2, cv::UMat &m3)
    {
        std::vector<cv::UMat> outMats = {m1, m2, m3};
        cv::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsence completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
    }
};

GAPI_GPU_KERNEL(GGPUSplit4, cv::gapi::core::GSplit4)
{
    static void run(const cv::UMat& in, cv::UMat &m1, cv::UMat &m2, cv::UMat &m3, cv::UMat &m4)
    {
        std::vector<cv::UMat> outMats = {m1, m2, m3, m4};
        cv::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsence completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
        m4 = outMats[3];
    }
};

GAPI_GPU_KERNEL(GGPUMerge3, cv::gapi::core::GMerge3)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, const cv::UMat& in3, cv::UMat &out)
    {
        std::vector<cv::UMat> inMats = {in1, in2, in3};
        cv::merge(inMats, out);
    }
};

GAPI_GPU_KERNEL(GGPUMerge4, cv::gapi::core::GMerge4)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, const cv::UMat& in3, const cv::UMat& in4, cv::UMat &out)
    {
        std::vector<cv::UMat> inMats = {in1, in2, in3, in4};
        cv::merge(inMats, out);
    }
};

GAPI_GPU_KERNEL(GGPUResize, cv::gapi::core::GResize)
{
    static void run(const cv::UMat& in, cv::Size sz, double fx, double fy, int interp, cv::UMat &out)
    {
        cv::resize(in, out, sz, fx, fy, interp);
    }
};

GAPI_GPU_KERNEL(GGPURemap, cv::gapi::core::GRemap)
{
    static void run(const cv::UMat& in, const cv::Mat& x, const cv::Mat& y, int a, int b, cv::Scalar s, cv::UMat& out)
    {
        cv::remap(in, out, x, y, a, b, s);
    }
};

GAPI_GPU_KERNEL(GGPUFlip, cv::gapi::core::GFlip)
{
    static void run(const cv::UMat& in, int code, cv::UMat& out)
    {
        cv::flip(in, out, code);
    }
};

GAPI_GPU_KERNEL(GGPUCrop, cv::gapi::core::GCrop)
{
    static void run(const cv::UMat& in, cv::Rect rect, cv::UMat& out)
    {
        cv::UMat(in, rect).copyTo(out);
    }
};

GAPI_GPU_KERNEL(GGPUConcatHor, cv::gapi::core::GConcatHor)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
    {
        cv::hconcat(in1, in2, out);
    }
};

GAPI_GPU_KERNEL(GGPUConcatVert, cv::gapi::core::GConcatVert)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
    {
        cv::vconcat(in1, in2, out);
    }
};

GAPI_GPU_KERNEL(GGPULUT, cv::gapi::core::GLUT)
{
    static void run(const cv::UMat& in, const cv::Mat& lut, cv::UMat& out)
    {
        cv::LUT(in, lut, out);
    }
};

GAPI_GPU_KERNEL(GGPUConvertTo, cv::gapi::core::GConvertTo)
{
    static void run(const cv::UMat& in, int rtype, double alpha, double beta, cv::UMat& out)
    {
        in.convertTo(out, rtype, alpha, beta);
    }
};

cv::gapi::GKernelPackage cv::gapi::core::gpu::kernels()
{
    static auto pkg = cv::gapi::kernels
        <  GGPUAdd
         , GGPUAddC
         , GGPUSub
         , GGPUSubC
         , GGPUSubRC
         , GGPUMul
         , GGPUMulC
         , GGPUMulCOld
         , GGPUDiv
         , GGPUDivC
         , GGPUDivRC
         , GGPUMean
         , GGPUMask
         , GGPUPolarToCart
         , GGPUCartToPolar
         , GGPUCmpGT
         , GGPUCmpGE
         , GGPUCmpLE
         , GGPUCmpLT
         , GGPUCmpEQ
         , GGPUCmpNE
         , GGPUCmpGTScalar
         , GGPUCmpGEScalar
         , GGPUCmpLEScalar
         , GGPUCmpLTScalar
         , GGPUCmpEQScalar
         , GGPUCmpNEScalar
         , GGPUAnd
         , GGPUAndS
         , GGPUOr
         , GGPUOrS
         , GGPUXor
         , GGPUXorS
         , GGPUNot
         , GGPUSelect
         //, GGPUMin
         //, GGPUMax
         , GGPUAbsDiff
         , GGPUAbsDiffC
         , GGPUSum
         , GGPUAddW
         , GGPUNormL1
         , GGPUNormL2
         , GGPUNormInf
         , GGPUIntegral
         , GGPUThreshold
         , GGPUThresholdOT
         , GGPUInRange
         , GGPUSplit3
         , GGPUSplit4
         , GGPUResize
         , GGPUMerge3
         , GGPUMerge4
         , GGPURemap
         , GGPUFlip
         , GGPUCrop
         , GGPUConcatHor
         , GGPUConcatVert
         , GGPULUT
         , GGPUConvertTo
         >();
    return pkg;
}
