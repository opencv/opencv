// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/ocl/core.hpp>
#include "backends/ocl/goclcore.hpp"

GAPI_OCL_KERNEL(GOCLAdd, cv::gapi::core::GAdd)
{
    static void run(const cv::UMat& a, const cv::UMat& b, int dtype, cv::UMat& out)
    {
        cv::add(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLAddC, cv::gapi::core::GAddC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, int dtype, cv::UMat& out)
    {
        cv::add(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLSub, cv::gapi::core::GSub)
{
    static void run(const cv::UMat& a, const cv::UMat& b, int dtype, cv::UMat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLSubC, cv::gapi::core::GSubC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, int dtype, cv::UMat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLSubRC, cv::gapi::core::GSubRC)
{
    static void run(const cv::Scalar& a, const cv::UMat& b, int dtype, cv::UMat& out)
    {
        cv::subtract(a, b, out, cv::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMul, cv::gapi::core::GMul)
{
    static void run(const cv::UMat& a, const cv::UMat& b, double scale, int dtype, cv::UMat& out)
    {
        cv::multiply(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMulCOld, cv::gapi::core::GMulCOld)
{
    static void run(const cv::UMat& a, double b, int dtype, cv::UMat& out)
    {
        cv::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMulC, cv::gapi::core::GMulC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, int dtype, cv::UMat& out)
    {
        cv::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLDiv, cv::gapi::core::GDiv)
{
    static void run(const cv::UMat& a, const cv::UMat& b, double scale, int dtype, cv::UMat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLDivC, cv::gapi::core::GDivC)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, double scale, int dtype, cv::UMat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLDivRC, cv::gapi::core::GDivRC)
{
    static void run(const cv::Scalar& a, const cv::UMat& b, double scale, int dtype, cv::UMat& out)
    {
        cv::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMask, cv::gapi::core::GMask)
{
    static void run(const cv::UMat& in, const cv::UMat& mask, cv::UMat& out)
    {
        out = cv::UMat::zeros(in.size(), in.type());
        in.copyTo(out, mask);
    }
};


GAPI_OCL_KERNEL(GOCLMean, cv::gapi::core::GMean)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::mean(in);
    }
};

GAPI_OCL_KERNEL(GOCLPolarToCart, cv::gapi::core::GPolarToCart)
{
    static void run(const cv::UMat& magn, const cv::UMat& angle, bool angleInDegrees, cv::UMat& outx, cv::UMat& outy)
    {
        cv::polarToCart(magn, angle, outx, outy, angleInDegrees);
    }
};

GAPI_OCL_KERNEL(GOCLCartToPolar, cv::gapi::core::GCartToPolar)
{
    static void run(const cv::UMat& x, const cv::UMat& y, bool angleInDegrees, cv::UMat& outmagn, cv::UMat& outangle)
    {
        cv::cartToPolar(x, y, outmagn, outangle, angleInDegrees);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGT, cv::gapi::core::GCmpGT)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGE, cv::gapi::core::GCmpGE)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLE, cv::gapi::core::GCmpLE)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLT, cv::gapi::core::GCmpLT)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpEQ, cv::gapi::core::GCmpEQ)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_EQ);
    }
};

GAPI_OCL_KERNEL(GOCLCmpNE, cv::gapi::core::GCmpNE)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_NE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGTScalar, cv::gapi::core::GCmpGTScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGEScalar, cv::gapi::core::GCmpGEScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_GE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLEScalar, cv::gapi::core::GCmpLEScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLTScalar, cv::gapi::core::GCmpLTScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_LT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpEQScalar, cv::gapi::core::GCmpEQScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_EQ);
    }
};

GAPI_OCL_KERNEL(GOCLCmpNEScalar, cv::gapi::core::GCmpNEScalar)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::compare(a, b, out, cv::CMP_NE);
    }
};

GAPI_OCL_KERNEL(GOCLAnd, cv::gapi::core::GAnd)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::bitwise_and(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLAndS, cv::gapi::core::GAndS)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::bitwise_and(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLOr, cv::gapi::core::GOr)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::bitwise_or(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLOrS, cv::gapi::core::GOrS)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::bitwise_or(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLXor, cv::gapi::core::GXor)
{
    static void run(const cv::UMat& a, const cv::UMat& b, cv::UMat& out)
    {
        cv::bitwise_xor(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLXorS, cv::gapi::core::GXorS)
{
    static void run(const cv::UMat& a, const cv::Scalar& b, cv::UMat& out)
    {
        cv::bitwise_xor(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLNot, cv::gapi::core::GNot)
{
    static void run(const cv::UMat& a, cv::UMat& out)
    {
        cv::bitwise_not(a, out);
    }
};

GAPI_OCL_KERNEL(GOCLSelect, cv::gapi::core::GSelect)
{
    static void run(const cv::UMat& src1, const cv::UMat& src2, const cv::UMat& mask, cv::UMat& out)
    {
        src2.copyTo(out);
        src1.copyTo(out, mask);
    }
};

////TODO: doesn't compiled with UMat
//GAPI_OCL_KERNEL(GOCLMin, cv::gapi::core::GMin)
//{
//    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
//    {
//        out = cv::min(in1, in2);
//    }
//};
//
////TODO: doesn't compiled with UMat
//GAPI_OCL_KERNEL(GOCLMax, cv::gapi::core::GMax)
//{
//    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
//    {
//        out = cv::max(in1, in2);
//    }
//};


GAPI_OCL_KERNEL(GOCLAbsDiff, cv::gapi::core::GAbsDiff)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
    {
        cv::absdiff(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLAbsDiffC, cv::gapi::core::GAbsDiffC)
{
    static void run(const cv::UMat& in1, const cv::Scalar& in2, cv::UMat& out)
    {
        cv::absdiff(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLSum, cv::gapi::core::GSum)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::sum(in);
    }
};

GAPI_OCL_KERNEL(GOCLCountNonZero, cv::gapi::core::GCountNonZero)
{
    static void run(const cv::UMat& in, int& out)
    {
        out = cv::countNonZero(in);
    }
};

GAPI_OCL_KERNEL(GOCLAddW, cv::gapi::core::GAddW)
{
    static void run(const cv::UMat& in1, double alpha, const cv::UMat& in2, double beta, double gamma, int dtype, cv::UMat& out)
    {
        cv::addWeighted(in1, alpha, in2, beta, gamma, out, dtype);
    }
};


GAPI_OCL_KERNEL(GOCLNormL1, cv::gapi::core::GNormL1)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_L1);
    }
};

GAPI_OCL_KERNEL(GOCLNormL2, cv::gapi::core::GNormL2)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_L2);
    }
};

GAPI_OCL_KERNEL(GOCLNormInf, cv::gapi::core::GNormInf)
{
    static void run(const cv::UMat& in, cv::Scalar& out)
    {
        out = cv::norm(in, cv::NORM_INF);
    }
};

GAPI_OCL_KERNEL(GOCLIntegral, cv::gapi::core::GIntegral)
{
    static void run(const cv::UMat& in, int sdepth, int sqdepth, cv::UMat& out, cv::UMat& outSq)
    {
        cv::integral(in, out, outSq, sdepth, sqdepth);
    }
};

GAPI_OCL_KERNEL(GOCLThreshold, cv::gapi::core::GThreshold)
{
    static void run(const cv::UMat& in, const cv::Scalar& a, const cv::Scalar& b, int type, cv::UMat& out)
    {
        cv::threshold(in, out, a.val[0], b.val[0], type);
    }
};

GAPI_OCL_KERNEL(GOCLThresholdOT, cv::gapi::core::GThresholdOT)
{
    static void run(const cv::UMat& in, const cv::Scalar& b, int type, cv::UMat& out, cv::Scalar& outScalar)
    {
        outScalar = cv::threshold(in, out, b.val[0], b.val[0], type);
    }
};


GAPI_OCL_KERNEL(GOCLInRange, cv::gapi::core::GInRange)
{
    static void run(const cv::UMat& in, const cv::Scalar& low, const cv::Scalar& up, cv::UMat& out)
    {
        cv::inRange(in, low, up, out);
    }
};

GAPI_OCL_KERNEL(GOCLSplit3, cv::gapi::core::GSplit3)
{
    static void run(const cv::UMat& in, cv::UMat &m1, cv::UMat &m2, cv::UMat &m3)
    {
        std::vector<cv::UMat> outMats = {m1, m2, m3};
        cv::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsense completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
    }
};

GAPI_OCL_KERNEL(GOCLSplit4, cv::gapi::core::GSplit4)
{
    static void run(const cv::UMat& in, cv::UMat &m1, cv::UMat &m2, cv::UMat &m3, cv::UMat &m4)
    {
        std::vector<cv::UMat> outMats = {m1, m2, m3, m4};
        cv::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsense completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
        m4 = outMats[3];
    }
};

GAPI_OCL_KERNEL(GOCLMerge3, cv::gapi::core::GMerge3)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, const cv::UMat& in3, cv::UMat &out)
    {
        std::vector<cv::UMat> inMats = {in1, in2, in3};
        cv::merge(inMats, out);
    }
};

GAPI_OCL_KERNEL(GOCLMerge4, cv::gapi::core::GMerge4)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, const cv::UMat& in3, const cv::UMat& in4, cv::UMat &out)
    {
        std::vector<cv::UMat> inMats = {in1, in2, in3, in4};
        cv::merge(inMats, out);
    }
};

GAPI_OCL_KERNEL(GOCLResize, cv::gapi::core::GResize)
{
    static void run(const cv::UMat& in, cv::Size sz, double fx, double fy, int interp, cv::UMat &out)
    {
        cv::resize(in, out, sz, fx, fy, interp);
    }
};

GAPI_OCL_KERNEL(GOCLRemap, cv::gapi::core::GRemap)
{
    static void run(const cv::UMat& in, const cv::Mat& x, const cv::Mat& y, int a, int b, cv::Scalar s, cv::UMat& out)
    {
        cv::remap(in, out, x, y, a, b, s);
    }
};

GAPI_OCL_KERNEL(GOCLFlip, cv::gapi::core::GFlip)
{
    static void run(const cv::UMat& in, int code, cv::UMat& out)
    {
        cv::flip(in, out, code);
    }
};

GAPI_OCL_KERNEL(GOCLCrop, cv::gapi::core::GCrop)
{
    static void run(const cv::UMat& in, cv::Rect rect, cv::UMat& out)
    {
        cv::UMat(in, rect).copyTo(out);
    }
};

GAPI_OCL_KERNEL(GOCLConcatHor, cv::gapi::core::GConcatHor)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
    {
        cv::hconcat(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLConcatVert, cv::gapi::core::GConcatVert)
{
    static void run(const cv::UMat& in1, const cv::UMat& in2, cv::UMat& out)
    {
        cv::vconcat(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLLUT, cv::gapi::core::GLUT)
{
    static void run(const cv::UMat& in, const cv::Mat& lut, cv::UMat& out)
    {
        cv::LUT(in, lut, out);
    }
};

GAPI_OCL_KERNEL(GOCLConvertTo, cv::gapi::core::GConvertTo)
{
    static void run(const cv::UMat& in, int rtype, double alpha, double beta, cv::UMat& out)
    {
        in.convertTo(out, rtype, alpha, beta);
    }
};


GAPI_OCL_KERNEL(GOCLTranspose, cv::gapi::core::GTranspose)
{
    static void run(const cv::UMat& in,  cv::UMat& out)
    {
        cv::transpose(in, out);
    }
};

cv::gapi::GKernelPackage cv::gapi::core::ocl::kernels()
{
    static auto pkg = cv::gapi::kernels
        <  GOCLAdd
         , GOCLAddC
         , GOCLSub
         , GOCLSubC
         , GOCLSubRC
         , GOCLMul
         , GOCLMulC
         , GOCLMulCOld
         , GOCLDiv
         , GOCLDivC
         , GOCLDivRC
         , GOCLMean
         , GOCLMask
         , GOCLPolarToCart
         , GOCLCartToPolar
         , GOCLCmpGT
         , GOCLCmpGE
         , GOCLCmpLE
         , GOCLCmpLT
         , GOCLCmpEQ
         , GOCLCmpNE
         , GOCLCmpGTScalar
         , GOCLCmpGEScalar
         , GOCLCmpLEScalar
         , GOCLCmpLTScalar
         , GOCLCmpEQScalar
         , GOCLCmpNEScalar
         , GOCLAnd
         , GOCLAndS
         , GOCLOr
         , GOCLOrS
         , GOCLXor
         , GOCLXorS
         , GOCLNot
         , GOCLSelect
         //, GOCLMin
         //, GOCLMax
         , GOCLAbsDiff
         , GOCLAbsDiffC
         , GOCLSum
         , GOCLCountNonZero
         , GOCLAddW
         , GOCLNormL1
         , GOCLNormL2
         , GOCLNormInf
         , GOCLIntegral
         , GOCLThreshold
         , GOCLThresholdOT
         , GOCLInRange
         , GOCLSplit3
         , GOCLSplit4
         , GOCLResize
         , GOCLMerge3
         , GOCLMerge4
         , GOCLRemap
         , GOCLFlip
         , GOCLCrop
         , GOCLConcatHor
         , GOCLConcatVert
         , GOCLLUT
         , GOCLConvertTo
         , GOCLTranspose
         >();
    return pkg;
}
