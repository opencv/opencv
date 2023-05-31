// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gcall.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/core.hpp>

#include <tuple>
#include <numeric>

namespace cv { namespace gapi {

GMat add(const GMat& src1, const GMat& src2, int dtype)
{
    return core::GAdd::on(src1, src2, dtype);
}

GMat addC(const GMat& src1, const GScalar& c, int dtype)
{
    return core::GAddC::on(src1, c, dtype);
}

GMat addC(const GScalar& c, const GMat& src1, int dtype)
{
    return core::GAddC::on(src1, c, dtype);
}

GMat sub(const GMat& src1, const GMat& src2, int dtype)
{
    return core::GSub::on(src1, src2, dtype);
}

GMat subC(const GMat& src1, const GScalar& c, int dtype)
{
    return core::GSubC::on(src1, c, dtype);
}

GMat subRC(const GScalar& c, const GMat& src, int dtype)
{
    return core::GSubRC::on(c, src, dtype);
}

GMat mul(const GMat& src1, const GMat& src2, double scale, int dtype)
{
    return core::GMul::on(src1, src2, scale, dtype);
}

GMat mulC(const GMat& src, double scale, int dtype)
{
    return core::GMulCOld::on(src, scale, dtype);
}

GMat mulC(const GMat& src, const GScalar& multiplier, int dtype)
{
    return core::GMulC::on(src, multiplier, dtype);
}

GMat mulC(const GScalar& multiplier, const GMat& src, int dtype)
{
    return core::GMulC::on(src, multiplier, dtype);
}

GMat div(const GMat& src1, const GMat& src2, double scale, int dtype)
{
    return core::GDiv::on(src1, src2, scale, dtype);
}

GMat divC(const GMat& src, const GScalar& divisor, double scale, int dtype)
{
    return core::GDivC::on(src, divisor, scale, dtype);
}

GMat divRC(const GScalar& divident, const GMat& src, double scale, int dtype)
{
    return core::GDivRC::on(divident, src, scale, dtype);
}

GScalar mean(const GMat& src)
{
    return core::GMean::on(src);
}

GMat mask(const GMat& src, const GMat& mask)
{
    return core::GMask::on(src, mask);
}

std::tuple<GMat, GMat> polarToCart(const GMat& magnitude, const GMat& angle,
                                   bool angleInDegrees)
{
    return core::GPolarToCart::on(magnitude, angle, angleInDegrees);
}

std::tuple<GMat, GMat> cartToPolar(const GMat& x, const GMat& y,
                                   bool angleInDegrees)
{
    return core::GCartToPolar::on(x, y, angleInDegrees);
}

GMat phase(const GMat &x, const GMat &y, bool angleInDegrees)
{
    return core::GPhase::on(x, y, angleInDegrees);
}

GMat cmpGT(const GMat& src1, const GMat& src2)
{
    return core::GCmpGT::on(src1, src2);
}

GMat cmpLT(const GMat& src1, const GMat& src2)
{
    return core::GCmpLT::on(src1, src2);
}

GMat cmpGE(const GMat& src1, const GMat& src2)
{
    return core::GCmpGE::on(src1, src2);
}

GMat cmpLE(const GMat& src1, const GMat& src2)
{
    return core::GCmpLE::on(src1, src2);
}

GMat cmpEQ(const GMat& src1, const GMat& src2)
{
    return core::GCmpEQ::on(src1, src2);
}

GMat cmpNE(const GMat& src1, const GMat& src2)
{
    return core::GCmpNE::on(src1, src2);
}

GMat cmpGT(const GMat& src1, const GScalar& src2)
{
    return core::GCmpGTScalar::on(src1, src2);
}

GMat cmpLT(const GMat& src1, const GScalar& src2)
{
    return core::GCmpLTScalar::on(src1, src2);
}

GMat cmpGE(const GMat& src1, const GScalar& src2)
{
    return core::GCmpGEScalar::on(src1, src2);
}

GMat cmpLE(const GMat& src1, const GScalar& src2)
{
    return core::GCmpLEScalar::on(src1, src2);
}

GMat cmpEQ(const GMat& src1, const GScalar& src2)
{
    return core::GCmpEQScalar::on(src1, src2);
}

GMat cmpNE(const GMat& src1, const GScalar& src2)
{
    return core::GCmpNEScalar::on(src1, src2);
}

GMat min(const GMat& src1, const GMat& src2)
{
    return core::GMin::on(src1, src2);
}

GMat max(const GMat& src1, const GMat& src2)
{
    return core::GMax::on(src1, src2);
}

GMat absDiff(const GMat& src1, const GMat& src2)
{
    return core::GAbsDiff::on(src1, src2);
}

GMat absDiffC(const GMat& src, const GScalar& c)
{
    return core::GAbsDiffC::on(src, c);
}

GMat bitwise_and(const GMat& src1, const GMat& src2)
{
    return core::GAnd::on(src1, src2);
}

GMat bitwise_and(const GMat& src1, const GScalar& src2)
{
    return core::GAndS::on(src1, src2);
}

GMat bitwise_or(const GMat& src1, const GMat& src2)
{
    return core::GOr::on(src1, src2);
}

GMat bitwise_or(const GMat& src1, const GScalar& src2)
{
    return core::GOrS::on(src1, src2);
}

GMat bitwise_xor(const GMat& src1, const GMat& src2)
{
    return core::GXor::on(src1, src2);
}

GMat bitwise_xor(const GMat& src1, const GScalar& src2)
{
    return core::GXorS::on(src1, src2);
}

GMat bitwise_not(const GMat& src1)
{
    return core::GNot::on(src1);
}

GMat select(const GMat& src1, const GMat& src2, const GMat& mask)
{
    return core::GSelect::on(src1, src2, mask);
}

GScalar sum(const GMat& src)
{
    return core::GSum::on(src);
}

GOpaque<int> countNonZero(const GMat& src)
{
    return core::GCountNonZero::on(src);
}

GMat addWeighted(const GMat& src1, double alpha, const GMat& src2, double beta, double gamma, int dtype)
{
    return core::GAddW::on(src1, alpha, src2, beta, gamma, dtype);
}

GScalar normL1(const GMat& src)
{
    return core::GNormL1::on(src);
}

GScalar normL2(const GMat& src)
{
    return core::GNormL2::on(src);
}

GScalar normInf(const GMat& src)
{
    return core::GNormInf::on(src);
}

std::tuple<GMat, GMat> integral(const GMat& src, int sdepth, int sqdepth)
{
    return core::GIntegral::on(src, sdepth, sqdepth);
}

GMat threshold(const GMat& src, const GScalar& thresh, const GScalar& maxval, int type)
{
    GAPI_Assert(type != cv::THRESH_TRIANGLE && type != cv::THRESH_OTSU);
    return core::GThreshold::on(src, thresh, maxval, type);
}

std::tuple<GMat, GScalar> threshold(const GMat& src, const GScalar& maxval, int type)
{
    GAPI_Assert(type == cv::THRESH_TRIANGLE || type == cv::THRESH_OTSU);
    return core::GThresholdOT::on(src, maxval, type);
}

GMat inRange(const GMat& src, const GScalar& threshLow, const GScalar& threshUp)
{
    return core::GInRange::on(src, threshLow, threshUp);
}

std::tuple<GMat, GMat, GMat> split3(const GMat& src)
{
    return core::GSplit3::on(src);
}

std::tuple<GMat, GMat, GMat, GMat> split4(const GMat& src)
{
    return core::GSplit4::on(src);
}

GMat merge3(const GMat& src1, const GMat& src2, const GMat& src3)
{
    return core::GMerge3::on(src1, src2, src3);
}

GMat merge4(const GMat& src1, const GMat& src2, const GMat& src3, const GMat& src4)
{
    return core::GMerge4::on(src1, src2, src3, src4);
}

GMat remap(const GMat& src, const Mat& map1, const Mat& map2,
           int interpolation, int borderMode,
           const Scalar& borderValue)
{
    return core::GRemap::on(src, map1, map2, interpolation, borderMode, borderValue);
}

GMat flip(const GMat& src, int flipCode)
{
    return core::GFlip::on(src, flipCode);
}

GMat crop(const GMat& src, const Rect& rect)
{
    return core::GCrop::on(src, rect);
}

GMat concatHor(const GMat& src1, const GMat& src2)
{
    return core::GConcatHor::on(src1, src2);
}

GMat concatHor(const std::vector<GMat>& v)
{
    GAPI_Assert(v.size() >= 2);
    return std::accumulate(v.begin()+1, v.end(), v[0], core::GConcatHor::on);
}

GMat concatVert(const GMat& src1, const GMat& src2)
{
    return core::GConcatVert::on(src1, src2);
}

GMat concatVert(const std::vector<GMat>& v)
{
    GAPI_Assert(v.size() >= 2);
    return std::accumulate(v.begin()+1, v.end(), v[0], core::GConcatVert::on);
}

GMat LUT(const GMat& src, const Mat& lut)
{
    return core::GLUT::on(src, lut);
}

GMat convertTo(const GMat& m, int rtype, double alpha, double beta)
{
    return core::GConvertTo::on(m, rtype, alpha, beta);
}

GMat sqrt(const GMat& src)
{
    return core::GSqrt::on(src);
}

GMat normalize(const GMat& _src, double a, double b,
               int norm_type, int ddepth)
{
    return core::GNormalize::on(_src, a, b, norm_type, ddepth);
}

GMat warpPerspective(const GMat& src, const Mat& M, const Size& dsize, int flags,
                     int borderMode, const Scalar& borderValue)
{
    return core::GWarpPerspective::on(src, M, dsize, flags, borderMode, borderValue);
}

GMat warpAffine(const GMat& src, const Mat& M, const Size& dsize, int flags,
                int borderMode, const Scalar& borderValue)
{
    return core::GWarpAffine::on(src, M, dsize, flags, borderMode, borderValue);
}

std::tuple<GOpaque<double>,GMat,GMat> kmeans(const GMat& data, const int K, const GMat& bestLabels,
                                             const TermCriteria& criteria, const int attempts,
                                             const KmeansFlags flags)
{
    return core::GKMeansND::on(data, K, bestLabels, criteria, attempts, flags);
}

std::tuple<GOpaque<double>,GMat,GMat> kmeans(const GMat& data, const int K,
                                             const TermCriteria& criteria, const int attempts,
                                             const KmeansFlags flags)
{
    return core::GKMeansNDNoInit::on(data, K, criteria, attempts, flags);
}

std::tuple<GOpaque<double>,GArray<int>,GArray<Point2f>> kmeans(const GArray<Point2f>& data,
                                                               const int              K,
                                                               const GArray<int>&     bestLabels,
                                                               const TermCriteria&    criteria,
                                                               const int              attempts,
                                                               const KmeansFlags      flags)
{
    return core::GKMeans2D::on(data, K, bestLabels, criteria, attempts, flags);
}

std::tuple<GOpaque<double>,GArray<int>,GArray<Point3f>> kmeans(const GArray<Point3f>& data,
                                                               const int              K,
                                                               const GArray<int>&     bestLabels,
                                                               const TermCriteria&    criteria,
                                                               const int              attempts,
                                                               const KmeansFlags      flags)
{
    return core::GKMeans3D::on(data, K, bestLabels, criteria, attempts, flags);
}


GMat transpose(const GMat& src)
{
    return core::GTranspose::on(src);
}

GOpaque<Size> streaming::size(const GMat& src)
{
    return streaming::GSize::on(src);
}

GOpaque<Size> streaming::size(const GOpaque<Rect>& r)
{
    return streaming::GSizeR::on(r);
}

GOpaque<Size> streaming::size(const GFrame& src)
{
    return streaming::GSizeMF::on(src);
}

} //namespace gapi
} //namespace cv
