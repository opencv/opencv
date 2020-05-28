// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
#define OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/video.hpp"

#ifdef HAVE_OPENCV_VIDEO
#include <opencv2/video.hpp>
#endif // HAVE_OPENCV_VIDEO



namespace opencv_test
{
namespace
{
G_TYPED_KERNEL(GMinScalar, <GScalar(GScalar,GScalar)>, "custom.MinScalar") {
    static GScalarDesc outMeta(GScalarDesc,GScalarDesc) { return empty_scalar_desc(); }
};
GAPI_OCV_KERNEL(GCPUMinScalar, GMinScalar) {
    static void run(const Scalar &sc1, const Scalar &sc2, Scalar &scOut) {
        scOut = Scalar(std::min(sc1[0], sc2[0]));
    }
};

inline void initTrackingPointsArray(std::vector<cv::Point2f>& points, int width, int height,
                                    int nPointsX, int nPointsY)
{
    if (nPointsX > width || nPointsY > height)
    {
        FAIL() << "Specified points number is too big";
    }

    int stepX = width  / nPointsX;
    int stepY = height / nPointsY;


    points.clear();
    GAPI_Assert((nPointsX >= 0) && (nPointsY) >= 0);
    points.reserve(static_cast<size_t>(nPointsX * nPointsY));

    for (int x = stepX / 2; x < width; x += stepX)
    {
        for (int y = stepY / 2; y < height; y += stepY)
        {
            Point2f pt(static_cast<float>(x), static_cast<float>(y));
            points.push_back(pt);
        }
    }
}

struct BuildOpticalFlowPyramidTestOutput
{
    BuildOpticalFlowPyramidTestOutput(std::vector<Mat> &pyr, int maxLvl) :
                                      pyramid(pyr), maxLevel(maxLvl) { }
    std::vector<Mat> &pyramid;
    int               maxLevel = 0;
};

template<typename Type>
struct OptFlowLKTestInput
{
    Type& prevData;
    Type& nextData;
    std::vector<cv::Point2f>& prevPoints;
};

struct OptFlowLKTestOutput
{
    std::vector<cv::Point2f> &nextPoints;
    std::vector<uchar>       &statuses;
    std::vector<float>       &errors;
};

struct BuildOpticalFlowPyramidTestParams
{
    BuildOpticalFlowPyramidTestParams(): fileName(""), winSize(-1), maxLevel(-1),
                                         withDerivatives(false), pyrBorder(-1),
                                         derivBorder(-1), tryReuseInputImage(false) { }

    BuildOpticalFlowPyramidTestParams(const std::string& name, int winSz, int maxLvl,
                                      bool withDeriv, int pBorder, int dBorder,
                                      bool tryReuse, const GCompileArgs& compArgs):

                                      fileName(name), winSize(winSz), maxLevel(maxLvl),
                                      withDerivatives(withDeriv), pyrBorder(pBorder),
                                      derivBorder(dBorder), tryReuseInputImage(tryReuse),
                                      compileArgs(compArgs) { }

    std::string fileName    = "";
    int winSize             = -1;
    int maxLevel            = -1;
    bool withDerivatives    = false;
    int pyrBorder           = -1;
    int derivBorder         = -1;
    bool tryReuseInputImage = false;
    cv::GCompileArgs compileArgs;
};

struct OptFlowLKTestParams
{
    OptFlowLKTestParams(): fileNamePattern(""), format(1), channels(0), pointsNum{0, 0},
                           winSize(0), maxLevel(3), minEigThreshold(1e-4), flags(0) { }

    OptFlowLKTestParams(const std::string& namePat, int chans,
                        const std::tuple<int,int>& ptsNum, int winSz,
                        const cv::TermCriteria& crit, const cv::GCompileArgs& compArgs,
                        int flgs = 0, int fmt = 1, int maxLvl = 3, double minEigThresh = 1e-4):

                        fileNamePattern(namePat), format(fmt), channels(chans),
                        pointsNum(ptsNum), winSize(winSz), maxLevel(maxLvl),
                        criteria(crit), minEigThreshold(minEigThresh), compileArgs(compArgs),
                        flags(flgs) { }

    std::string fileNamePattern   = "";
    int format                    = 1;
    int channels                  = 0;
    std::tuple<int,int> pointsNum = std::make_tuple(0, 0);
    int winSize                   = 0;
    int maxLevel                  = 3;
    cv::TermCriteria criteria;
    double minEigThreshold        = 1e-4;
    cv::GCompileArgs compileArgs;
    int flags                     = 0;
};

#ifdef HAVE_OPENCV_VIDEO

inline GComputation runOCVnGAPIBuildOptFlowPyramid(TestFunctional& testInst,
                                                   const BuildOpticalFlowPyramidTestParams& params,
                                                   BuildOpticalFlowPyramidTestOutput& outOCV,
                                                   BuildOpticalFlowPyramidTestOutput& outGAPI)
{
    testInst.initMatFromImage(CV_8UC1, params.fileName);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        outOCV.maxLevel = cv::buildOpticalFlowPyramid(testInst.in_mat1, outOCV.pyramid,
                                                      Size(params.winSize, params.winSize),
                                                      params.maxLevel, params.withDerivatives,
                                                      params.pyrBorder, params.derivBorder,
                                                      params.tryReuseInputImage);
    }

    // G-API code //////////////////////////////////////////////////////////////
    GMat         in;
    GArray<GMat> out;
    GScalar      outMaxLevel;
    std::tie(out, outMaxLevel) =
         cv::gapi::buildOpticalFlowPyramid(in, Size(params.winSize, params.winSize),
                                           params.maxLevel, params.withDerivatives,
                                           params.pyrBorder, params.derivBorder,
                                           params.tryReuseInputImage);

    GComputation c(GIn(in), GOut(out, outMaxLevel));

    Scalar outMaxLevelSc;
    c.apply(gin(testInst.in_mat1), gout(outGAPI.pyramid, outMaxLevelSc),
            std::move(const_cast<GCompileArgs&>(params.compileArgs)));
    outGAPI.maxLevel = static_cast<int>(outMaxLevelSc[0]);

    return c;
}

template<typename GType, typename Type>
cv::GComputation runOCVnGAPIOptFlowLK(OptFlowLKTestInput<Type>& in,
                                      int width, int height,
                                      const OptFlowLKTestParams& params,
                                      OptFlowLKTestOutput& ocvOut,
                                      OptFlowLKTestOutput& gapiOut)
{

    int nPointsX = 0, nPointsY = 0;
    std::tie(nPointsX, nPointsY) = params.pointsNum;

    initTrackingPointsArray(in.prevPoints, width, height, nPointsX, nPointsY);

    cv::Size winSize(params.winSize, params.winSize);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::calcOpticalFlowPyrLK(in.prevData, in.nextData, in.prevPoints,
                                 ocvOut.nextPoints, ocvOut.statuses, ocvOut.errors,
                                 winSize, params.maxLevel, params.criteria,
                                 params.flags, params.minEigThreshold);
    }

    // G-API code //////////////////////////////////////////////////////////////
    {
        GType               inPrev,  inNext;
        GArray<cv::Point2f> prevPts, predPts, nextPts;
        GArray<uchar>       statuses;
        GArray<float>       errors;
        std::tie(nextPts, statuses, errors) = cv::gapi::calcOpticalFlowPyrLK(
                                                    inPrev, inNext,
                                                    prevPts, predPts, winSize,
                                                    params.maxLevel, params.criteria,
                                                    params.flags, params.minEigThreshold);

        cv::GComputation c(cv::GIn(inPrev, inNext, prevPts, predPts),
                           cv::GOut(nextPts, statuses, errors));

        c.apply(cv::gin(in.prevData, in.nextData, in.prevPoints, std::vector<cv::Point2f>{ }),
                cv::gout(gapiOut.nextPoints, gapiOut.statuses, gapiOut.errors),
                std::move(const_cast<cv::GCompileArgs&>(params.compileArgs)));

        return c;
    }
}

inline cv::GComputation runOCVnGAPIOptFlowLK(TestFunctional& testInst,
                                             std::vector<cv::Point2f>& inPts,
                                             const OptFlowLKTestParams& params,
                                             OptFlowLKTestOutput& ocvOut,
                                             OptFlowLKTestOutput& gapiOut)
{
    testInst.initMatsFromImages(params.channels,
                                params.fileNamePattern,
                                params.format);

    OptFlowLKTestInput<cv::Mat> in{ testInst.in_mat1, testInst.in_mat2, inPts };

    return runOCVnGAPIOptFlowLK<cv::GMat>(in,
                                          testInst.in_mat1.cols,
                                          testInst.in_mat1.rows,
                                          params,
                                          ocvOut,
                                          gapiOut);
}

inline cv::GComputation runOCVnGAPIOptFlowLKForPyr(TestFunctional& testInst,
                                                   OptFlowLKTestInput<std::vector<cv::Mat>>& in,
                                                   const OptFlowLKTestParams& params,
                                                   bool withDeriv,
                                                   OptFlowLKTestOutput& ocvOut,
                                                   OptFlowLKTestOutput& gapiOut)
{
    testInst.initMatsFromImages(params.channels,
                                params.fileNamePattern,
                                params.format);

    cv::Size winSize(params.winSize, params.winSize);

    OptFlowLKTestParams updatedParams(params);
    updatedParams.maxLevel = cv::buildOpticalFlowPyramid(testInst.in_mat1, in.prevData,
                                                         winSize, params.maxLevel, withDeriv);
    updatedParams.maxLevel = cv::buildOpticalFlowPyramid(testInst.in_mat2, in.nextData,
                                                         winSize, params.maxLevel, withDeriv);


    return runOCVnGAPIOptFlowLK<cv::GArray<cv::GMat>>(in,
                                                      testInst.in_mat1.cols,
                                                      testInst.in_mat1.rows,
                                                      updatedParams,
                                                      ocvOut,
                                                      gapiOut);
}

inline GComputation runOCVnGAPIOptFlowPipeline(TestFunctional& testInst,
                                               const BuildOpticalFlowPyramidTestParams& params,
                                               OptFlowLKTestOutput& outOCV,
                                               OptFlowLKTestOutput& outGAPI,
                                               std::vector<Point2f>& prevPoints)
{
    testInst.initMatsFromImages(3, params.fileName, 1);

    initTrackingPointsArray(prevPoints, testInst.in_mat1.cols, testInst.in_mat1.rows, 15, 15);

    Size winSize = Size(params.winSize, params.winSize);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<Mat> pyr1, pyr2;
        int maxLevel1 = cv::buildOpticalFlowPyramid(testInst.in_mat1, pyr1, winSize,
                                                    params.maxLevel, params.withDerivatives,
                                                    params.pyrBorder, params.derivBorder,
                                                    params.tryReuseInputImage);
        int maxLevel2 = cv::buildOpticalFlowPyramid(testInst.in_mat2, pyr2, winSize,
                                                    params.maxLevel, params.withDerivatives,
                                                    params.pyrBorder, params.derivBorder,
                                                    params.tryReuseInputImage);
        cv::calcOpticalFlowPyrLK(pyr1, pyr2, prevPoints,
                                 outOCV.nextPoints, outOCV.statuses, outOCV.errors,
                                 winSize, std::min(maxLevel1, maxLevel2));
    }

    // G-API code //////////////////////////////////////////////////////////////
    GMat                in1,        in2;
    GArray<GMat>        gpyr1,      gpyr2;
    GScalar             gmaxLevel1, gmaxLevel2;
    GArray<cv::Point2f> gprevPts, gpredPts, gnextPts;
    GArray<uchar>       gstatuses;
    GArray<float>       gerrors;

    std::tie(gpyr1, gmaxLevel1) = cv::gapi::buildOpticalFlowPyramid(
                                      in1, winSize, params.maxLevel,
                                      params.withDerivatives, params.pyrBorder,
                                      params.derivBorder, params.tryReuseInputImage);

    std::tie(gpyr2, gmaxLevel2) = cv::gapi::buildOpticalFlowPyramid(
                                      in2, winSize, params.maxLevel,
                                      params.withDerivatives, params.pyrBorder,
                                      params.derivBorder, params.tryReuseInputImage);

    GScalar gmaxLevel = GMinScalar::on(gmaxLevel1, gmaxLevel2);

    std::tie(gnextPts, gstatuses, gerrors) = cv::gapi::calcOpticalFlowPyrLK(
                                              gpyr1, gpyr2, gprevPts, gpredPts, winSize,
                                              gmaxLevel);

    cv::GComputation c(GIn(in1, in2, gprevPts, gpredPts), cv::GOut(gnextPts, gstatuses, gerrors));

    c.apply(cv::gin(testInst.in_mat1, testInst.in_mat2, prevPoints, std::vector<cv::Point2f>{ }),
            cv::gout(outGAPI.nextPoints, outGAPI.statuses, outGAPI.errors),
            std::move(const_cast<cv::GCompileArgs&>(params.compileArgs)));

    return c;
}

#else // !HAVE_OPENCV_VIDEO

inline cv::GComputation runOCVnGAPIBuildOptFlowPyramid(TestFunctional&,
                                                       const BuildOpticalFlowPyramidTestParams&,
                                                       BuildOpticalFlowPyramidTestOutput&,
                                                       BuildOpticalFlowPyramidTestOutput&)
{
    GAPI_Assert(0 && "This function shouldn't be called without opencv_video");
}

inline cv::GComputation runOCVnGAPIOptFlowLK(TestFunctional&,
                                             std::vector<cv::Point2f>&,
                                             const OptFlowLKTestParams&,
                                             OptFlowLKTestOutput&,
                                             OptFlowLKTestOutput&)
{
    GAPI_Assert(0 && "This function shouldn't be called without opencv_video");
}

inline cv::GComputation runOCVnGAPIOptFlowLKForPyr(TestFunctional&,
                                                   OptFlowLKTestInput<std::vector<cv::Mat>>&,
                                                   const OptFlowLKTestParams&,
                                                   bool,
                                                   OptFlowLKTestOutput&,
                                                   OptFlowLKTestOutput&)
{
    GAPI_Assert(0 && "This function shouldn't be called without opencv_video");
}

inline GComputation runOCVnGAPIOptFlowPipeline(TestFunctional&,
                                               const BuildOpticalFlowPyramidTestParams&,
                                               OptFlowLKTestOutput&,
                                               OptFlowLKTestOutput&,
                                               std::vector<Point2f>&)
{
    GAPI_Assert(0 && "This function shouldn't be called without opencv_video");
}

#endif // HAVE_OPENCV_VIDEO

inline void compareOutputPyramids(const BuildOpticalFlowPyramidTestOutput& outOCV,
                                  const BuildOpticalFlowPyramidTestOutput& outGAPI)
{
    GAPI_Assert(outGAPI.maxLevel == outOCV.maxLevel);
    GAPI_Assert(outOCV.maxLevel >= 0);
    size_t maxLevel = static_cast<size_t>(outOCV.maxLevel);
    for (size_t i = 0; i <= maxLevel; i++)
    {
        EXPECT_TRUE(AbsExact().to_compare_f()(outOCV.pyramid[i], outGAPI.pyramid[i]));
    }
}

template <typename Elem>
inline bool compareVectorsAbsExactForOptFlow(std::vector<Elem> outOCV, std::vector<Elem> outGAPI)
{
    return AbsExactVector<Elem>().to_compare_f()(outOCV, outGAPI);
}

inline void compareOutputsOptFlow(const OptFlowLKTestOutput& outOCV,
                                  const OptFlowLKTestOutput& outGAPI)
{
    EXPECT_TRUE(compareVectorsAbsExactForOptFlow(outGAPI.nextPoints, outOCV.nextPoints));
    EXPECT_TRUE(compareVectorsAbsExactForOptFlow(outGAPI.statuses,   outOCV.statuses));
    EXPECT_TRUE(compareVectorsAbsExactForOptFlow(outGAPI.errors,     outOCV.errors));
}

inline std::ostream& operator<<(std::ostream& os, const cv::TermCriteria& criteria)
{
    os << "{";
    switch (criteria.type) {
    case cv::TermCriteria::COUNT:
        os << "COUNT; ";
        break;
    case cv::TermCriteria::EPS:
        os << "EPS; ";
        break;
    case cv::TermCriteria::COUNT | cv::TermCriteria::EPS:
        os << "COUNT | EPS; ";
        break;
    default:
        os << "TypeUndefined; ";
        break;
    };

    return os << criteria.maxCount << "; " << criteria.epsilon <<"}";
}
} // namespace
} // namespace opencv_test


#endif // OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
