// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/video.hpp>
#include <opencv2/gapi/cpu/video.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#ifdef HAVE_OPENCV_VIDEO
#include <opencv2/video.hpp>
#endif // HAVE_OPENCV_VIDEO

#ifdef HAVE_OPENCV_VIDEO

GAPI_OCV_KERNEL(GCPUBuildOptFlowPyramid, cv::gapi::video::GBuildOptFlowPyramid)
{
    static void run(const cv::Mat              &img,
                    const cv::Size             &winSize,
                    const cv::Scalar           &maxLevel,
                          bool                  withDerivatives,
                          int                   pyrBorder,
                          int                   derivBorder,
                          bool                  tryReuseInputImage,
                          std::vector<cv::Mat> &outPyr,
                          cv::Scalar           &outMaxLevel)
    {
        outMaxLevel = cv::buildOpticalFlowPyramid(img, outPyr, winSize,
                                                  static_cast<int>(maxLevel[0]),
                                                  withDerivatives, pyrBorder,
                                                  derivBorder, tryReuseInputImage);
    }
};

GAPI_OCV_KERNEL(GCPUCalcOptFlowLK, cv::gapi::video::GCalcOptFlowLK)
{
    static void run(const cv::Mat                  &prevImg,
                    const cv::Mat                  &nextImg,
                    const std::vector<cv::Point2f> &prevPts,
                    const std::vector<cv::Point2f> &predPts,
                    const cv::Size                 &winSize,
                    const cv::Scalar               &maxLevel,
                    const cv::TermCriteria         &criteria,
                          int                       flags,
                          double                    minEigThresh,
                          std::vector<cv::Point2f> &outPts,
                          std::vector<uchar>       &status,
                          std::vector<float>       &err)
    {
        if (flags & cv::OPTFLOW_USE_INITIAL_FLOW)
            outPts = predPts;
        cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, outPts, status, err, winSize,
                                 static_cast<int>(maxLevel[0]), criteria, flags, minEigThresh);
    }
};

GAPI_OCV_KERNEL(GCPUCalcOptFlowLKForPyr, cv::gapi::video::GCalcOptFlowLKForPyr)
{
    static void run(const std::vector<cv::Mat>     &prevPyr,
                    const std::vector<cv::Mat>     &nextPyr,
                    const std::vector<cv::Point2f> &prevPts,
                    const std::vector<cv::Point2f> &predPts,
                    const cv::Size                 &winSize,
                    const cv::Scalar               &maxLevel,
                    const cv::TermCriteria         &criteria,
                          int                       flags,
                          double                    minEigThresh,
                          std::vector<cv::Point2f> &outPts,
                          std::vector<uchar>       &status,
                          std::vector<float>       &err)
    {
        if (flags & cv::OPTFLOW_USE_INITIAL_FLOW)
            outPts = predPts;
        cv::calcOpticalFlowPyrLK(prevPyr, nextPyr, prevPts, outPts, status, err, winSize,
                                 static_cast<int>(maxLevel[0]), criteria, flags, minEigThresh);
    }
};

cv::gapi::GKernelPackage cv::gapi::video::cpu::kernels()
{
    static auto pkg = cv::gapi::kernels
        < GCPUBuildOptFlowPyramid
        , GCPUCalcOptFlowLK
        , GCPUCalcOptFlowLKForPyr
        >();
    return pkg;
}

#else

cv::gapi::GKernelPackage cv::gapi::video::cpu::kernels()
{
    return GKernelPackage();
}

#endif // HAVE_OPENCV_VIDEO
