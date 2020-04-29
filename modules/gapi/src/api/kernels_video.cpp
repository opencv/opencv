// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/video.hpp>

namespace cv { namespace gapi {
using namespace video;

GBuildPyrOutput buildOpticalFlowPyramid(const GMat    &img,
                                        const Size    &winSize,
                                        const GScalar &maxLevel,
                                              bool     withDerivatives,
                                              int      pyrBorder,
                                              int      derivBorder,
                                              bool     tryReuseInputImage)
{
    return GBuildOptFlowPyramid::on(img, winSize, maxLevel, withDerivatives, pyrBorder,
                                    derivBorder, tryReuseInputImage);
}

GOptFlowLKOutput calcOpticalFlowPyrLK(const GMat                    &prevImg,
                                      const GMat                    &nextImg,
                                      const cv::GArray<cv::Point2f> &prevPts,
                                      const cv::GArray<cv::Point2f> &predPts,
                                      const Size                    &winSize,
                                      const GScalar                 &maxLevel,
                                      const TermCriteria            &criteria,
                                            int                      flags,
                                            double                   minEigThresh)
{
    return GCalcOptFlowLK::on(prevImg, nextImg, prevPts, predPts, winSize, maxLevel,
                              criteria, flags, minEigThresh);
}

GOptFlowLKOutput calcOpticalFlowPyrLK(const cv::GArray<cv::GMat>    &prevPyr,
                                      const cv::GArray<cv::GMat>    &nextPyr,
                                      const cv::GArray<cv::Point2f> &prevPts,
                                      const cv::GArray<cv::Point2f> &predPts,
                                      const Size                    &winSize,
                                      const GScalar                 &maxLevel,
                                      const TermCriteria            &criteria,
                                            int                      flags,
                                            double                   minEigThresh)
{
    return GCalcOptFlowLKForPyr::on(prevPyr, nextPyr, prevPts, predPts, winSize, maxLevel,
                                    criteria, flags, minEigThresh);
}

} //namespace gapi
} //namespace cv
