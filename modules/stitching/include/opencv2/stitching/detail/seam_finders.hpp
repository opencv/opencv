/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_STITCHING_SEAM_FINDERS_HPP__
#define __OPENCV_STITCHING_SEAM_FINDERS_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/opencv_modules.hpp"

namespace cv {
namespace detail {

class CV_EXPORTS_AS(SeamFinder) SeamFinder
{
public:
    virtual ~SeamFinder() {}
    CV_WRAP virtual void find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                      std::vector<Mat> &masks) = 0;
};


class CV_EXPORTS NoSeamFinder : public SeamFinder
{
public:
    void find(const std::vector<Mat>&, const std::vector<Point>&, std::vector<Mat>&) {}
};


class CV_EXPORTS PairwiseSeamFinder : public SeamFinder
{
public:
    virtual void find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                      std::vector<Mat> &masks);

protected:
    void run();
    virtual void findInPair(size_t first, size_t second, Rect roi) = 0;

    std::vector<Mat> images_;
    std::vector<Size> sizes_;
    std::vector<Point> corners_;
    std::vector<Mat> masks_;
};


class CV_EXPORTS VoronoiSeamFinder : public PairwiseSeamFinder
{
public:
    virtual void find(const std::vector<Size> &size, const std::vector<Point> &corners,
                      std::vector<Mat> &masks);
private:
    void findInPair(size_t first, size_t second, Rect roi);
};


class CV_EXPORTS GraphCutSeamFinderBase
{
public:
    enum { COST_COLOR, COST_COLOR_GRAD };
};


class CV_EXPORTS_AS(GraphCutSeamFinder) GraphCutSeamFinder : public SeamFinder, public GraphCutSeamFinderBase
{
public:
    CV_WRAP GraphCutSeamFinder(int cost_type = GraphCutSeamFinder::COST_COLOR_GRAD, float terminal_cost = 10000.f,
                       float bad_region_penalty = 1000.f);

    ~GraphCutSeamFinder();

    void find(const std::vector<Mat> &src, const std::vector<Point> &corners,
              std::vector<Mat> &masks);

private:
    // To avoid GCGraph dependency
    class Impl;
    Ptr<Impl> impl_;
};


#ifdef HAVE_OPENCV_GPU
class CV_EXPORTS GraphCutSeamFinderGpu : public GraphCutSeamFinderBase, public PairwiseSeamFinder
{
public:
    GraphCutSeamFinderGpu(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                          float bad_region_penalty = 1000.f)
                          : cost_type_(cost_type), terminal_cost_(terminal_cost), 
                            bad_region_penalty_(bad_region_penalty) {}

    void find(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners, 
              std::vector<cv::Mat> &masks);
    void findInPair(size_t first, size_t second, Rect roi);

private:
    void setGraphWeightsColor(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &mask1, const cv::Mat &mask2,
                              cv::Mat &terminals, cv::Mat &leftT, cv::Mat &rightT, cv::Mat &top, cv::Mat &bottom);
    void setGraphWeightsColorGrad(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &dx1, const cv::Mat &dx2, 
                                  const cv::Mat &dy1, const cv::Mat &dy2, const cv::Mat &mask1, const cv::Mat &mask2, 
                                  cv::Mat &terminals, cv::Mat &leftT, cv::Mat &rightT, cv::Mat &top, cv::Mat &bottom);
    std::vector<Mat> dx_, dy_;
    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
};
#endif

} // namespace detail
} // namespace cv

#endif // __OPENCV_STITCHING_SEAM_FINDERS_HPP__
