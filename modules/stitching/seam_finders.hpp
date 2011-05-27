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
#ifndef __OPENCV_SEAM_FINDERS_HPP__
#define __OPENCV_SEAM_FINDERS_HPP__

#include "precomp.hpp"

class SeamFinder
{
public:
    enum { NO, VORONOI, GC_COLOR, GC_COLOR_GRAD };
    static cv::Ptr<SeamFinder> createDefault(int type);

    virtual ~SeamFinder() {}
    virtual void find(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners,
                      std::vector<cv::Mat> &masks) = 0;
};


class NoSeamFinder : public SeamFinder
{
public:
    void find(const std::vector<cv::Mat>&, const std::vector<cv::Point>&, std::vector<cv::Mat>&) {}
};


class PairwiseSeamFinder : public SeamFinder
{
public:
    virtual void find(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners,
                      std::vector<cv::Mat> &masks);

protected:
    virtual void findInPair(size_t first, size_t second, cv::Rect roi) = 0;

    std::vector<cv::Mat> images_;
    std::vector<cv::Point> corners_;
    std::vector<cv::Mat> masks_;
};


class VoronoiSeamFinder : public PairwiseSeamFinder
{
private:
    void findInPair(size_t first, size_t second, cv::Rect roi);
};


class GraphCutSeamFinder : public SeamFinder
{
public:
    enum { COST_COLOR, COST_COLOR_GRAD };
    GraphCutSeamFinder(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                       float bad_region_penalty = 1000.f);

    void find(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners,
              std::vector<cv::Mat> &masks);

private:
    class Impl;
    cv::Ptr<Impl> impl_;
};

#endif // __OPENCV_SEAM_FINDERS_HPP__
