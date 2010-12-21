/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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

#ifndef BLOBDETECTOR_HPP_
#define BLOBDETECTOR_HPP_

#include "precomp.hpp"

struct BlobDetectorParameters
{
  BlobDetectorParameters();

  float thresholdStep;
  float minThreshold;
  float maxThreshold;
  float maxCentersDist;
  int defaultKeypointSize;
  size_t minRepeatability;
  bool computeRadius;
  bool isGrayscaleCentroid;
  int centroidROIMargin;

  bool filterByArea, filterByInertia, filterByCircularity, filterByColor, filterByConvexity;
  float minArea;
  float maxArea;
  float minCircularity;
  float minInertiaRatio;
  float minConvexity;
};

class BlobDetector //: public cv::FeatureDetector
{
public:
  BlobDetector(const BlobDetectorParameters &parameters = BlobDetectorParameters());
  void detect(const cv::Mat& image, std::vector<cv::Point2f>& keypoints, const cv::Mat& mask = cv::Mat()) const;
protected:
  struct Center
  {
    cv::Point2d location;
    double radius;
    double confidence;
  };

  virtual void detectImpl(const cv::Mat& image, std::vector<cv::Point2f>& keypoints, const cv::Mat& mask = cv::Mat()) const;
  virtual void findBlobs(const cv::Mat &image, const cv::Mat &binaryImage, std::vector<Center> &centers) const;

  cv::Point2d computeGrayscaleCentroid(const cv::Mat &image, const std::vector<cv::Point> &contour) const;

  BlobDetectorParameters params;
};

#endif /* BLOBDETECTOR_HPP_ */
