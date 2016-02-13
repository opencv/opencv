/****************************************************************************************
 * By downloading, copying, installing or using the software you agree to this license. *
 * If you do not agree to this license, do not download, install,                       *
 * copy or use the software.                                                            *
 *                                                                                      *
 *                                                                                      *
 *                         License Agreement                                            *
 *              For Open Source Computer Vision Library                                 *
 *                     (3-clause BSD License)                                           *
 *                                                                                      *
 * Copyright (C) 2000-2016, Intel Corporation, all rights reserved.                     *
 * Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.                    *
 * Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.                    *
 * Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.          *
 * Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.                     *
 * Copyright (C) 2015-2016, Itseez Inc., all rights reserved.                           *
 * Third party copyrights are property of their respective owners.                      *
 *                                                                                      *
 * Redistribution and use in source and binary forms, with or without modification,     *
 * are permitted provided that the following conditions are met:                        *
 *                                                                                      *
 * Redistributions of source code must retain the above copyright notice,               *
 * this list of conditions and the following disclaimer.                                *
 *                                                                                      *
 * Redistributions in binary form must reproduce the above copyright notice,            *
 * this list of conditions and the following disclaimer in the documentation            *
 * and/or other materials provided with the distribution.                               *
 *                                                                                      *
 * Neither the names of the copyright holders nor the names of the contributors         *
 * may be used to endorse or promote products derived from this software                *
 * without specific prior written permission.                                           *
 *                                                                                      *
 * This software is provided by the copyright holders and contributors "as is" and      *
 * any express or implied warranties, including, but not limited to, the implied        *
 * warranties of merchantability and fitness for a particular purpose are disclaimed.   *
 * In no event shall copyright holders or contributors be liable for any direct,        *
 * indirect, incidental, special, exemplary, or consequential damages                   *
 * (including, but not limited to, procurement of substitute goods or services;         *
 * loss of use, data, or profits; or business interruption) however caused              *
 * and on any theory of liability, whether in contract, strict liability,               *
 * or tort (including negligence or otherwise) arising in any way out of                *
 * the use of this software, even if advised of the possibility of such damage.         *
 ****************************************************************************************/

 /**
  * @author {aravind | arvindsuresh2009@gmail.com}
  * Created on 2016-02-13 00:27
  */

// Necessary headers

#include <precomp.hpp>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#ifdef __cplusplus
namespace cv {
#endif

/**
 * [linspace - Constructs linearly spaced vector of length n between start and end values ( both inclusive )]
 * @param  start   [start value]
 * @param  end     [end value]
 * @param  n       [length of the vector]
 * @return vec     [linearly spaced vector]
 */
static inline std::vector<float> linspace(float start, float end, int n) {
  /**
   * A utility function for generating a vector of linearly spaced integers.
   */

  // Finding diff
  float diff = (end - start)*1.0/(n-1);
  std::vector<float> vec(n);
  float val = start;

  for(int i=0; i<n; ++i) {
    vec[i] = val;
    // Adding diff to val
    val += diff;
  }

  // Returning the vector
  return vec;
}

/**
 * [meshgrid Returns a rectangular grid 2D space]
 * @param matX [Repeating vector along X direction]
 * @param matY [Repeating vector along Y direction]
 * @param X    [Final meshgrid matrix X]
 * @param Y    [Final meshgrid matrix Y]
 */
static inline void meshgrid(const Mat &matX, const Mat &matY, Mat &X, Mat &Y) {
  /**
   * A utitlity function to compute the meshgrid matrices for given input vectors.
   */
  repeat(matX.reshape(1,1), matY.total(), 1, X);
  repeat(matY.reshape(1,1).t(), 1, matX.total(), Y);
}

/**
 * [radonTransform Computes the Radon Transform of the given input array]
 * @param src      [Input image]
 * @param dst      [Output sinogram, referenced]
 * @param accuracy [Minimum accuracy for angles ( IN DEGREES ), defaults to 1 degree]
 */
void radonTransform(InputArray src, OutputArray dst, float accuracy) {
  /**
   * Implementation of the Radon Transform
   *
   * Example:
   * 					::
   * 					cv::Mat img, imgRadTrans;
   * 					:::: img holds some image ::::
   *
   * 					radonTransform(img, imgRadTrans, 0.5);
   * 					:::: imgRadTrans contains the radon transform ::::
   */
  // Asserting accuracy to be positive
  CV_Assert(accuracy > 0);
  CV_Assert(src.type() == CV_32F);

  Mat srcMat = src.getMat();
  int rows = srcMat.rows, cols = srcMat.cols;
  float diag = std::sqrt(rows*rows + cols*cols);

  Mat srcMatPadded;
  int padWidth = (std::ceil(diag - cols) + 2)/2;
  int padHeight = (std::ceil(diag - rows) + 2)/2;
  copyMakeBorder(srcMat, srcMatPadded, padWidth, padHeight, padWidth, padHeight, BORDER_CONSTANT, 0);

  int n = 180/accuracy + 1;
  std::vector<float> vecTheta = linspace(0, 180, n);
  int dstCols = vecTheta.size(), dstRows = srcMatPadded.rows;

  dst.create(Size(dstCols, dstRows), CV_32F);
  Mat dstMat = dst.getMat();

  int num = srcMatPadded.cols;
  std::vector<float> vecVal = linspace(-1, 1, num);

  Mat meshX, meshY, meshXX, meshYY;
  meshgrid(Mat(vecVal), Mat(vecVal), meshX, meshY);

  for(int i = 0; i < dstCols; ++i) {
    float theta = (90 - vecTheta[i])*CV_PI/180.0;
    float cosTh = std::cos(theta), sinTh = std::sin(theta);

    Mat temp(srcMatPadded.size(), CV_32F);

    // Rotating meshgrid entries by an angle theta
    meshXX = cosTh*meshX - sinTh*meshY;
    meshYY = sinTh*meshX + cosTh*meshY;

    // Clamping values to [-1, 1]
    for(int l = 0; l < meshXX.cols; ++l) {
      for(int m = 0; m < meshXX.rows; ++m) {
        if(meshXX.at<float>(m, l) < -1) meshXX.at<float>(m, l) = -1;
        else if(meshXX.at<float>(m, l) > 1) meshXX.at<float>(m, l) = 1;

        if(meshYY.at<float>(m, l) < -1) meshYY.at<float>(m, l) = -1;
        else if(meshYY.at<float>(m, l) > 1) meshYY.at<float>(m, l) = 1;
      }
    }

    // Normalizing meshXX and meshYY to proper indices in the image
    normalize(meshXX, meshXX, 0, srcMatPadded.cols - 1, NORM_MINMAX, -1);
    normalize(meshYY, meshYY, 0, srcMatPadded.cols - 1, NORM_MINMAX, -1);

    // Remapping, so as to interpolate for non-integer points
    remap(srcMatPadded, temp, meshXX, meshYY, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    // Computing columnwise sum and hence the a column of the sinogram
    Mat dstColReduced;
    reduce(temp, dstColReduced, 1, CV_REDUCE_SUM, -1);
    dstColReduced.copyTo(dstMat.col(i));
  }
}

#ifdef __cplusplus
}
#endif
