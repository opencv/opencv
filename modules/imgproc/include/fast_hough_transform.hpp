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
// Copyright (C) 2015, Smart Engines Ltd, all rights reserved.
// Copyright (C) 2015, Institute for Information Transmission Problems of the Russian Academy of Sciences (Kharkevich Institute), all rights reserved.
// Copyright (C) 2015, Dmitry Nikolaev, Simon Karpenko, Michail Aliev, Elena Kuznetsova, all rights reserved.
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

#ifndef __OPENCV_FAST_HOUGH_TRANSFORM_HPP__
#define __OPENCV_FAST_HOUGH_TRANSFORM_HPP__
#ifdef __cplusplus

#include "opencv2/core.hpp"


namespace cv
{
/**
* @brief   Specifies the part of Hough space to calculate
* @details The enum specifies the part of Hough space to calculate. Each
* member specifies primarily direction of lines (horizontal or vertical)
* and the direction of angle changes.
* Direction of angle changes is from multiples of 90 to odd multiples of 45.
* The image considered to be written top-down and left-to-right.
* Angles are started from vertical line and go clockwise.
* Separate quarters and halves are written in orientation they should be in
* full Hough space.
*/
enum AngleRangeOption
{
  ARO_0_45    = 0, //< Vertical primarily direction and clockwise angle changes
  ARO_45_90   = 1, //< Horizontal primarily direction and counterclockwise angle changes
  ARO_90_135  = 2, //< Horizontal primarily direction and clockwise angle changes
  ARO_315_0   = 3, //< Vertical primarily direction and counterclockwise angle changes
  ARO_315_45  = 4, //< Vertical primarily direction
  ARO_45_135  = 5, //< Horizontal primarily direction
  ARO_315_135 = 6, //< Full set of directions
  ARO_CTR_HOR = 7, //< 90 +/- atan(0.5), interval approximately from 64.5 to 116.5 degrees.
                   //< It is used for calculating Fast Hough Transform for images skewed by atan(0.5).
  ARO_CTR_VER = 8  //< +/- atan(0.5), interval approximately from 333.5(-26.5) to 26.5 degrees
                   //< It is used for calculating Fast Hough Transform for images skewed by atan(0.5).
};

/**
 * @brief   Specifies binary operations.
 * @details The enum specifies binary operations, that is such ones which involve
 *          two operands. Formally, a binary operation @f$ f @f$ on a set @f$ S @f$
 *          is a binary relation that maps elements of the Cartesian product
 *          @f$ S \times S @f$ to @f$ S @f$:
*           @f[ f: S \times S \to S @f]
 * @ingroup MinUtils_MathOper
 */
enum HoughOp
{
  FHT_MIN = 0,  //< Binary minimum operation. The constant specifies the binary minimum operation
                //< @f$ f @f$ that is defined as follows: @f[ f(x, y) = \min(x, y) @f]
  FHT_MAX = 1,  //< Binary maximum operation. The constant specifies the binary maximum operation
                //< @f$ f @f$ that is defined as follows: @f[ f(x, y) = \max(x, y) @f]
  FHT_ADD = 2,  //< Binary addition operation. The constant specifies the binary addition operation
                //< @f$ f @f$ that is defined as follows: @f[ f(x, y) = x + y @f]
  FHT_AVE = 3   //< Binary average operation. The constant specifies the binary average operation
                //< @f$ f @f$ that is defined as follows: @f[ f(x, y) = \frac{x + y}{2} @f]
};

/**
* @brief   Specifies to do or not to do skewing of Hough transform image
* @details The enum specifies to do or not to do skewing of Hough transform image
* so it would be no cycling in Hough transform image through borders of image.
*/
enum HoughDeskewOption
{
  HDO_RAW    = 0, //< Use raw cyclic image
  HDO_DESKEW = 1  //< Prepare deskewed image
};

/**
 * @brief   Specifies the degree of rules validation.
 * @details The enum specifies the degree of rules validation. This can be used,
 *          for example, to choose a proper way of input arguments validation.
 */
typedef enum {
  RO_STRICT          = 0x00,  ///< Validate each rule in a proper way.
  RO_IGNORE_BORDERS  = 0x01,  ///< Skip validations of image borders.
} RulesOption;

/**
* @brief   Calculates 2D Fast Hough transform of an image.
* @param   dst         The destination image, result of transformation.
* @param   src         The source (input) image.
* @param   dstMatDepth The depth of destination image
* @param   operation   The operation to be applied, see cv::HoughOp
* @param   angleRange  The part of Hough space to calculate, see cv::AngleRangeOption
* @param   makeSkew    Specifies to do or not to do image skewing, see cv::HoughDeskewOption
*
* The function calculates the fast Hough transform for full, half or quarter
* range of angles.
*/
CV_EXPORTS_W void FastHoughTransform( InputArray src,
                                      OutputArray dst,
                                      int         dstMatDepth,
                                      int         angleRange = ARO_315_135,
                                      int         op = FHT_ADD,
                                      int         makeSkew = HDO_DESKEW );

/**
* @brief   Calculates coordinates of line segment corresponded by point in Hough space.
* @param   line       Coordinates of line segment corresponded by point in Hough space.
* @param   houghPoint  Point in Hough space.
* @param   srcImgInfo The source (input) image of Hough transform.
* @param   angleRange  The part of Hough space where point is situated, see cv::AngleRangeOption
* @param   makeSkew    Specifies to do or not to do image skewing, see cv::HoughDeskewOption
* @param   rules       Specifies strictness of line segment calculating, see cv::RulesOption
* @remarks If rules parameter set to RO_STRICT
           then returned line cut along the border of source image.
* @remarks If rules parameter set to RO_WEAK then in case of point, which belongs
           the incorrect part of Hough image, returned line will not intersect source image.
*
* The function calculates coordinates of line segment corresponded by point in Hough space.
*/
CV_EXPORTS_W void HoughPoint2Line( OutputArray  line,
                                   const Point &houghPoint,
                                   const Mat   &srcImgInfo,
                                   int          angleRange = ARO_315_135,
                                   int          makeSkew = HDO_DESKEW,
                                   int          rules = RO_IGNORE_BORDERS );

} // namespace cv

#endif //__cplusplus
#endif //__OPENCV_FAST_HOUGH_TRANSFORM_HPP__
