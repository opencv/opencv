/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// Copyright (C) 2013, Alfonso Sanchez-Beato, all rights reserved.
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
// In no event shall the contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef MAP_H_
#define MAP_H_

#include <opencv2/core.hpp> // Basic OpenCV structures (cv::Mat, Scalar)

namespace cv {
namespace reg {


/*!
 * Defines a map T from one coordinate system to another
 */
class CV_EXPORTS Map
{
public:
    /*!
     * Virtual destructor
     */
    virtual ~Map(void);

    /*!
     * Warps image to a new coordinate frame. The calculation is img2(x)=img1(T^{-1}(x)), as we
     * have to apply the inverse transformation to the points to move them to were the values
     * of img2 are.
     * \param[in] img1 Original image
     * \param[out] img2 Warped image
     */
    virtual void warp(const cv::Mat& img1, cv::Mat& img2) const;

    /*!
     * Warps image to a new coordinate frame. The calculation is img2(x)=img1(T(x)), so in fact
     * this is the inverse warping as we are taking the value of img1 with the forward
     * transformation of the points.
     * \param[in] img1 Original image
     * \param[out] img2 Warped image
     */
    virtual void inverseWarp(const cv::Mat& img1, cv::Mat& img2) const = 0;

    /*!
     * Calculates the inverse map
     * \return Inverse map
     */
    virtual cv::Ptr<Map> inverseMap(void) const = 0;

    /*!
     * Changes the map composing the current transformation with the one provided in the call.
     * The order is first the current transformation, then the input argument.
     * \param[in] map Transformation to compose with.
     */
    virtual void compose(const Map& map) = 0;

    /*!
     * Scales the map by a given factor as if the coordinates system is expanded/compressed
     * by that factor.
     * \param[in] factor Expansion if bigger than one, compression if smaller than one
     */
    virtual void scale(double factor) = 0;
};


}}  // namespace cv::reg

#endif  // MAP_H_
