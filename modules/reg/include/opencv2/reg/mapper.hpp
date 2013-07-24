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

#ifndef MAPPER_H_
#define MAPPER_H_

#include <opencv2/core.hpp> // Basic OpenCV structures (cv::Mat, Scalar)
#include "map.hpp"

namespace cv {
namespace reg {

/*
 * Encapsulates ways of calculating mappings between two images
 */
class CV_EXPORTS Mapper
{
public:
    virtual ~Mapper(void) {}

    /*
     * Calculate mapping between two images
     * \param[in] img1 Reference image
     * \param[in] img2 Warped image
     * \param[in,out] res Map from img1 to img2, stored in a smart pointer. If present as input,
     *       it is an initial rough estimation that the mapper will try to refine.
     */
    virtual void calculate(const cv::Mat& img1, const cv::Mat& img2, cv::Ptr<Map>& res) const = 0;

    /*
     * Returns a map compatible with the Mapper class
     * \return Pointer to identity Map
     */
    virtual cv::Ptr<Map> getMap(void) const = 0;

protected:
    /*
     * Calculates gradient and difference between images
     * \param[in] img1 Image one
     * \param[in] img2 Image two
     * \param[out] Ix Gradient x-coordinate
     * \param[out] Iy Gradient y-coordinate
     * \param[out] It Difference of images
     */
    void gradient(const cv::Mat& img1, const cv::Mat& img2,
                  cv::Mat& Ix, cv::Mat& Iy, cv::Mat& It) const;

    /*
     * Fills matrices with pixel coordinates of an image
     * \param[in] img Image
     * \param[out] grid_r Row (y-coordinate)
     * \param[out] grid_c Column (x-coordinate)
     */
    void grid(const Mat& img, Mat& grid_r, Mat& grid_c) const;

    /*
     * Per-element square of a matrix
     * \param[in] mat1 Input matrix
     * \return mat1[i,j]^2
     */
    cv::Mat sqr(const cv::Mat& mat1) const
    {
        cv::Mat res;
        res.create(mat1.size(), mat1.type());
        res = mat1.mul(mat1);
        return res;
    }
};


}}  // namespace cv::reg

#endif  // MAPPER_H_

