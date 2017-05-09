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

#ifndef MAPPROJEC_H_
#define MAPPROJEC_H_

#include "map.hpp"


namespace cv {
namespace reg {


/*!
 * Defines an transformation that consists on a projective transformation
 */
class CV_EXPORTS MapProjec : public Map
{
public:
    /*!
     * Default constructor builds an identity map
     */
    MapProjec(void);

    /*!
     * Constructor providing explicit values
     * \param[in] projTr Projective transformation
     */
    MapProjec(const cv::Matx<double, 3, 3>& projTr);

    /*!
     * Destructor
     */
    ~MapProjec(void);

    void inverseWarp(const cv::Mat& img1, cv::Mat& img2) const;

    cv::Ptr<Map> inverseMap(void) const;

    void compose(const Map& map);

    void scale(double factor);

    /*!
     * Returns projection matrix
     * \return Projection matrix
     */
    const cv::Matx<double, 3, 3>& getProjTr() const {
        return projTr_;
    }

    /*!
     * Normalizes object's homography
     */
    void normalize(void) {
        double z = 1./projTr_(2, 2);
        for(size_t v_i = 0; v_i < sizeof(projTr_.val)/sizeof(projTr_.val[0]); ++v_i)
            projTr_.val[v_i] *= z;
    }

private:
    cv::Matx<double, 3, 3> projTr_;       /*< Projection matrix */
};


}}  // namespace cv::reg

#endif  // MAPPROJEC_H_
