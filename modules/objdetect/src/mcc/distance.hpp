// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_MCC_DISTANCE_HPP__
#define __OPENCV_MCC_DISTANCE_HPP__

#include "utils.hpp"
#include "opencv2/objdetect/ccm.hpp"

namespace cv {
namespace ccm {
/** possibale functions to calculate the distance between
    colors.see https://en.wikipedia.org/wiki/Color_difference for details;*/

/** @brief  distance between two points in formula CIE76
    @param lab1 a 3D vector
    @param lab2 a 3D vector
    @return distance between lab1 and lab2
*/

double deltaCIE76(const Vec3d& lab1, const Vec3d& lab2);

/** @brief  distance between two points in formula CIE94
    @param lab1 a 3D vector
    @param lab2 a 3D vector
    @param kH Hue scale
    @param kC Chroma scale
    @param kL Lightness scale
    @param k1 first scale parameter
    @param k2 second scale parameter
    @return distance between lab1 and lab2
*/

double deltaCIE94(const Vec3d& lab1, const Vec3d& lab2, const double& kH = 1.0,
        const double& kC = 1.0, const double& kL = 1.0, const double& k1 = 0.045,
        const double& k2 = 0.015);

double deltaCIE94GraphicArts(const Vec3d& lab1, const Vec3d& lab2);

double toRad(const double& degree);

double deltaCIE94Textiles(const Vec3d& lab1, const Vec3d& lab2);

/** @brief  distance between two points in formula CIE2000
    @param lab1 a 3D vector
    @param lab2 a 3D vector
    @param kL Lightness scale
    @param kC Chroma scale
    @param kH Hue scale
    @return distance between lab1 and lab2
*/
double deltaCIEDE2000_(const Vec3d& lab1, const Vec3d& lab2, const double& kL = 1.0,
        const double& kC = 1.0, const double& kH = 1.0);
double deltaCIEDE2000(const Vec3d& lab1, const Vec3d& lab2);

/** @brief  distance between two points in formula CMC
    @param lab1 a 3D vector
    @param lab2 a 3D vector
    @param kL Lightness scale
    @param kC Chroma scale
    @return distance between lab1 and lab2
*/

double deltaCMC(const Vec3d& lab1, const Vec3d& lab2, const double& kL = 1, const double& kC = 1);

double deltaCMC1To1(const Vec3d& lab1, const Vec3d& lab2);

double deltaCMC2To1(const Vec3d& lab1, const Vec3d& lab2);

Mat distance(Mat src,Mat ref, DISTANCE_TYPE distance_type);

}
}  // namespace cv::ccm

#endif