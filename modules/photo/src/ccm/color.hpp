// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_CCM_COLOR_HPP__
#define __OPENCV_CCM_COLOR_HPP__

#include "distance.hpp"
#include "colorspace.hpp"
#include "opencv2/photo.hpp"

namespace cv {
namespace ccm {

/** @brief Color defined by color_values and color space
*/

class Color
{
public:
    /** @param grays mask of grayscale color
        @param colored mask of colored color
        @param history storage of historical conversion
    */
    Mat colors;
    std::shared_ptr<ColorSpaceBase> cs;
    Mat grays;
    Mat colored;
    std::map<ColorSpaceBase, std::shared_ptr<Color>> history;

    Color();
    Color(Mat colors_, enum COLOR_SPACE cs_);
    Color(Mat colors_, enum COLOR_SPACE cs_, Mat colored);
    Color(Mat colors_, const ColorSpaceBase& cs, Mat colored);
    Color(Mat colors_, const ColorSpaceBase& cs);
    Color(Mat colors_, std::shared_ptr<ColorSpaceBase> cs_);
    virtual ~Color() {};

    /** @brief Change to other color space.
                 The conversion process incorporates linear transformations to speed up.
        @param other type of ColorSpaceBase.
        @param  method the chromatic adapation method.
        @param save when save if True, get data from history first.
        @return Color.
    */
    Color to(const ColorSpaceBase& other, CAM method = BRADFORD, bool save = true);

    /** @brief Convert color to another color space using COLOR_SPACE enum.
        @param other type of COLOR_SPACE.
        @param method chromatic adaption method.
        @param save whether to save the result.
        @return Color.
    */
    Color to(COLOR_SPACE other, CAM method = BRADFORD, bool save = true);

    /** @brief Channels split.
       @return each channel.
    */
    Mat channel(Mat m, int i);

    /** @brief To Gray.
    */
    Mat toGray(IO io, CAM method = BRADFORD, bool save = true);

    /** @brief To Luminant.
    */
    Mat toLuminant(IO io, CAM method = BRADFORD, bool save = true);

    /** @brief Diff without IO.
        @param other type of Color.
        @param method type of distance.
        @return distance between self and other
    */
    Mat diff(Color& other, DistanceType method = DISTANCE_CIE2000);

    /** @brief Diff with IO.
        @param other type of Color.
        @param io type of IO.
        @param method type of distance.
        @return distance between self and other
    */
    Mat diff(Color& other, IO io, DistanceType method = DISTANCE_CIE2000);

    /** @brief Calculate gray mask.
    */
    void getGray(double JDN = 2.0);

    /** @brief Operator for mask copy.
    */
    Color operator[](Mat mask);
};

class GetColor
{
public:
    static std::shared_ptr<Color> getColor(ColorCheckerType const_color);
    static Mat getColorChecker(const double* checker, int row);
    static Mat getColorCheckerMask(const uchar* checker, int row);
};

}
}  // namespace cv::ccm

#endif