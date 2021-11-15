// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_REMAP_TYPE_HPP
#define OPENCV_REMAP_TYPE_HPP

namespace cv {

enum class RemapType {
    fp32_mapxy,
    fp32_mapx_mapy,
    fixedPointQ16_5,
    fixedPointQ32_5,
    int16,
    int32
};

inline RemapType checkAndGetRemapType(Mat &map1, Mat &map2)
{
    CV_Assert(!map1.empty() || !map2.empty());
    if (map2.channels() == 2 && !map2.empty())
        std::swap(map1, map2);
    CV_Assert((map2.empty() && map1.channels() == 2) ||
              (!map2.empty() && map1.size() == map2.size() && map2.channels() == 1));

    if ((map1.depth() == CV_32F && !map1.empty()) || (map2.depth() == CV_32F && !map2.empty())) // fp32_mapxy or fp32_mapx_mapy
    {
        if (map1.type() == CV_32FC2 && map2.empty())
            return RemapType::fp32_mapxy;
        else if (map1.type() == CV_32FC1 && map2.type() == CV_32FC1)
            return RemapType::fp32_mapx_mapy;
    }
    else if (map1.channels() == 2) // int or fixedPointInt
    {
        if (map2.empty()) // int16 or int32
        {
            if (map1.type() == CV_16SC2) // int16
                return RemapType::int16;
            else if (map1.type() == CV_32SC2) // int32
                return RemapType::int32;
        }
        else if (map2.channels() == 1 && (map2.type() == CV_16UC1 || map2.type() == CV_16SC1)) // fixedPointQ16_5 or fixedPointQ32_5
        {
            if (map1.type() == CV_16SC2) // fixedPointQ16_5
                return RemapType::fixedPointQ16_5;
            else if (map1.type() == CV_32SC2) // fixedPointQ32_5
                return RemapType::fixedPointQ32_5;
        }
    }
    CV_Error(cv::Error::StsBadSize, format("remap doesn't support this type map1.type(): %d, map2.type(): %d \n"
                                           "fp32_mapxy: map1 having the type CV_32FC2; map2 is empty\n"
                                           "fp32_mapx_mapy: map1 having the type CV_32FC1; map2 having the type CV_32FC1\n"
                                           "fixedPointQ16_5: map1 having the type CV_16SC2; map2 having the type CV_16UC1 or CV_16SC1\n"
                                           "int16: map1 having the type CV_16SC2; map2 is empty\n"
                                           "If map2 isn't empty, map1.size() must be equal to map2.size().\n",
                                           map1.type(), map2.type()));
}

} // namespace cv

#endif
