// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_CCM_IllumObserver_HPP__
#define __OPENCV_CCM_IllumObserver_HPP__

#include <opencv2/core.hpp>
#include <map>

namespace cv {
namespace ccm {

enum IllumObserver_TYPE
{
    A_2,
    A_10,
    D50_2,
    D50_10,
    D55_2,
    D55_10,
    D65_2,
    D65_10,
    D75_2,
    D75_10,
    E_2,
    E_10
};

/** @brief IllumObserver is the meaning of illuminant and observer. See notes of ccm.hpp
           for supported list for illuminant and observer*/
class IllumObserver
{
public:
    std::string illuminant;
    std::string observer;
    IllumObserver() {};
    IllumObserver(std::string illuminant, std::string observer);
    virtual ~IllumObserver() {};
    bool operator<(const IllumObserver& other) const;
    bool operator==(const IllumObserver& other) const;
    static IllumObserver getIllumObservers(IllumObserver_TYPE illumobserver);
};
std::vector<double> xyY2XYZ(const std::vector<double>& xyY);

}
}  // namespace cv::ccm

#endif