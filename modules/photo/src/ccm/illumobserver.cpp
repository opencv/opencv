// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "illumobserver.hpp"
namespace cv {
namespace ccm {
IllumObserver::IllumObserver(std::string illuminant_, std::string observer_)
    : illuminant(illuminant_)
    , observer(observer_) {};

bool IllumObserver::operator<(const IllumObserver& other) const
{
    return (illuminant < other.illuminant || ((illuminant == other.illuminant) && (observer < other.observer)));
}

bool IllumObserver::operator==(const IllumObserver& other) const
{
    return illuminant == other.illuminant && observer == other.observer;
};

IllumObserver IllumObserver::getIllumObservers(IllumObserver_TYPE illumobserver)
{
    switch (illumobserver)
    {
    case cv::ccm::A_2:
    {
        IllumObserver A_2_IllumObserver("A", "2");
        return A_2_IllumObserver;
        break;
    }
    case cv::ccm::A_10:
    {
        IllumObserver A_1O_IllumObserver("A", "10");
        return A_1O_IllumObserver;
        break;
    }
    case cv::ccm::D50_2:
    {
        IllumObserver D50_2_IllumObserver("D50", "2");
        return D50_2_IllumObserver;
        break;
    }
    case cv::ccm::D50_10:
    {
        IllumObserver D50_10_IllumObserver("D50", "10");
        return D50_10_IllumObserver;
        break;
    }
    case cv::ccm::D55_2:
    {
        IllumObserver D55_2_IllumObserver("D55", "2");
        return D55_2_IllumObserver;
        break;
    }
    case cv::ccm::D55_10:
    {
        IllumObserver D55_10_IllumObserver("D55", "10");
        return D55_10_IllumObserver;
        break;
    }
    case cv::ccm::D65_2:
    {
        IllumObserver D65_2_IllumObserver("D65", "2");
        return D65_2_IllumObserver;
    }
    case cv::ccm::D65_10:
    {
        IllumObserver D65_10_IllumObserver("D65", "10");
        return D65_10_IllumObserver;
        break;
    }
    case cv::ccm::D75_2:
    {
        IllumObserver D75_2_IllumObserver("D75", "2");
        return D75_2_IllumObserver;
        break;
    }
    case cv::ccm::D75_10:
    {
        IllumObserver D75_10_IllumObserver("D75", "10");
        return D75_10_IllumObserver;
        break;
    }
    case cv::ccm::E_2:
    {
        IllumObserver E_2_IllumObserver("E", "2");
        return E_2_IllumObserver;
        break;
    }
    case cv::ccm::E_10:
    {
        IllumObserver E_10_IllumObserver("E", "10");
        return E_10_IllumObserver;
        break;
    }
    default:
        return IllumObserver();
        break;
    }
}
// data from https://en.wikipedia.org/wiki/Standard_illuminant.
std::vector<double> xyY2XYZ(const std::vector<double>& xyY)
{
    double Y = xyY.size() >= 3 ? xyY[2] : 1;
    return { Y * xyY[0] / xyY[1], Y, Y / xyY[1] * (1 - xyY[0] - xyY[1]) };
}

}
}  // namespace cv::ccm
