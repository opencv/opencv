// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "io.hpp"
namespace cv {
namespace ccm {
IO::IO(std::string illuminant_, std::string observer_)
    : illuminant(illuminant_)
    , observer(observer_) {};

bool IO::operator<(const IO& other) const
{
    return (illuminant < other.illuminant || ((illuminant == other.illuminant) && (observer < other.observer)));
}

bool IO::operator==(const IO& other) const
{
    return illuminant == other.illuminant && observer == other.observer;
};

IO IO::getIOs(IO_TYPE io)
{
    switch (io)
    {
    case cv::ccm::A_2:
    {
        IO A_2_IO("A", "2");
        return A_2_IO;
        break;
    }
    case cv::ccm::A_10:
    {
        IO A_1O_IO("A", "10");
        return A_1O_IO;
        break;
    }
    case cv::ccm::D50_2:
    {
        IO D50_2_IO("D50", "2");
        return D50_2_IO;
        break;
    }
    case cv::ccm::D50_10:
    {
        IO D50_10_IO("D50", "10");
        return D50_10_IO;
        break;
    }
    case cv::ccm::D55_2:
    {
        IO D55_2_IO("D55", "2");
        return D55_2_IO;
        break;
    }
    case cv::ccm::D55_10:
    {
        IO D55_10_IO("D55", "10");
        return D55_10_IO;
        break;
    }
    case cv::ccm::D65_2:
    {
        IO D65_2_IO("D65", "2");
        return D65_2_IO;
    }
    case cv::ccm::D65_10:
    {
        IO D65_10_IO("D65", "10");
        return D65_10_IO;
        break;
    }
    case cv::ccm::D75_2:
    {
        IO D75_2_IO("D75", "2");
        return D75_2_IO;
        break;
    }
    case cv::ccm::D75_10:
    {
        IO D75_10_IO("D75", "10");
        return D75_10_IO;
        break;
    }
    case cv::ccm::E_2:
    {
        IO E_2_IO("E", "2");
        return E_2_IO;
        break;
    }
    case cv::ccm::E_10:
    {
        IO E_10_IO("E", "10");
        return E_10_IO;
        break;
    }
    default:
        return IO();
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
