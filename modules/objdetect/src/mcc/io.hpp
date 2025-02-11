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

#ifndef __OPENCV_MCC_IO_HPP__
#define __OPENCV_MCC_IO_HPP__

#include <opencv2/core.hpp>
#include <map>

namespace cv {
namespace ccm {

enum IO_TYPE
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

/** @brief Io is the meaning of illuminant and observer. See notes of ccm.hpp
           for supported list for illuminant and observer*/
class IO
{
public:
    std::string illuminant;
    std::string observer;
    IO() {};
    IO(std::string illuminant, std::string observer);
    virtual ~IO() {};
    bool operator<(const IO& other) const;
    bool operator==(const IO& other) const;
    static IO getIOs(IO_TYPE io);
};
std::vector<double> xyY2XYZ(const std::vector<double>& xyY);

}
}  // namespace cv::ccm

#endif