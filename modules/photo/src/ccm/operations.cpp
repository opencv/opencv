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

#include "operations.hpp"
#include "utils.hpp"
namespace cv {
namespace ccm {

Mat Operation::operator()(Mat& abc)
{
    if (!linear)
    {
        return f(abc);
    }
    if (M.empty())
    {
        return abc;
    }
    return multiple(abc, M);
};

void Operation::add(const Operation& other)
{
    if (M.empty())
    {
        M = other.M.clone();
    }
    else
    {
        M = M * other.M;
    }
};

void Operation::clear()
{
    M = Mat();
};

Operations& Operations::add(const Operations& other)
{
    ops.insert(ops.end(), other.ops.begin(), other.ops.end());
    return *this;
};

Mat Operations::run(Mat abc)
{
    Operation hd;
    for (auto& op : ops)
    {
        if (op.linear)
        {
            hd.add(op);
        }
        else
        {
            abc = hd(abc);
            hd.clear();
            abc = op(abc);
        }
    }
    abc = hd(abc);
    return abc;
};

}
}  // namespace cv::ccm