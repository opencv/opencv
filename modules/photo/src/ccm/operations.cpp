// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
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