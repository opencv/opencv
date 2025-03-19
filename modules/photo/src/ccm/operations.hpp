// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_CCM_OPERATIONS_HPP__
#define __OPENCV_CCM_OPERATIONS_HPP__

#include "utils.hpp"

namespace cv {
namespace ccm {

/** @brief Operation class contains some operarions used for color space
           conversion containing linear transformation and non-linear transformation
   */
class Operation
{
public:
    typedef std::function<Mat(Mat)> MatFunc;
    bool linear;
    Mat M;
    MatFunc f;

    Operation()
        : linear(true)
        , M(Mat()) {};
    Operation(Mat M_)
        : linear(true)
        , M(M_) {};
    Operation(MatFunc f_)
        : linear(false)
        , f(f_) {};
    virtual ~Operation() {};

    /** @brief operator function will run operation
    */
    Mat operator()(Mat& abc);

    /** @brief add function will conbine this operation
               with other linear transformation operation
    */
    void add(const Operation& other);

    void clear();
    static Operation& getIdentityOp()
    {
        static Operation identity_op([](Mat x) { return x; });
        return identity_op;
    }
};

class Operations
{
public:
    std::vector<Operation> ops;
    Operations()
        : ops {} {};
    Operations(std::initializer_list<Operation> op)
        : ops { op } {};
    virtual ~Operations() {};

    /** @brief add function will conbine this operation with other transformation operations
    */
    Operations& add(const Operations& other);

    /** @brief run operations to make color conversion
    */
    Mat run(Mat abc);
    static const Operations& getIdentityOps()
    {
        static Operations Operation_op {Operation::getIdentityOp()};
        return Operation_op;
    }
};

}
}  // namespace cv::ccm

#endif