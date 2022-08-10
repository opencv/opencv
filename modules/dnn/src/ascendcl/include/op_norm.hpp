// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_OP_NORM_HPP
#define OPENCV_DNN_ASCENDCL_OP_NORM_HPP

#include "ascendcl.hpp"
#include "operator.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

class LRN : public Operator
{
public:
    LRN(int depth_radius = 5, float bias = 1.0, float alpha = 1.0, float beta = 0.75, String norm_region = "ACROSS_CHANNELS");
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    int depth_radius_;
    float bias_;
    float alpha_;
    float beta_;
    String norm_region_;
};

class BatchNorm : public Operator
{
public:
    BatchNorm(float epsilon = 1e-6);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    float epsilon_;
    String data_format_;
    bool is_training_;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_OP_POOLING_HPP
