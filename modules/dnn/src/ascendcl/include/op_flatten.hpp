// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_OP_FLATTEN_HPP
#define OPENCV_DNN_ASCENDCL_OP_FLATTEN_HPP

namespace cv { namespace dnn { namespace ascendcl {

#ifdef HAVE_ASCENDCL

class Flatten : public Operator
{
public:
    Flatten(int axis = 0);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    int axis_;
};

#endif // HAVE_ASCENDCL

}}} // namespace cv::dnn::ascendcl

#endif // OPENCV_DNN_ASCENDCL_OP_FLATTEN_HPP
