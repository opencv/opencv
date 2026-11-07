// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_BASE_HPP
#define OPENCV_DNN_VKCOM_OP_BASE_HPP

#include "../../precomp.hpp"
#include "vkcom.hpp"
#include "context.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

// Forward declare
class Context;

class OpBase
{
public:
    OpBase();
    virtual ~OpBase();
    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) = 0;
protected:
    std::vector<VkDescriptorType> destTypes; // Save the input data type.
    std::string shader_name; // the key which is used for retrieve Pipeline from PipelineFactory.
    std::string type_;
    int group_x_;
    int group_y_;
    int group_z_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_BASE_HPP
