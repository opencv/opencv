// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_COMMON_HPP
#define OPENCV_DNN_VKCOM_COMMON_HPP

#include <math.h>
#include <string.h>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

#include "opencv2/core/utils/logger.hpp"
#include "../vulkan/vk_functions.hpp"
#include "../include/vkcom.hpp"
#include "../shader/spv_shader.hpp"
//#include "../vulkan/vk_functions.hpp"
//#include "../vulkan/vk_loader.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN
extern VkQueue kQueue;
extern VkDevice kDevice;
extern cv::Mutex kContextMtx;
extern Ptr<CommandPool> cmdPoolPtr;
extern Ptr<PipelineFactory> pipelineFactoryPtr;
extern VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;



enum ShapeIdx
{
    kShapeIdxBatch = 0,
    kShapeIdxChannel,
    kShapeIdxHeight,
    kShapeIdxWidth,
};

#define VK_CHECK_RESULT(f) \
{ \
        if (f != VK_SUCCESS) \
        { \
            CV_LOG_ERROR(NULL, "Vulkan check failed, result = " << (int)f); \
            CV_Error(Error::StsError, "Vulkan check failed"); \
        } \
}

#define VKCOM_CHECK_BOOL_RET_VAL(val, ret) \
{ \
    bool res = (val); \
    if (!res) \
    { \
        CV_LOG_WARNING(NULL, "Check bool failed"); \
        return ret; \
    } \
}

#define VKCOM_CHECK_POINTER_RET_VOID(p) \
{ \
    if (NULL == (p)) \
    { \
        CV_LOG_WARNING(NULL, "Check pointer failed"); \
        return; \
    } \
}

#define VKCOM_CHECK_POINTER_RET_VAL(p, val) \
{ \
    if (NULL == (p)) \
    { \
        CV_LOG_WARNING(NULL, "Check pointer failed"); \
        return (val); \
    } \
}

bool checkFormat(Format fmt);
size_t elementSize(Format fmt);
int shapeCount(const Shape& shape, int start = -1, int end = -1);
#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_COMMON_HPP
