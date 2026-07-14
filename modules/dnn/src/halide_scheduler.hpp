// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_HALIDE_SCHEDULER_HPP__
#define __OPENCV_DNN_HALIDE_SCHEDULER_HPP__

#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{

class HalideScheduler
{
public:
    HalideScheduler(const std::string& configFile);

    ~HalideScheduler();

    // Returns true if pipeline found in scheduling file.
    // If more than one function, returns true if the top function scheduled.
    // Other functions are optional to scheduling.
    bool process(Ptr<BackendNode>& node);

private:
    FileStorage fs;
};

}  // namespace dnn
}  // namespace cv

#endif  // __OPENCV_DNN_HALIDE_SCHEDULER_HPP__
