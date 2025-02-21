/// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef OPENCV_DNN_ALGORITHM_SUPER_SCALE_HPP
#define OPENCV_DNN_ALGORITHM_SUPER_SCALE_HPP

#include "opencv2/dnn.hpp"
#include <string>

namespace cv {

class CV_EXPORTS_W SuperScale
{
protected:
    SuperScale();
public:
    virtual ~SuperScale();
    CV_WRAP virtual void processImageScale(const Mat &src, Mat &dst, float scale, int sr_max_size = 160) = 0;
    CV_WRAP virtual bool isOpened() const = 0;
    static CV_WRAP Ptr<SuperScale> create(const std::string &proto_path, const std::string &model_path);
};

} // namespace cv

#endif
