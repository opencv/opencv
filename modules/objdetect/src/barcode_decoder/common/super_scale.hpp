/// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef OPENCV_BARCODE_SUPER_SCALE_HPP
#define OPENCV_BARCODE_SUPER_SCALE_HPP

#ifdef HAVE_OPENCV_DNN
# include "opencv2/dnn.hpp"
#endif

namespace cv {
namespace barcode {

class SuperScale
{
public:
    SuperScale() = default;

    ~SuperScale() = default;

    int init(const std::string &proto_path, const std::string &model_path);

    void processImageScale(const Mat &src, Mat &dst, float scale, const bool &use_sr, int sr_max_size = 160);

#ifdef HAVE_OPENCV_DNN
private:
    dnn::Net srnet_;
    bool net_loaded_ = false;

    int superResolutionScale(const cv::Mat &src, cv::Mat &dst);
#endif
};

} // namespace barcode
} // namespace cv

#endif // OPENCV_BARCODE_SUPER_SCALE_HPP
