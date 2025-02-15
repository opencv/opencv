// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
// Modified by darkliang wangberlinT

#include "precomp.hpp"

namespace cv {

constexpr static float MAX_SCALE = 4.0f;

class SuperScaleImpl : public SuperScale
{
private:
    dnn::Net srnet_;

public:
    SuperScaleImpl(const std::string &proto_path, const std::string &model_path)
    {
        srnet_ = dnn::readNetFromCaffe(proto_path, model_path);
    }
    ~SuperScaleImpl() {}
    void processImageScale(const Mat &src, Mat &dst, float scale, int sr_max_size = 160) CV_OVERRIDE
    {
        scale = min(scale, MAX_SCALE);
        if (scale > .0 && scale < 1.0)
        {  // down sample
            resize(src, dst, Size(), scale, scale, INTER_AREA);
        }
        else if (scale > 1.5 && scale < 2.0)
        {
            resize(src, dst, Size(), scale, scale, INTER_CUBIC);
        }
        else if (scale >= 2.0)
        {
            int width = src.cols;
            int height = src.rows;
            if ((int) sqrt(width * height * 1.0) < sr_max_size && !srnet_.empty())
            {
                superResolutionScale(src, dst);
                if (scale > 2.0)
                {
                    processImageScale(dst, dst, scale / 2.0f);
                }
            }
            else
            {
                resize(src, dst, Size(), scale, scale, INTER_CUBIC);
            }
        }
    }
    bool isOpened() const CV_OVERRIDE
    {
        return !srnet_.empty();
    }

protected:
    int superResolutionScale(const Mat &src, Mat &dst)
    {
        Mat blob;
        dnn::blobFromImage(src, blob, 1.0 / 255, Size(src.cols, src.rows), {0.0f}, false, false);

        srnet_.setInput(blob);
        auto prob = srnet_.forward();

        dst = Mat(prob.size[2], prob.size[3], CV_8UC1);

        for (int row = 0; row < prob.size[2]; row++)
        {
            const float *prob_score = prob.ptr<float>(0, 0, row);
            auto *dst_row = dst.ptr<uchar>(row);
            for (int col = 0; col < prob.size[3]; col++)
            {
                dst_row[col] = saturate_cast<uchar>(prob_score[col] * 255.0f);
            }
        }
        return 0;
    }

};

SuperScale::SuperScale() {}

SuperScale::~SuperScale() {}

cv::Ptr<SuperScale> SuperScale::create(const std::string &proto_path, const std::string &model_path)
{
    return makePtr<SuperScaleImpl>(proto_path, model_path);
}


}  // namespace cv
