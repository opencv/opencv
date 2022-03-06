// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include <algorithm>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void softmax(InputArray inblob, OutputArray outblob)
{
    CV_Assert(inblob.rows() == 1);
    CV_Assert(inblob.type() == CV_32FC1);

    const Mat input = inblob.getMat();
    outblob.create(inblob.size(), inblob.type());

    Mat exp;
    const float max = *std::max_element(input.begin<float>(), input.end<float>());
    cv::exp((input - max), exp);
    outblob.getMat() = exp / cv::sum(exp)[0];
}

CV__DNN_INLINE_NS_END
}// dnn
}// cv
