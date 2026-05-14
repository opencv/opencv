// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_TOKENIZER_IMPL_HPP
#define OPENCV_DNN_TOKENIZER_IMPL_HPP

#include <opencv2/dnn/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct Tokenizer::Impl {
    virtual ~Impl() {}
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& tokens) = 0;
};

CV__DNN_INLINE_NS_END
}}

#endif // OPENCV_DNN_TOKENIZER_IMPL_HPP
