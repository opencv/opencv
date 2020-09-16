// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2012 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/nnparsers.hpp>

#include <tuple>
#include <numeric>

namespace cv { namespace gapi {

nn::parsers::GDetections parseSSD(const GMat& in,
                                  const GOpaque<Size>& in_sz,
                                  const float confidence_threshold,
                                  const int filter_label)
{
    return nn::parsers::GParseSSDBL::on(in, in_sz, confidence_threshold, filter_label);
}

nn::parsers::GRects parseSSD(const GMat& in,
                             const GOpaque<Size>& in_sz,
                             const float confidence_threshold,
                             const bool alignment_to_square,
                             const bool filter_out_of_bounds)
{
    return nn::parsers::GParseSSD::on(in, in_sz, confidence_threshold, alignment_to_square, filter_out_of_bounds);
}

nn::parsers::GDetections parseYolo(const GMat& in,
                                   const GOpaque<Size>& in_sz,
                                   const float confidence_threshold,
                                   const float nms_threshold,
                                   const std::vector<float>& anchors)
{
    return nn::parsers::GParseYolo::on(in, in_sz, confidence_threshold, nms_threshold, anchors);
}

} //namespace gapi
} //namespace cv
