// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/infer/parsers.hpp>

#include <tuple>
#include <numeric>

namespace cv { namespace gapi {

nn::parsers::GDetections parseSSD(const GMat& in,
                                  const GOpaque<Size>& inSz,
                                  const float confidenceThreshold,
                                  const int filterLabel)
{
    return nn::parsers::GParseSSDBL::on(in, inSz, confidenceThreshold, filterLabel);
}

nn::parsers::GRects parseSSD(const GMat& in,
                             const GOpaque<Size>& inSz,
                             const float confidenceThreshold,
                             const bool alignmentToSquare,
                             const bool filterOutOfBounds)
{
    return nn::parsers::GParseSSD::on(in, inSz, confidenceThreshold, alignmentToSquare, filterOutOfBounds);
}

nn::parsers::GDetections parseYolo(const GMat& in,
                                   const GOpaque<Size>& inSz,
                                   const float confidenceThreshold,
                                   const float nmsThreshold,
                                   const std::vector<float>& anchors)
{
    return nn::parsers::GParseYolo::on(in, inSz, confidenceThreshold, nmsThreshold, anchors);
}

} //namespace gapi
} //namespace cv
