// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_PATTERN_MATCHING_HPP
#define OPENCV_GAPI_PATTERN_MATCHING_HPP

#include <stack>
#include <map>

#include "opencv2/gapi/gcomputation.hpp"
#include "opencv2/gapi/gcompiled.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "api/gcomputation_priv.hpp"
#include "api/gcall_priv.hpp"
#include "api/gnode_priv.hpp"

#include "compiler/gcompiled_priv.hpp"
#include "compiler/gmodel.hpp"

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>

namespace cv {
namespace gapi {

    GAPI_EXPORTS std::list<ade::NodeHandle> findMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph);

} //namespace gapi
} //namespace cv
#endif // OPENCV_GAPI_PATTERN_MATCHING_HPP
