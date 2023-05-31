// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_COMPILER_PASSES_HELPERS_HPP
#define OPENCV_GAPI_COMPILER_PASSES_HELPERS_HPP

// FIXME: DROP THIS and REUSE ADE utilities
// (which serve as passes already but are not exposed as standalone functions)

#include <vector>

#include <ade/passes/pass_base.hpp>
#include <ade/node.hpp> // FIXME: Forward declarations instead?
#include <ade/graph.hpp>

namespace cv {
namespace gimpl {
namespace pass_helpers {

bool hasCycles(const ade::Graph &graph);
std::vector<ade::NodeHandle> topoSort(const ade::Graph &graph);

} // namespace pass_helpers
} // namespace gimpl
} // name

#endif // OPENCV_GAPI_COMPILER_PASSES_HELPERS_HPP
