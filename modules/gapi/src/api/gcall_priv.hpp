// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GCALL_PRIV_HPP
#define OPENCV_GCALL_PRIV_HPP

#include <vector>
#include <unordered_map>

#include "opencv2/gapi/garg.hpp"
#include "opencv2/gapi/gcall.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "api/gnode.hpp"

namespace cv {

// GCall is used to capture details (arguments) passed to operation when the graph is
// constructed. It is, in fact, just a "serialization" of a function call (to some extent). The
// only place where new GCall objects are constructed is KernelName::on(). Note that GCall not
// only stores its input arguments, but also yields operation's pseudo-results to return
// "results".
// GCall arguments are GArgs which can wrap either our special types (like GMat) or other
// stuff user may pass according to operation's signature (opaque to us).
// If a dynamic g-object is wrapped in GArg, it has origin - something where that object comes
// from. It is either another function call (again, a GCall) or nothing (for graph's starting
// points, for example). By using these links, we understand what the flow is and construct the
// real graph. Origin is a node in a graph, represented by GNode.
// When a GCall is created, it instantiates it's appropriate GNode since we need an origin for
// objects we produce with this call. This is what is stored in m_node and then is used in every
// yield() call (the framework calls yield() according to template signature which we strip then
// - aka type erasure).
// Here comes the recursion - GNode knows it is created for GCall, and GCall stores that node
// object as origin for yield(). In order to break it, in GNode's object destructor this m_node
// pointer is reset (note - GCall::Priv remains alive). Now GCall's ownership "moves" to GNode
// and remains there until the API part is destroyed.
class GCall::Priv
{
public:
    std::vector<GArg> m_args;
    const GKernel     m_k;

    // TODO: Rename to "constructionNode" or smt to reflect its lifetime
    GNode             m_node;

    explicit Priv(const GKernel &k);
};

}

#endif // OPENCV_GCALL_PRIV_HPP
