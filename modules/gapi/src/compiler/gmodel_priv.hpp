// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GMODEL_PRIV_HPP
#define OPENCV_GAPI_GMODEL_PRIV_HPP

#include <ade/graph.hpp>
#include "compiler/gmodel.hpp"
#include "api/gproto_priv.hpp" // origin_of

namespace cv { namespace gimpl {

// The mapping between user-side GMat/GScalar/... objects
// and its  appropriate nodes. Can be stored in graph optionally
// (NOT used by any compiler or backends, introspection purposes
// only)
struct Layout
{
    static const char *name() { return "Layout"; }
    GOriginMap<ade::NodeHandle> object_nodes;
};

namespace GModel {

using LayoutGraph = ade::TypedGraph
    < Layout
    >;

using ConstLayoutGraph = ade::ConstTypedGraph
    < Layout
    >;

    ade::NodeHandle mkDataNode(Graph &g, const GOrigin& origin);

namespace detail
{
    // FIXME: GAPI_EXPORTS only because of tests!!!
    GAPI_EXPORTS ade::NodeHandle dataNodeOf(const ConstLayoutGraph& g, const GOrigin &origin);
}

template<typename T> inline ade::NodeHandle dataNodeOf(const ConstLayoutGraph& g, T &&t)
{
    return detail::dataNodeOf(g, cv::gimpl::proto::origin_of(GProtoArg{t}));
}

inline ade::NodeHandle producerOf(const cv::gimpl::GModel::Graph& gm, ade::NodeHandle dh)
{
    GAPI_Assert(gm.metadata(dh).get<NodeType>().t == NodeType::DATA);
    auto ins = dh->inNodes();
    return ins.empty() ? ade::NodeHandle{ } : *ins.begin();
}


}}}

#endif // OPENCV_GAPI_GMODEL_PRIV_HPP
