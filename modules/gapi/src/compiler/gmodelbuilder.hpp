// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GMODEL_BUILDER_HPP
#define OPENCV_GAPI_GMODEL_BUILDER_HPP

#include <map>
#include <unordered_map>

#include "opencv2/gapi/gproto.hpp"
#include "opencv2/gapi/gcall.hpp"

#include "api/gapi_priv.hpp"
#include "api/gnode.hpp"
#include "compiler/gmodel.hpp"

namespace cv { namespace gimpl {

struct Unrolled
{
    std::vector<cv::GNode> all_ops;
    GOriginSet             all_data;

    // NB.: Right now, as G-API operates with GMats only and that
    // GMats have no type or dimensions (when a computation is built),
    // track only origins (data links) with no any additional meta.
};

// FIXME: GAPI_EXPORTS only because of tests!!!
GAPI_EXPORTS Unrolled unrollExpr(const GProtoArgs &ins, const GProtoArgs &outs);

// This class generates an ADE graph with G-API specific metadata
// to represent user-specified computation in terms of graph model
//
// Resulting graph is built according to the following rules:
// - Every operation is a node
// - Every dynamic object (GMat) is a node
// - Edges between nodes represent producer/consumer relationships
//   between operations and data objects.
// FIXME: GAPI_EXPORTS only because of tests!!!
class GAPI_EXPORTS GModelBuilder
{
    GModel::Graph m_g;

    // Mappings of G-API user framework entities to ADE node handles
    std::unordered_map<const cv::GNode::Priv*, ade::NodeHandle> m_graph_ops;
    GOriginMap<ade::NodeHandle> m_graph_data;

    // Internal methods for mapping APIs into ADE during put()
    ade::NodeHandle put_OpNode(const cv::GNode &node);
    ade::NodeHandle put_DataNode(const cv::GOrigin &origin);

public:
    explicit GModelBuilder(ade::Graph &g);

    // TODO: replace GMat with a generic type
    // TODO: Cover with tests! (as the rest of internal stuff)
    // FIXME: Calling this method multiple times is currently UB
    // TODO: add a semantic link between "ints" returned and in-model data IDs.
    typedef std::tuple<std::vector<RcDesc>,
                       std::vector<RcDesc>,
                       std::vector<ade::NodeHandle>,
                       std::vector<ade::NodeHandle> > ProtoSlots;

    ProtoSlots put(const GProtoArgs &ins, const GProtoArgs &outs);

protected:
    ade::NodeHandle opNode(cv::GMat gmat);
};

}}

#endif // OPENCV_GAPI_GMODEL_BUILDER_HPP
