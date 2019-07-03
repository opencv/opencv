// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <iostream>                              // cout
#include <sstream>                               // stringstream
#include <fstream>                               // ofstream
#include <map>

#include <ade/passes/check_cycles.hpp>
#include <ade/util/zip_range.hpp>                // indexed()

#include <opencv2/gapi/gproto.hpp>
#include "compiler/gmodel.hpp"
#include "compiler/gislandmodel.hpp"
#include "compiler/passes/passes.hpp"

namespace cv { namespace gimpl { namespace passes {

void addStreaming(ade::passes::PassContext &ctx)
{
    GModel::Graph gr(ctx.graph);
    if (!gr.metadata().contains<Streaming>()) {
        return;
    }

    // Note: This pass is working on a GIslandModel.
    // FIXME: May be introduce a new variant of GIslandModel to
    // deal with streams?
    auto igr = gr.metadata().get<IslandModel>().model;
    GIslandModel::Graph igm(*igr);

    // First collect all data slots & their respective original
    // data objects
    using M = std::unordered_map
        < ade::NodeHandle  // key: a GModel's data object node
        , ade::NodeHandle // value: an appropriate GIslandModel's slot node
        , ade::HandleHasher<ade::Node>
        >;
    M orig_to_isl;
    for (auto &&nh : igm.nodes()) {
        if (igm.metadata(nh).get<NodeKind>().k == NodeKind::SLOT) {
            const auto &orig_nh = igm.metadata(nh).get<DataSlot>().original_data_node;
            orig_to_isl[orig_nh] = nh;
        }
    }

    // Now walk through the list of input slots and connect those
    // to a Streaming source.
    const auto proto = gr.metadata().get<Protocol>();
    for (auto &&it : ade::util::indexed(proto.in_nhs)) {
        const auto in_idx = ade::util::index(it);
        const auto in_nh  = ade::util::value(it);
        auto emit_nh = GIslandModel::mkEmitNode(igm, in_idx);
        igm.link(emit_nh, orig_to_isl.at(in_nh));
    }
}

}}} // cv::gimpl::passes
