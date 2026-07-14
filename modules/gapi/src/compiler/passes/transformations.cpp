// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "precomp.hpp"

#include <ade/util/zip_range.hpp>
#include <ade/graph.hpp>

#include "api/gcomputation_priv.hpp"

#include "compiler/gmodel.hpp"
#include "compiler/gmodelbuilder.hpp"
#include "compiler/passes/passes.hpp"
#include "compiler/passes/pattern_matching.hpp"

#include <sstream>

namespace cv { namespace gimpl { namespace passes {
namespace
{
using Graph = GModel::Graph;
using Metadata = typename Graph::CMetadataT;

// Checks pairs of {pattern node, substitute node} and asserts if there are any incompatibilities
void checkDataNodes(const Graph& pattern,
                    const Graph& substitute,
                    const std::vector<ade::NodeHandle>& patternNodes,
                    const std::vector<ade::NodeHandle>& substituteNodes)
{
    for (auto it : ade::util::zip(patternNodes, substituteNodes)) {
        auto pNodeMeta = pattern.metadata(std::get<0>(it));
        auto sNodeMeta = substitute.metadata(std::get<1>(it));
        GAPI_Assert(pNodeMeta.get<NodeType>().t == NodeType::DATA);
        GAPI_Assert(pNodeMeta.get<NodeType>().t == sNodeMeta.get<NodeType>().t);
        GAPI_Assert(pNodeMeta.get<Data>().shape == sNodeMeta.get<Data>().shape);
    }
}

// Checks compatibility of pattern and substitute graphs based on in/out nodes
void checkCompatibility(const Graph& pattern,
                        const Graph& substitute,
                        const Protocol& patternP,
                        const Protocol& substituteP)
{
    const auto& patternDataInputs = patternP.in_nhs;
    const auto& patternDataOutputs = patternP.out_nhs;

    const auto& substituteDataInputs = substituteP.in_nhs;
    const auto& substituteDataOutputs = substituteP.out_nhs;

    // number of data nodes must be the same
    GAPI_Assert(patternDataInputs.size() == substituteDataInputs.size());
    GAPI_Assert(patternDataOutputs.size() == substituteDataOutputs.size());

    // for each pattern input node, verify a corresponding substitute input node
    checkDataNodes(pattern, substitute, patternDataInputs, substituteDataInputs);

    // for each pattern output node, verify a corresponding substitute output node
    checkDataNodes(pattern, substitute, patternDataOutputs, substituteDataOutputs);
}

// Tries to substitute __single__ pattern with substitute in the given graph
bool tryToSubstitute(ade::Graph& main,
                     const std::unique_ptr<ade::Graph>& patternG,
                     const cv::GComputation& substitute)
{
    GModel::Graph gm(main);

    // 1. find a pattern in main graph
    auto match1 = findMatches(*patternG, gm);
    if (!match1.ok()) {
        return false;
    }

    // 2. build substitute graph inside the main graph
    cv::gimpl::GModelBuilder builder(main);
    auto expr = cv::util::get<cv::GComputation::Priv::Expr>(substitute.priv().m_shape);
    const auto& proto_slots = builder.put(expr.m_ins, expr.m_outs);
    Protocol substituteP;
    std::tie(substituteP.inputs, substituteP.outputs, substituteP.in_nhs, substituteP.out_nhs) =
        proto_slots;

    const Protocol& patternP = GModel::Graph(*patternG).metadata().get<Protocol>();

    // 3. check that pattern and substitute are compatible
    // FIXME: in theory, we should always have compatible pattern/substitute. if not, we're in
    //        half-completed state where some transformations are already applied - what can we do
    //        to handle the situation better?  -- use transactional API as in fuse_islands pass?
    checkCompatibility(*patternG, gm, patternP, substituteP);

    // 4. make substitution
    performSubstitution(gm, patternP, substituteP, match1);

    return true;
}
}  // anonymous namespace

void applyTransformations(ade::passes::PassContext& ctx,
                          const GKernelPackage& pkg,
                          const std::vector<std::unique_ptr<ade::Graph>>& patterns)
{
    const auto& transforms = pkg.get_transformations();
    const auto size = transforms.size();
    if (0u == size) return;
    // Note: patterns are already generated at this point
    GAPI_Assert(patterns.size() == transforms.size());

    // transform as long as it is possible
    bool canTransform = true;
    while (canTransform)
    {
        canTransform = false;

        // iterate through every transformation and try to transform graph parts
        for (auto it : ade::util::zip(ade::util::toRange(transforms), ade::util::toRange(patterns)))
        {
            const auto& t = std::get<0>(it);
            auto& pattern = std::get<1>(it);  // Note: using pre-created graphs
            GAPI_Assert(nullptr != pattern);

            // if transformation is successful (pattern found and substituted), it is possible that
            // other transformations will also be successful, so set canTransform to the returned
            // value from tryToSubstitute
            canTransform = tryToSubstitute(ctx.graph, pattern, t.substitute());

            // Note: apply the __same__ substitution as many times as possible and only after go to
            //       the next one. BUT it can happen that after applying some substitution, some
            //       _previous_ patterns will also be found and these will be applied first
            if (canTransform) {
                break;
            }
        }
    }
}
}  // namespace passes
}  // namespace gimpl
}  // namespace cv
