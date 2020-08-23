// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation

#include "precomp.hpp"

#include <functional>
#include <unordered_set>

#include <ade/util/algorithm.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/util/chain_range.hpp>
#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!
#include "api/render_ocv.hpp"

#include "backends/render/grenderocvbackend.hpp"

#include <opencv2/gapi/render/render.hpp>
#include "api/ocv_mask_creator.hpp"
#include "api/ft_render.hpp"


using GRenderModel = ade::TypedGraph
    < cv::gimpl::render::ocv::RenderUnit
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstRenderModel = ade::ConstTypedGraph
    < cv::gimpl::render::ocv::RenderUnit
    >;

cv::gimpl::render::ocv::GRenderExecutable::GRenderExecutable(const ade::Graph &g,
                                                             const std::vector<ade::NodeHandle> &nodes,
                                                             std::unique_ptr<cv::gapi::wip::draw::FTTextRender>&& ftpr)
    : m_g(g), m_gm(m_g), m_ftpr(std::move(ftpr)) {
        GConstRenderModel gcm(m_g);

        auto is_op = [&](ade::NodeHandle nh) {
            return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
        };

        auto it = ade::util::find_if(nodes, is_op);

        GAPI_Assert(it != nodes.end());
        this_nh = *it;

        if (!std::none_of(std::next(it), nodes.end(), is_op)) {
            util::throw_error(std::logic_error("Multi-node rendering is not supported!"));
        }
}

void cv::gimpl::render::ocv::GRenderExecutable::run(std::vector<InObj>  &&input_objs,
                                                    std::vector<OutObj> &&output_objs) {
    GConstRenderModel gcm(m_g);

    for (auto& it : input_objs)   magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : output_objs)  magazine::bindOutArg(m_res, it.first, it.second);

    const auto &op = m_gm.metadata(this_nh).get<Op>();

    // Initialize kernel's execution context:
    // - Input parameters
    GCPUContext context;
    context.m_args.reserve(op.args.size());
    using namespace std::placeholders;
    ade::util::transform(op.args,
                          std::back_inserter(context.m_args),
                          std::bind(&GRenderExecutable::packArg, this, _1));

    // - Output parameters.
    for (const auto &out_it : ade::util::indexed(op.outs)) {
        // FIXME: Can the same GArg type resolution mechanism be reused here?
        const auto out_port  = ade::util::index(out_it);
        const auto out_desc  = ade::util::value(out_it);
        context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc);
    }

    auto k = gcm.metadata(this_nh).get<RenderUnit>().k;

    context.m_args.emplace_back(m_ftpr.get());

    k.m_runF(context);

    for (auto &it : output_objs) magazine::writeBack(m_res, it.first, it.second);
}

cv::GArg cv::gimpl::render::ocv::GRenderExecutable::packArg(const cv::GArg &arg) {
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY);

    if (arg.kind != cv::detail::ArgKind::GOBJREF) {
        util::throw_error(std::logic_error("Render supports G-types ONLY!"));
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT:   return GArg(m_res.slot<cv::Mat>()[ref.id]);
    case GShape::GARRAY: return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

namespace {
    class GRenderBackendImpl final: public cv::gapi::GBackend::Priv {
        virtual void unpackKernel(ade::Graph &gr,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override {
            GRenderModel rm(gr);
            auto render_impl = cv::util::any_cast<cv::GCPUKernel>(impl.opaque);
            rm.metadata(op_node).set(cv::gimpl::render::ocv::RenderUnit{render_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs& args,
                             const std::vector<ade::NodeHandle> &nodes) const override {

            using namespace cv::gapi::wip::draw;
            auto has_freetype_font = cv::gapi::getCompileArg<freetype_font>(args);
            std::unique_ptr<FTTextRender> ftpr;
            if (has_freetype_font)
            {
#ifndef HAVE_FREETYPE
                throw std::runtime_error("Freetype not found");
#else
                ftpr.reset(new FTTextRender(has_freetype_font.value().path));
#endif
            }
            return EPtr{new cv::gimpl::render::ocv::GRenderExecutable(graph, nodes, std::move(ftpr))};
        }
    };
}

cv::gapi::GBackend cv::gapi::render::ocv::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GRenderBackendImpl>());
    return this_backend;
}
