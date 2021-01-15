// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <iostream>

#include <ade/util/zip_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>

#include "api/gproto_priv.hpp" // ptr(GRunArgP)
#include "executor/gexecutor.hpp"
#include "compiler/passes/passes.hpp"

cv::gimpl::GExecutor::GExecutor(std::unique_ptr<ade::Graph> &&g_model)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_gm(*m_orig_graph)
    , m_gim(*m_island_graph)
{
}

void cv::gimpl::GExecutor::run(cv::gimpl::GRuntimeArgs &&args)
{
    // (2)
    const auto proto = m_gm.metadata().get<Protocol>();

    // Basic check if input/output arguments are correct
    // FIXME: Move to GCompiled (do once for all GExecutors)
    if (proto.inputs.size() != args.inObjs.size()) // TODO: Also check types
    {
        util::throw_error(std::logic_error
                          ("Computation's input protocol doesn\'t "
                           "match actual arguments!"));
    }
    if (proto.outputs.size() != args.outObjs.size()) // TODO: Also check types
    {
        util::throw_error(std::logic_error
                          ("Computation's output protocol doesn\'t "
                           "match actual arguments!"));
    }

    namespace util = ade::util;

    // ensure that output Mat parameters are correctly allocated
    // FIXME: avoid copy of NodeHandle and GRunRsltComp ?
    for (auto index : util::iota(proto.out_nhs.size()))
    {
        auto& nh = proto.out_nhs.at(index);
        const Data &d = m_gm.metadata(nh).get<Data>();
        if (d.shape == GShape::GMAT)
        {
            using cv::util::get;
            const auto desc = get<cv::GMatDesc>(d.meta);

            auto check_rmat = [&desc, &args, &index]()
            {
                auto& out_mat = *get<cv::RMat*>(args.outObjs.at(index));
                GAPI_Assert(desc.canDescribe(out_mat));
            };

#if !defined(GAPI_STANDALONE)
            // Building as part of OpenCV - follow OpenCV behavior In
            // the case of cv::Mat if output buffer is not enough to
            // hold the result, reallocate it
            if (cv::util::holds_alternative<cv::Mat*>(args.outObjs.at(index)))
            {
                auto& out_mat = *get<cv::Mat*>(args.outObjs.at(index));
                createMat(desc, out_mat);
            }
            // In the case of RMat check to fit required meta
            else
            {
                check_rmat();
            }
#else
            // Building standalone - output buffer should always exist,
            // and _exact_ match our inferred metadata
            if (cv::util::holds_alternative<cv::Mat*>(args.outObjs.at(index)))
            {
                auto& out_mat = *get<cv::Mat*>(args.outObjs.at(index));
                GAPI_Assert(out_mat.data != nullptr &&
                        desc.canDescribe(out_mat));
            }
            // In the case of RMat check to fit required meta
            else
            {
                check_rmat();
            }
#endif // !defined(GAPI_STANDALONE)
        }
    }

    runImpl(std::move(args));
}

const cv::gimpl::GModel::Graph& cv::gimpl::GExecutor::model() const
{
    return m_gm;
}
