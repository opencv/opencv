// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2022 Intel Corporation

#include "precomp.hpp"

#include <memory> // make_shared
#include <list>

#include <ade/util/zip_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>
#include <logger.hpp>

#include "utils/itt.hpp"
#include "compiler/passes/passes.hpp"
#include "backends/common/gbackend.hpp" // createMat


#include "executor/gstreamingexecutor.hpp"

namespace {

void sync_data(cv::GRunArgs &results, cv::GRunArgsP &outputs) {
    for (auto && it : ade::util::zip(ade::util::toRange(outputs),
                                     ade::util::toRange(results))) {
        auto &out_obj = std::get<0>(it);
        auto &res_obj = std::get<1>(it);

        // FIXME: this conversion should be unified
        using T = cv::GRunArgP;
        switch (out_obj.index()) {
        case T::index_of<cv::Mat*>():
        {
            auto out_mat_p = cv::util::get<cv::Mat*>(out_obj);
            auto view = cv::util::get<cv::RMat>(res_obj).access(cv::RMat::Access::R);
            *out_mat_p = cv::gimpl::asMat(view).clone();
        } break;
        case T::index_of<cv::RMat*>():
            *cv::util::get<cv::RMat*>(out_obj) = std::move(cv::util::get<cv::RMat>(res_obj));
            break;
        case T::index_of<cv::Scalar*>():
            *cv::util::get<cv::Scalar*>(out_obj) = std::move(cv::util::get<cv::Scalar>(res_obj));
            break;
        case T::index_of<cv::detail::VectorRef>():
            cv::util::get<cv::detail::VectorRef>(out_obj).mov(cv::util::get<cv::detail::VectorRef>(res_obj));
            break;
        case T::index_of<cv::detail::OpaqueRef>():
            cv::util::get<cv::detail::OpaqueRef>(out_obj).mov(cv::util::get<cv::detail::OpaqueRef>(res_obj));
            break;
        case T::index_of<cv::MediaFrame*>():
            *cv::util::get<cv::MediaFrame*>(out_obj) = std::move(cv::util::get<cv::MediaFrame>(res_obj));
            break;
        default:
            GAPI_Assert(false && "This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
    }
}

} // anonymous namespace

// GStreamingExecutor expects compile arguments as input to have possibility to do
// proper graph reshape and islands recompilation
cv::gimpl::GStreamingExecutor::GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model,
                                                  const GCompileArgs &comp_args)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_comp_args(comp_args)
    , m_gim(*m_island_graph) {
}

cv::gimpl::GStreamingExecutor::~GStreamingExecutor() {
    // FIXME: this is a temporary try-catch exception hadling.
    // Need to eliminate throwings from stop(): make stop_impl() & wait_shutdown() no_throw
    try {
        if (state == State::READY || state == State::RUNNING) {
            stop();
        }
    } catch (const std::exception& e) {
        std::stringstream message;
        message << "~GStreamingExecutor() threw exception with message '" << e.what() << "'\n";
        GAPI_LOG_WARNING(NULL, message.str());
    }
}

void cv::gimpl::GStreamingExecutor::start() {
    if (state == State::STOPPED) {
        util::throw_error(std::logic_error("Please call setSource() before start() "
                                           "if the pipeline has been already stopped"));
    }
    GAPI_Assert(state == State::READY);
    start_impl();
}

bool cv::gimpl::GStreamingExecutor::pull(cv::GRunArgsP &&outs) {
    GAPI_ITT_STATIC_LOCAL_HANDLE(pull_hndl, "GStreamingExecutor::pull");
    GAPI_ITT_AUTO_TRACE_GUARD(pull_hndl);

    if (state == State::STOPPED) {
        return false;
    }
    GAPI_Assert(state == State::RUNNING && "GStreamingExecutor is not started");
    GAPI_Assert(GModel::Graph(*m_orig_graph).metadata().get<Protocol>().out_nhs.size() == outs.size() &&
                "Number of data objects in cv::gout() must match the number of graph outputs in cv::GOut()");

    cv::GRunArgs this_result;
    if (!pull_impl(this_result)) {
        wait_shutdown();
        return false;
    }
    sync_data(this_result, outs);
    return true;
}

bool cv::gimpl::GStreamingExecutor::try_pull(cv::GRunArgsP &&outs) {
    if (state == State::STOPPED) {
        return false;
    }

    GAPI_Assert(GModel::Graph(*m_orig_graph).metadata().get<Protocol>().out_nhs.size() == outs.size() &&
                "Number of data objects in cv::gout() must match the number of graph outputs in cv::GOut()");

    cv::GRunArgs this_result;
    if (!try_pull_impl(this_result)) {
        wait_shutdown();
        return false;
    }
    sync_data(this_result, outs);
    return true;
}

void cv::gimpl::GStreamingExecutor::stop() {
    if (state == State::STOPPED) {
        return;
    }
    stop_impl();
    wait_shutdown();
}

bool cv::gimpl::GStreamingExecutor::running() const {
    return (state == State::RUNNING);
}
