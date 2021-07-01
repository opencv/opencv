// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_GSTREAMING_COMPILED_PRIV_HPP
#define OPENCV_GAPI_GSTREAMING_COMPILED_PRIV_HPP

#include <memory> // unique_ptr
#include "executor/gstreamingexecutor.hpp"

namespace cv {

namespace gimpl
{
    struct GRuntimeArgs;
};

// FIXME: GAPI_EXPORTS is here only due to tests and Windows linker issues
// FIXME: It seems it clearly duplicates the GStreamingCompiled and
// GStreamingIntrinExecutable APIs so is highly redundant now.
// Same applies to GCompiled/GCompiled::Priv/GExecutor.
class GAPI_EXPORTS GStreamingCompiled::Priv
{
    GMetaArgs  m_metas;    // passed by user
    GMetaArgs  m_outMetas; // inferred by compiler
    std::unique_ptr<cv::gimpl::GStreamingExecutor> m_exec;

    // NB: Used by python wrapper to clarify input/output types
    GTypesInfo m_out_info;
    GTypesInfo m_in_info;

public:
    void setup(const GMetaArgs &metaArgs,
               const GMetaArgs &outMetas,
               std::unique_ptr<cv::gimpl::GStreamingExecutor> &&pE);
    void setup(std::unique_ptr<cv::gimpl::GStreamingExecutor> &&pE);
    bool isEmpty() const;

    const GMetaArgs& metas() const;
    const GMetaArgs& outMetas() const;

    void setSource(GRunArgs &&args);
    void start();
    bool pull(cv::GRunArgsP &&outs);
    bool pull(cv::GOptRunArgsP &&outs);
    std::tuple<bool, cv::util::variant<cv::GRunArgs, cv::GOptRunArgs>> pull();
    bool try_pull(cv::GRunArgsP &&outs);
    void stop();

    bool running() const;

    void setOutInfo(const GTypesInfo& info) { m_out_info = std::move(info); }
    const GTypesInfo& outInfo() const { return m_out_info; }

    void setInInfo(const GTypesInfo& info) { m_in_info = std::move(info); }
    const GTypesInfo& inInfo() const { return m_in_info; }
};

} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_COMPILED_PRIV_HPP
