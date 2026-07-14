// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef OPENCV_GAPI_GABSTRACT_STREAMING_EXECUTOR_HPP
#define OPENCV_GAPI_GABSTRACT_STREAMING_EXECUTOR_HPP

#include <memory> // unique_ptr, shared_ptr

#include <ade/graph.hpp>

#include "backends/common/gbackend.hpp"

namespace cv {
namespace gimpl {

class GAbstractStreamingExecutor
{
protected:
    std::unique_ptr<ade::Graph> m_orig_graph;
    std::shared_ptr<ade::Graph> m_island_graph;
    cv::GCompileArgs m_comp_args;

    cv::gimpl::GIslandModel::Graph m_gim; // FIXME: make const?
    const bool m_desync;

public:
    explicit GAbstractStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model,
                                        const cv::GCompileArgs &comp_args);
    virtual ~GAbstractStreamingExecutor() = default;
    virtual void setSource(GRunArgs &&args) = 0;
    virtual void start() = 0;
    virtual bool pull(cv::GRunArgsP &&outs) = 0;
    virtual bool pull(cv::GOptRunArgsP &&outs) = 0;

    using PyPullResult = std::tuple<bool, cv::util::variant<cv::GRunArgs, cv::GOptRunArgs>>;
    virtual PyPullResult pull() = 0;

    virtual bool try_pull(cv::GRunArgsP &&outs) = 0;
    virtual void stop() = 0;
    virtual bool running() const = 0;
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GABSTRACT_STREAMING_EXECUTOR_HPP
