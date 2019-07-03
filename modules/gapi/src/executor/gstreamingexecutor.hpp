// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GSTREAMING_EXECUTOR_HPP
#define OPENCV_GAPI_GSTREAMING_EXECUTOR_HPP

#include <memory> // unique_ptr, shared_ptr

#include <utility> // tuple, required by magazine
#include <unordered_map> // required by magazine

#include <ade/graph.hpp>

#include "backends/common/gbackend.hpp"

namespace cv {
namespace gimpl {

// FIXME: Currently all GExecutor comments apply also
// to this one. Please document it separately in the future.

class GStreamingExecutor
{
protected:
    std::unique_ptr<ade::Graph> m_orig_graph;
    std::shared_ptr<ade::Graph> m_island_graph;

    cv::gimpl::GModel::Graph       m_gm;  // FIXME: make const?
    cv::gimpl::GIslandModel::Graph m_gim; // FIXME: make const?

    // FIXME: Naive executor details are here for now
    // but then it should be moved to another place
    struct OpDesc
    {
        std::vector<RcDesc> in_objects;
        std::vector<RcDesc> out_objects;
        std::shared_ptr<GIslandExecutable> isl_exec;
    };
    std::vector<OpDesc> m_ops;

    struct DataDesc
    {
        ade::NodeHandle slot_nh;
        ade::NodeHandle data_nh;
    };
    std::vector<DataDesc> m_slots;

    // Order in this vector follows the GComputaion's protocol
    std::vector<ade::NodeHandle> m_emitters;

    Mag m_res;

    void initResource(const ade::NodeHandle &orig_nh); // FIXME: shouldn't it be RcDesc?

public:
    explicit GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model);
    void setSource(GRunArgs &&args);
    void start();
    bool pull(cv::GRunArgsP &&outs);
    void stop();
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_EXECUTOR_HPP
