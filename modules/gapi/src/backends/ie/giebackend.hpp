// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation

#ifndef OPENCV_GAPI_GIEBACKEND_HPP
#define OPENCV_GAPI_GIEBACKEND_HPP

// Include anyway - cv::gapi::ie::backend() still needs to be defined
#include "opencv2/gapi/infer/ie.hpp"

#ifdef HAVE_INF_ENGINE

#include <ade/util/algorithm.hpp> // type_list_index
#include <condition_variable>

#include <inference_engine.hpp>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

#include "backends/ie/giebackend/giewrapper.hpp" // wrap::Plugin

namespace cv {
namespace gimpl {
namespace ie {

struct IECompiled {
    std::vector<InferenceEngine::InferRequest> createInferRequests();

    cv::gapi::ie::detail::ParamDesc     params;
    cv::gimpl::ie::wrap::Plugin         this_plugin;
    InferenceEngine::ExecutableNetwork  this_network;
};

class RequestPool;

class GIEExecutable final: public GIslandExecutable
{
    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;

    // The only executable stuff in this graph
    // (assuming it is always single-op)
    ade::NodeHandle this_nh;
    IECompiled this_iec;

    // List of all resources in graph (both internal and external)
    std::vector<ade::NodeHandle> m_dataNodes;

    // To manage multiple async requests
    std::unique_ptr<RequestPool> m_reqPool;

public:
    GIEExecutable(const ade::Graph                   &graph,
                  const std::vector<ade::NodeHandle> &nodes);

    virtual inline bool canReshape() const override { return false; }
    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override {
        GAPI_Error("InternalError"); // Not implemented yet
    }

    virtual void run(std::vector<InObj>  &&,
                     std::vector<OutObj> &&) override {
        GAPI_Error("Not implemented");
    }

    virtual void run(GIslandExecutable::IInput  &in,
                     GIslandExecutable::IOutput &out) override;

};

}}}

#endif // HAVE_INF_ENGINE
#endif // OPENCV_GAPI_GIEBACKEND_HPP
