// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_GOVBACKEND_HPP
#define OPENCV_GAPI_GOVBACKEND_HPP

// Include anyway - cv::gapi::ov::backend() still needs to be defined
#include "opencv2/gapi/infer/ov.hpp"

#if defined HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000

#include <openvino/openvino.hpp>

#include "backends/common/gbackend.hpp"

namespace cv {
namespace gimpl {
namespace ov {

struct OVCompiled {
    ::ov::CompiledModel compiled_model;
};

class RequestPool;

struct Options {
    // Only performs inference of the model
    // without i/o data transfer if enabled.
    bool inference_only = false;
};

class GOVExecutable final: public GIslandExecutable
{
    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;

    // The only executable stuff in this graph
    // (assuming it is always single-op)
    ade::NodeHandle this_nh;
    OVCompiled compiled;

    // List of all resources in graph (both internal and external)
    std::vector<ade::NodeHandle> m_dataNodes;

    // To manage multiple async requests
    std::unique_ptr<RequestPool> m_reqPool;

    // To manage additional execution options
    Options m_options;
    std::shared_ptr<void> listenerRemover;
    void setWorkLoadType(const std::string &type);

public:
    GOVExecutable(const ade::Graph                   &graph,
                  const cv::GCompileArgs             &compileArgs,
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

#endif // HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000
#endif // OPENCV_GAPI_GOVBACKEND_HPP
