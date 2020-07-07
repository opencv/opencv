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

#include <inference_engine.hpp>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

namespace cv {
namespace gimpl {
namespace ie {

struct IECompiled {
#if INF_ENGINE_RELEASE < 2019020000  // < 2019.R2
    InferenceEngine::InferencePlugin   this_plugin;
#else
    InferenceEngine::Core              this_core;
#endif
    InferenceEngine::ExecutableNetwork this_network;
    InferenceEngine::InferRequest      this_request;
};

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

    // Actual data of all resources in graph (both internal and external)
    Mag m_res;

    // Execution helpers
    GArg packArg(const GArg &arg);

public:
    GIEExecutable(const ade::Graph                   &graph,
                  const std::vector<ade::NodeHandle> &nodes);

    virtual inline bool canReshape() const override { return false; }
    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override {
        GAPI_Assert(false); // Not implemented yet
    }

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};

}}}

#endif // HAVE_INF_ENGINE
#endif // OPENCV_GAPI_GIEBACKEND_HPP
