// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#ifndef OPENCV_GAPI_GIEBACKEND_HPP
#define OPENCV_GAPI_GIEBACKEND_HPP

// Include anyway - cv::gapi::ie::backend() still needs to be defined
#include "opencv2/gapi/infer/ie.hpp"

#ifdef HAVE_INF_ENGINE

#include <ade/util/algorithm.hpp> // type_list_index

////////////////////////////////////////////////////////////////////////////////
// FIXME: Suppress deprecation warnings for OpenVINO 2019R2+
// BEGIN {{{
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(disable: 4996)  // was declared deprecated
#endif

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#include <inference_engine.hpp>

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
// END }}}
////////////////////////////////////////////////////////////////////////////////

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

namespace cv {
namespace gimpl {
namespace ie {

struct IECompiled {
    InferenceEngine::InferencePlugin   this_plugin;
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
