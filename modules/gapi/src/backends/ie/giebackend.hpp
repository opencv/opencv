// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GIEBACKEND_HPP
#define OPENCV_GAPI_GIEBACKEND_HPP

#include <map>                // map
#include <unordered_map>      // unordered_map
#include <tuple>              // tuple
#include <ade/util/algorithm.hpp> // type_list_index
#include <inference_engine.hpp>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/infer/ie.hpp>

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

    virtual inline bool canReshape() const override { return true; }
    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override {
        GAPI_Assert(false); // Not implemented yet
    }

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};

}}}

#endif // OPENCV_GAPI_GIEBACKEND_HPP
