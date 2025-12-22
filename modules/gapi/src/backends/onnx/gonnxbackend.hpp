// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_GONNXBACKEND_HPP
#define OPENCV_GAPI_GONNXBACKEND_HPP

#include "opencv2/gapi/infer/onnx.hpp"
#ifdef HAVE_ONNX

#include <onnxruntime_cxx_api.h>
#include <ade/util/algorithm.hpp> // type_list_index

#include "backends/common/gbackend.hpp"

namespace cv {
namespace gimpl {
namespace onnx {

class GONNXExecutable final: public GIslandExecutable
{
    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;

    // The only executable stuff in this graph
    // (assuming it is always single-op)
    ade::NodeHandle this_nh;

    // List of all resources in graph (both internal and external)
    std::vector<ade::NodeHandle> m_dataNodes;

    // Actual data of all resources in graph (both internal and external)
    Mag m_res;

    // Execution helpers
    GArg packArg(const GArg &arg);

public:
    GONNXExecutable(const ade::Graph                   &graph,
                    const std::vector<ade::NodeHandle> &nodes,
                    const cv::GCompileArgs             &compileArgs);

    virtual inline bool canReshape() const override { return false; }
    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override {
        GAPI_Error("InternalError"); // Not implemented yet
    }

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};

}}} // namespace cv::gimpl::onnx

#endif // HAVE_ONNX
#endif // OPENCV_GAPI_GONNXBACKEND_HPP
