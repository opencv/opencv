// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GRENDEROCVBACKEND_HPP
#define OPENCV_GAPI_GRENDEROCVBACKEND_HPP

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/render/render.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

#include "backends/render/grenderocv.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace cv
{
namespace gimpl
{
namespace render
{
namespace ocv
{

struct RenderUnit
{
    static const char *name() { return "RenderUnit"; }
    GCPUKernel k;
};

class GRenderExecutable final: public GIslandExecutable
{
    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;
    std::unique_ptr<cv::gapi::wip::draw::FTTextRender> m_ftpr;

    // The only executable stuff in this graph
    // (assuming it is always single-op)
    ade::NodeHandle this_nh;

    //// Actual data of all resources in graph (both internal and external)
    Mag m_res;

    //// Execution helpers
    GArg packArg(const GArg &arg);

public:
    GRenderExecutable(const ade::Graph                   &graph,
                      const std::vector<ade::NodeHandle> &nodes,
                      std::unique_ptr<cv::gapi::wip::draw::FTTextRender>&& ftpr);

    virtual inline bool canReshape() const override { return false; }

    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override {
        GAPI_Assert(false); // Not implemented yet
    }

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};

} // namespace ocv
} // namespace render
} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GRENDEROCVBACKEND_HPP
