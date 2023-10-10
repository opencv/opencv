// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation


#ifndef OPENCV_GAPI_GOVCVBACKEND_HPP
#define OPENCV_GAPI_GOVCVBACKEND_HPP

#include "backends/ov/ovdef.hpp"
#ifdef HAVE_OPENVINO_2_0

#include <map>
#include <unordered_map>
#include <tuple>
#include <ade/util/algorithm.hpp> // type_list_index

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>

#include "api/gorigin.hpp"
#include "compiler/gislandmodel.hpp"
#include "backends/common/gbackend.hpp"
#include "backends/ov/govcvkernel.hpp"

namespace cv { namespace gimpl {

struct OVCVUnit
{
    static const char *name() { return "OVCVKernel"; }
    GOVCVKernel k;
};

class GOVCVExecutable final: public GIslandExecutable
{
public:
    GOVCVExecutable(const ade::Graph&                   graph,
                    const std::vector<ade::NodeHandle>& nodes,
                    const std::vector<cv::gimpl::Data>& ins_data,
                    const std::vector<cv::gimpl::Data>& outs_data);

    virtual inline bool canReshape() const override { return false; }
    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override
    {
        util::throw_error(std::logic_error("GOVCVExecutable::reshape() should never be called"));
    }

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;

private:
    void bindInArg  (const RcDesc &rc, const GRunArg  &arg);
    void bindOutArg (const RcDesc &rc, const GRunArgP &arg);

    void compile(const std::vector<cv::gimpl::Data>& ins_data,
                 const std::vector<cv::gimpl::Data>& outs_data);

    // FIXME User should also can pass config via compile args,
    // and these args should be somehow unified with the NN params
    // (w.r.t. device and plugin configuration)
    void initConfig();

    GArg packArg(const GArg &arg);

    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;

    std::vector<ade::NodeHandle> m_all_ops;

    using Mag = detail::magazine<ov::Output<ov::Node> >;
    Mag m_res;

    std::shared_ptr<ov::Model> m_ov_model;
    ov::CompiledModel m_ov_compiled;
    ov::InferRequest m_ov_req;

    // These two vectors define mapping between G-API Island inputs/outputs
    // and the synthesized OpenVINO model:
    // - key: G-API's internal Resource ID
    // - val: Index of ov::Model Param/Result
    std::unordered_map<std::size_t, std::size_t> m_param_remap;
    std::unordered_map<std::size_t, std::size_t> m_result_remap;
};

} // gimpl
} // cv

#endif // HAVE_OPENVINO_2_0
#endif // OPENCV_GAPI_GOVCVBACKEND_HPP
