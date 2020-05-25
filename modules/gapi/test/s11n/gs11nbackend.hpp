// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_GS11NBACKEND_HPP
#define OPENCV_GAPI_GS11NBACKEND_HPP

#include <map>                // map
#include <unordered_map>      // unordered_map
#include <tuple>              // tuple
#include <ade/util/algorithm.hpp> // type_list_index

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>
#include "gs11nkernel.hpp"
#include "gs11nkernels.hpp"

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"
#include "compiler/passes/passes.hpp"

namespace opencv_test{ namespace s11n { namespace impl {

struct Unit
{
    static const char *name() { return "Serialization HostKernel"; }
    GS11NKernel k;
};

class GS11NExecutable final: public cv::gimpl::GIslandExecutable
{
    const ade::Graph &m_g;
    cv::gimpl::GModel::ConstGraph m_gm;

    std::shared_ptr<ade::Graph> m_gp;

    struct OperationInfo
    {
        ade::NodeHandle nh;
        cv::GMetaArgs expected_out_metas;
    };

    // Execution script, currently absolutely naive
    std::vector<OperationInfo> m_script;
    // List of all resources in graph (both internal and external)
    std::vector<ade::NodeHandle> m_dataNodes;

    // Actual data of all resources in graph (both internal and external)
    cv::gimpl::Mag m_res;
    cv::GArg packArg(const cv::GArg &arg);

public:
    GS11NExecutable(const ade::Graph                   &graph,
                   const std::vector<ade::NodeHandle> &nodes,
                   std::shared_ptr<ade::Graph> gp);

    virtual inline bool canReshape() const override { return false; }
    virtual inline void reshape(ade::Graph&, const cv::GCompileArgs&) override
    {
        // FIXME: CPU plugin is in fact reshapeable (as it was initially,
        // even before outMeta() has been introduced), so this limitation
        // should be dropped.
        cv::util::throw_error(std::logic_error("GS11NExecutable::reshape() should never be called"));
    }

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};

}}}

#endif // OPENCV_GAPI_GS11NBACKEND_HPP
