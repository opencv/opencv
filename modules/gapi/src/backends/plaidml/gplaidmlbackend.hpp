// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifdef HAVE_PLAIDML

#ifndef OPENCV_GAPI_GPLAIDMLBACKEND_HPP
#define OPENCV_GAPI_GPLAIDMLBACKEND_HPP

#include <map>                // map
#include <unordered_map>      // unordered_map
#include <tuple>              // tuple
#include <ade/util/algorithm.hpp> // type_list_index

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/plaidml/gplaidmlkernel.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"

#include "compiler/gislandmodel.hpp"

#include <plaidml2/exec/exec.h>
#include <plaidml2/core/core.h>

namespace cv { namespace gimpl {

struct PlaidMLUnit
{
    static const char *name() { return "PlaidMLKernel"; }
    GPlaidMLKernel k;
};

class GPlaidMLExecutable final: public GIslandExecutable
{
public:
    struct Config
    {
        std::string dev_id;
        std::string trg_id;
    };

    GPlaidMLExecutable(Config                              cfg,
                       const ade::Graph&                   graph,
                       const std::vector<ade::NodeHandle>& nodes,
                       const std::vector<cv::gimpl::Data>& ins_data,
                       const std::vector<cv::gimpl::Data>& outs_data);

    virtual inline bool canReshape() const override { return false; }

    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override
    {
        util::throw_error(std::logic_error("GPlaidMLExecutable::reshape() should never be called"));
    }

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;

private:
    void initBuffers(const std::vector<cv::gimpl::Data>& ins_data,
                     std::vector<plaidml::exec::Binding>& bindings);

    void bindInArg  (const RcDesc &rc, const GRunArg  &arg);
    void bindOutArg (const RcDesc &rc, const GRunArgP &arg);

    void compile(const std::vector<cv::gimpl::Data>& ins_data,
                 const std::vector<cv::gimpl::Data>& outs_data);

    // FIXME User also can pass config via compile args ?
    void initConfig();

    GArg packArg(const GArg &arg);

    Config m_cfg;

    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;

    std::vector<ade::NodeHandle> m_all_ops;

    std::vector<size_t> output_ids_;

    std::unique_ptr<plaidml::exec::Binder>     binder_;
    std::shared_ptr<plaidml::exec::Executable> exec_;

    std::vector<plaidml::exec::Binding> input_bindings_;
    std::vector<plaidml::exec::Binding> output_bindings_;

    using Mag = detail::magazine<plaidml::edsl::Tensor, plaidml::Buffer*>;
    Mag m_res;
};

}}

#endif // OPENCV_GAPI_GPLAIDMLBACKEND_HPP

#endif // HAVE_PLAIDML
