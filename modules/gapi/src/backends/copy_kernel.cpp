//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you ("License"). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit
// this software or the related documents without Intel's prior written
// permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <opencv2/gapi/util/throw.hpp> // throw_error

#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"

#include "copy_kernel.hpp"

namespace {

class GCopyExecutable final: public cv::gimpl::GIslandExecutable {
    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override {
        // NB: Is it really needed in non-streaming  mode ?
        GAPI_Assert(false && "Not implemented");
    }

    virtual void run(GIslandExecutable::IInput &in,
                     GIslandExecutable::IOutput &out) override;

    virtual bool canReshape() const override { return false;  }
    virtual void reshape(ade::Graph&, const cv::GCompileArgs&) override {
        cv::util::throw_error(std::logic_error("GCopyExecutable::reshape() is not supported"));
    }

public:
    GCopyExecutable(const ade::Graph                   &,
                    const std::vector<ade::NodeHandle> &);
};

void GCopyExecutable::run(GIslandExecutable::IInput  &in,
                          GIslandExecutable::IOutput &out) {
    while (true) {
        const auto in_msg = in.get();
        if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg)) {
            out.post(cv::gimpl::EndOfStream{});
            return;
        }

        // Do nothing smart here, just repost the input object to the output
        const cv::GRunArgs &in_args = cv::util::get<cv::GRunArgs>(in_msg);
        GAPI_Assert(in_args.size() == 1u);

        cv::GRunArgP out_arg = out.get(0);
        *cv::util::get<cv::MediaFrame*>(out_arg) = cv::util::get<cv::MediaFrame>(in_args[0]);
        out.post(std::move(out_arg));
    }
}


class GCopyBackendImpl final: public cv::gapi::GBackend::Priv {
    virtual void unpackKernel(ade::Graph            &,
                              const ade::NodeHandle &,
                              const cv::GKernelImpl &) override {
        // literally do nothing here
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &nodes) const override {
        return EPtr{new GCopyExecutable(graph, nodes)};
    }

    virtual cv::gapi::GKernelPackage auxiliaryKernels() const override {
        return cv::gapi::kernels<cv::gimpl::GCopyKernel>();
    }
};

GCopyExecutable::GCopyExecutable(const ade::Graph&,
                                 const std::vector<ade::NodeHandle>&)
{
}


} // anonymous namespace

cv::gapi::GBackend cv::gimpl::GCopyKernel::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GCopyBackendImpl>());
    return this_backend;
}

cv::GKernelImpl cv::gimpl::GCopyKernel::kernel() {
    return cv::GKernelImpl{};
}

cv::gapi::GKernelPackage cv::gapi::streaming::kernels() {
    return cv::gapi::kernels<cv::gimpl::GCopyKernel>();
}
