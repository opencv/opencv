// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"
#include <opencv2/gapi/garg.hpp>

cv::GRunArg::GRunArg() {
}

cv::GRunArg::GRunArg(const cv::GRunArg &arg)
    : cv::GRunArgBase(static_cast<const cv::GRunArgBase&>(arg))
    , meta(arg.meta) {
}

cv::GRunArg::GRunArg(cv::GRunArg &&arg)
    : cv::GRunArgBase(std::move(static_cast<const cv::GRunArgBase&>(arg)))
    , meta(std::move(arg.meta)) {
}

cv::GRunArg& cv::GRunArg::operator= (const cv::GRunArg &arg) {
    cv::GRunArgBase::operator=(static_cast<const cv::GRunArgBase&>(arg));
    meta = arg.meta;
    return *this;
}

cv::GRunArg& cv::GRunArg::operator= (cv::GRunArg &&arg) {
    cv::GRunArgBase::operator=(std::move(static_cast<const cv::GRunArgBase&>(arg)));
    meta = std::move(arg.meta);
    return *this;
}

// NB: Construct GRunArgsP based on passed info and store the memory in passed cv::GRunArgs.
// Needed for python bridge, because in case python user doesn't pass output arguments to apply.
void cv::detail::constructGraphOutputs(const cv::GTypesInfo &out_info,
                                       cv::GRunArgs         &args,
                                       cv::GRunArgsP        &outs)
{
    for (auto&& info : out_info)
    {
        switch (info.shape)
        {
            case cv::GShape::GMAT:
            {
                args.emplace_back(cv::Mat{});
                outs.emplace_back(&cv::util::get<cv::Mat>(args.back()));
                break;
            }
            case cv::GShape::GSCALAR:
            {
                args.emplace_back(cv::Scalar{});
                outs.emplace_back(&cv::util::get<cv::Scalar>(args.back()));
                break;
            }
            case cv::GShape::GARRAY:
            {
                cv::detail::VectorRef ref;
                util::get<cv::detail::ConstructVec>(info.ctor)(ref);
                args.emplace_back(ref);
                outs.emplace_back(cv::util::get<cv::detail::VectorRef>(args.back()));
                break;
            }
            case cv::GShape::GOPAQUE:
            {
                cv::detail::OpaqueRef ref;
                util::get<cv::detail::ConstructOpaque>(info.ctor)(ref);
                args.emplace_back(ref);
                outs.emplace_back(ref);
                break;
            }

            default:
                util::throw_error(std::logic_error("Unsupported output shape for python"));
        }
    }
}
