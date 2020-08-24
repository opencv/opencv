// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>

#include "backends/common/serialization.hpp"

std::vector<char> cv::gapi::serialize(const cv::GComputation &c) {
    cv::gimpl::s11n::ByteMemoryOutStream os;
    c.serialize(os);
    return os.data();
}

cv::GComputation cv::gapi::detail::getGraph(const std::vector<char> &p) {
    cv::gimpl::s11n::ByteMemoryInStream is(p);
    return cv::GComputation(is);
}

cv::GMetaArgs cv::gapi::detail::getMetaArgs(const std::vector<char> &p) {
    cv::gimpl::s11n::ByteMemoryInStream is(p);
    return meta_args_deserialize(is);
}

cv::GRunArgs cv::gapi::detail::getRunArgs(const std::vector<char> &p) {
    cv::gimpl::s11n::ByteMemoryInStream is(p);
    return run_args_deserialize(is);
}

std::vector<char> cv::gapi::serialize(const cv::GMetaArgs& ma)
{
    cv::gimpl::s11n::ByteMemoryOutStream os;
    serialize(os, ma);
    return os.data();
}

std::vector<char> cv::gapi::serialize(const cv::GRunArgs& ra)
{
    cv::gimpl::s11n::ByteMemoryOutStream os;
    serialize(os, ra);
    return os.data();
}

cv::GRunArgsP cv::gapi::bind(cv::GRunArgs &results)
{
    cv::GRunArgsP outputs;
    outputs.reserve(results.size());
    for (cv::GRunArg &res_obj : results)
    {
        using T = cv::GRunArg;
        switch (res_obj.index())
        {
#if !defined(GAPI_STANDALONE)
        case T::index_of<cv::UMat>() :
            outputs.emplace_back((cv::UMat*)(&(cv::util::get<cv::UMat>(res_obj))));
            break;
#endif
        case cv::GRunArg::index_of<cv::Mat>() :
            outputs.emplace_back((cv::Mat*)(&(cv::util::get<cv::Mat>(res_obj))));
            break;
        case cv::GRunArg::index_of<cv::Scalar>() :
            outputs.emplace_back((cv::Scalar*)(&(cv::util::get<cv::Scalar>(res_obj))));
            break;
        case T::index_of<cv::detail::VectorRef>() :
            outputs.emplace_back(cv::util::get<cv::detail::VectorRef>(res_obj));
            break;
        case T::index_of<cv::detail::OpaqueRef>() :
            outputs.emplace_back(cv::util::get<cv::detail::OpaqueRef>(res_obj));
            break;
        default:
            GAPI_Assert(false && "This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
    }
    return outputs;
}
