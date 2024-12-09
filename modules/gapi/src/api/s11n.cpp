// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>

#include "backends/common/serialization.hpp"

std::vector<char> cv::gapi::serialize(const cv::GComputation &c) {
    cv::gapi::s11n::ByteMemoryOutStream os;
    c.serialize(os);
    return os.data();
}

cv::GComputation cv::gapi::detail::getGraph(const std::vector<char> &p) {
    cv::gapi::s11n::ByteMemoryInStream is(p);
    return cv::GComputation(is);
}

cv::GMetaArgs cv::gapi::detail::getMetaArgs(const std::vector<char> &p) {
    cv::gapi::s11n::ByteMemoryInStream is(p);
    return meta_args_deserialize(is);
}

cv::GRunArgs cv::gapi::detail::getRunArgs(const std::vector<char> &p) {
    cv::gapi::s11n::ByteMemoryInStream is(p);
    return run_args_deserialize(is);
}

std::vector<std::string> cv::gapi::detail::getVectorOfStrings(const std::vector<char> &p) {
    cv::gapi::s11n::ByteMemoryInStream is(p);
    return vector_of_strings_deserialize(is);
}

std::vector<char> cv::gapi::serialize(const cv::GMetaArgs& ma)
{
    cv::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, ma);
    return os.data();
}

std::vector<char> cv::gapi::serialize(const cv::GRunArgs& ra)
{
    cv::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, ra);
    return os.data();
}

std::vector<char> cv::gapi::serialize(const cv::GCompileArgs& ca)
{
    cv::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, ca);
    return os.data();
}

std::vector<char> cv::gapi::serialize(const std::vector<std::string>& vs)
{
    cv::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, vs);
    return os.data();
}

// FIXME: This function should move from S11N to GRunArg-related entities.
// it has nothing to do with the S11N as it is
cv::GRunArgsP cv::gapi::bind(cv::GRunArgs &out_args)
{
    cv::GRunArgsP outputs;
    outputs.reserve(out_args.size());
    for (cv::GRunArg &res_obj : out_args)
    {
        using T = cv::GRunArg;
        switch (res_obj.index())
        {
#if !defined(GAPI_STANDALONE)
        case T::index_of<cv::UMat>() :
            outputs.emplace_back(&(cv::util::get<cv::UMat>(res_obj)));
            break;
#endif
        case cv::GRunArg::index_of<cv::Mat>() :
            outputs.emplace_back(&(cv::util::get<cv::Mat>(res_obj)));
            break;
        case cv::GRunArg::index_of<cv::Scalar>() :
            outputs.emplace_back(&(cv::util::get<cv::Scalar>(res_obj)));
            break;
        case T::index_of<cv::detail::VectorRef>() :
            outputs.emplace_back(cv::util::get<cv::detail::VectorRef>(res_obj));
            break;
        case T::index_of<cv::detail::OpaqueRef>() :
            outputs.emplace_back(cv::util::get<cv::detail::OpaqueRef>(res_obj));
            break;
        case cv::GRunArg::index_of<cv::RMat>() :
            outputs.emplace_back(&(cv::util::get<cv::RMat>(res_obj)));
            break;
        case cv::GRunArg::index_of<cv::MediaFrame>() :
            outputs.emplace_back(&(cv::util::get<cv::MediaFrame>(res_obj)));
            break;
        default:
            GAPI_Error("This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
    }
    return outputs;
}

// FIXME: move it out of s11n to api/
// FIXME: don't we have such function already?
cv::GRunArg cv::gapi::bind(cv::GRunArgP &out)
{
    using T = cv::GRunArgP;
    switch (out.index())
    {
#if !defined(GAPI_STANDALONE)
    case T::index_of<cv::UMat*>() :
        GAPI_Error("Please implement this!");
        break;
#endif

    case T::index_of<cv::detail::VectorRef>() :
        return cv::GRunArg(cv::util::get<cv::detail::VectorRef>(out));

    case T::index_of<cv::detail::OpaqueRef>() :
        return cv::GRunArg(cv::util::get<cv::detail::OpaqueRef>(out));

    case T::index_of<cv::Mat*>() :
        return cv::GRunArg(*cv::util::get<cv::Mat*>(out));

    case T::index_of<cv::Scalar*>() :
        return cv::GRunArg(*cv::util::get<cv::Scalar*>(out));

    case T::index_of<cv::RMat*>() :
        return cv::GRunArg(*cv::util::get<cv::RMat*>(out));

    case T::index_of<cv::MediaFrame*>() :
        return cv::GRunArg(*cv::util::get<cv::MediaFrame*>(out));

    default:
        // ...maybe our types were extended
        GAPI_Error("This value type is UNKNOWN!");
        break;
    }
    return cv::GRunArg();
}
