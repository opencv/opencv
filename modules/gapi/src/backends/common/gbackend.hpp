// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GBACKEND_HPP
#define OPENCV_GAPI_GBACKEND_HPP

#include <string>
#include <memory>

#include <ade/node.hpp>

#include "opencv2/gapi/garg.hpp"

#include "opencv2/gapi/util/optional.hpp"

#include "compiler/gmodel.hpp"

namespace cv {
namespace gimpl {

    // Forward declarations
    struct Data;
    struct RcDesc;

namespace magazine {
    template<typename... Ts> struct Class
    {
        template<typename T> using MapT = std::unordered_map<int, T>;
        template<typename T>       MapT<T>& slot()
        {
            return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
        }
        template<typename T> const MapT<T>& slot() const
        {
            return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
        }
    private:
        std::tuple<MapT<Ts>...> slots;
    };

} // namespace magazine
#if !defined(GAPI_STANDALONE)
using Mag = magazine::Class<cv::Mat, cv::UMat, cv::Scalar, cv::detail::VectorRef, cv::detail::OpaqueRef>;
#else
using Mag = magazine::Class<cv::Mat, cv::Scalar, cv::detail::VectorRef, cv::detail::OpaqueRef>;
#endif

namespace magazine
{
    void         GAPI_EXPORTS bindInArg (Mag& mag, const RcDesc &rc, const GRunArg  &arg, bool is_umat = false);
    void         GAPI_EXPORTS bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg, bool is_umat = false);

    void         resetInternalData(Mag& mag, const Data &d);
    cv::GRunArg  getArg    (const Mag& mag, const RcDesc &ref);
    cv::GRunArgP getObjPtr (      Mag& mag, const RcDesc &rc, bool is_umat = false);
    void         writeBack (const Mag& mag, const RcDesc &rc, GRunArgP &g_arg, bool is_umat = false);
} // namespace magazine

namespace detail
{
template<typename... Ts> struct magazine
{
    template<typename T> using MapT = std::unordered_map<int, T>;
    template<typename T>       MapT<T>& slot()
    {
        return std::get<util::type_list_index<T, Ts...>::value>(slots);
    }
    template<typename T> const MapT<T>& slot() const
    {
        return std::get<util::type_list_index<T, Ts...>::value>(slots);
    }
private:
    std::tuple<MapT<Ts>...> slots;
};
} // namespace detail

struct GRuntimeArgs
{
    GRunArgs   inObjs;
    GRunArgsP outObjs;
};

template<typename T>
inline cv::util::optional<T> getCompileArg(const cv::GCompileArgs &args)
{
    for (auto &compile_arg : args)
    {
        if (compile_arg.tag == cv::detail::CompileArgTag<T>::tag())
        {
            return cv::util::optional<T>(compile_arg.get<T>());
        }
    }
    return cv::util::optional<T>();
}

void createMat(const cv::GMatDesc& desc, cv::Mat& mat);

}} // cv::gimpl

#endif // OPENCV_GAPI_GBACKEND_HPP
