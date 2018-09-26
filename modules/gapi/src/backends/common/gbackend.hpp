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
#include "opencv2/gapi/own/mat.hpp"

#include "opencv2/gapi/util/optional.hpp"
#include "opencv2/gapi/own/scalar.hpp"

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

using Mag = magazine::Class<cv::gapi::own::Mat, cv::gapi::own::Scalar, cv::detail::VectorRef>;

namespace magazine
{
    void         bindInArg (Mag& mag, const RcDesc &rc, const GRunArg  &arg);
    void         bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg);

    void         resetInternalData(Mag& mag, const Data &d);
    cv::GRunArg  getArg    (const Mag& mag, const RcDesc &ref);
    cv::GRunArgP getObjPtr (      Mag& mag, const RcDesc &rc);
    void         writeBack (const Mag& mag, const RcDesc &rc, GRunArgP &g_arg);
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



}} // cv::gimpl

#endif // OPENCV_GAPI_GBACKEND_HPP
