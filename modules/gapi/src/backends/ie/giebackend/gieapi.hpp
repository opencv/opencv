// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_IEAPI_HPP
#define OPENCV_GAPI_IEAPI_HPP

#ifdef HAVE_INF_ENGINE

#include <inference_engine.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>

#include "opencv2/gapi/infer/ie.hpp"

namespace IE = InferenceEngine;

namespace cv {
namespace gapi {
namespace ie {
namespace wrap {

std::vector<std::string> getExtensions(const cv::gapi::ie::detail::ParamDesc& params);

#if INF_ENGINE_RELEASE < 2020000000  // < 2020.1
// Load extensions (taken from DNN module)
std::vector<std::string> getExtensions(const cv::gapi::ie::detail::ParamDesc& params) {
    std::vector<std::string> candidates;
    if (params.device_id == "CPU" || params.device_id == "FPGA")
    {
        const std::string suffixes[] = { "_avx2", "_sse4", ""};
        const bool haveFeature[] = {
            cv::checkHardwareSupport(CPU_AVX2),
            cv::checkHardwareSupport(CPU_SSE4_2),
            true
        };
        for (auto &&it : ade::util::zip(ade::util::toRange(suffixes),
                                        ade::util::toRange(haveFeature)))
        {
            std::string suffix;
            bool available = false;
            std::tie(suffix, available) = it;
            if (!available) continue;
#ifdef _WIN32
            candidates.push_back("cpu_extension" + suffix + ".dll");
#elif defined(__APPLE__)
            candidates.push_back("libcpu_extension" + suffix + ".so");  // built as loadable module
            candidates.push_back("libcpu_extension" + suffix + ".dylib");  // built as shared library
#else
            candidates.push_back("libcpu_extension" + suffix + ".so");
#endif  // _WIN32
        }
    }
    return candidates;
}
#else // >= 2020.1
std::vector<std::string> getExtensions(const cv::gapi::ie::detail::ParamDesc&) {
    return std::vector<std::string>();
}
#endif // INF_ENGINE_RELEASE < 2020000000

#if INF_ENGINE_RELEASE < 2019020000  // < 2019.R2
IE::InferencePlugin getPlugin(const cv::gapi::ie::detail::ParamDesc& params);
inline IE::ExecutableNetwork loadNetwork(      IE::InferencePlugin& plugin,
                                         const IE::CNNNetwork&      net,
                                         const cv::gapi::ie::detail::ParamDesc&);

IE::InferencePlugin getPlugin(const cv::gapi::ie::detail::ParamDesc& params) {
    auto plugin = IE::PluginDispatcher().getPluginByDevice(params.device_id);
    if (params.device_id == "CPU" || params.device_id == "FPGA")
    {
        for (auto &&extlib : getExtensions(params))
        {
            try
            {
                plugin.AddExtension(IE::make_so_pointer<IE::IExtension>(extlib));
                CV_LOG_INFO(NULL, "DNN-IE: Loaded extension plugin: " << extlib);
                break;
            }
            catch(...)
            {
                CV_LOG_INFO(NULL, "Failed to load IE extension: " << extlib);
            }
        }
    }
    return plugin;
}

inline IE::ExecutableNetwork loadNetwork(      IE::InferencePlugin& plugin,
                                         const IE::CNNNetwork&      net,
                                         const cv::gapi::ie::detail::ParamDesc&) {
    return plugin.LoadNetwork(net, {}); // FIXME: 2nd parameter to be
                                        // configurable via the API
}
#else // >= 2019.R2
IE::Core getCore();
IE::Core getPlugin(const cv::gapi::ie::detail::ParamDesc& params);
inline IE::ExecutableNetwork loadNetwork(      IE::Core&                        core,
                                         const IE::CNNNetwork&                  net,
                                         const cv::gapi::ie::detail::ParamDesc& params);

IE::Core getCore() {
    static IE::Core core;
    return core;
}

IE::Core getPlugin(const cv::gapi::ie::detail::ParamDesc& params) {
    auto plugin = getCore();
    if (params.device_id == "CPU" || params.device_id == "FPGA")
    {
        for (auto &&extlib : getExtensions(params))
        {
            try
            {
                plugin.AddExtension(IE::make_so_pointer<IE::IExtension>(extlib), params.device_id);
                CV_LOG_INFO(NULL, "DNN-IE: Loaded extension plugin: " << extlib);
                break;
            }
            catch(...)
            {
                CV_LOG_INFO(NULL, "Failed to load IE extension: " << extlib);
            }
        }
    }
    return plugin;
}

inline IE::ExecutableNetwork loadNetwork(      IE::Core&                        core,
                                         const IE::CNNNetwork&                  net,
                                         const cv::gapi::ie::detail::ParamDesc& params) {
    return core.LoadNetwork(net, params.device_id);
}
#endif // INF_ENGINE_RELEASE < 2019020000

IE::CNNNetwork readNetwork(const cv::gapi::ie::detail::ParamDesc& params);

#if INF_ENGINE_RELEASE < 2020000000  // < 2020.1
IE::CNNNetwork readNetwork(const cv::gapi::ie::detail::ParamDesc& params) {
    IE::CNNNetReader reader;
    reader.ReadNetwork(params.model_path);
    reader.ReadWeights(params.weights_path);
    return reader.getNetwork();
}
#else // >= 2020.1
IE::CNNNetwork readNetwork(const cv::gapi::ie::detail::ParamDesc& params) {
    auto core = getCore();
    return core.ReadNetwork(params.model_path, params.weights_path);
}
#endif // INF_ENGINE_RELEASE < 2020000000
}}}}

#endif //HAVE_INF_ENGINE
#endif // OPENCV_GAPI_IEAPI_HPP
