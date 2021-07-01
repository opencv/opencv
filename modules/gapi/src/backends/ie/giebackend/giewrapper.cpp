// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifdef HAVE_INF_ENGINE

#include <vector>
#include <string>
#include <tuple>

#include "backends/ie/giebackend/giewrapper.hpp"

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace IE = InferenceEngine;
namespace giewrap = cv::gimpl::ie::wrap;
using GIEParam = cv::gapi::ie::detail::ParamDesc;

IE::InputsDataMap giewrap::toInputsDataMap (const IE::ConstInputsDataMap& inputs) {
    IE::InputsDataMap transformed;
    auto convert = [](const std::pair<std::string, IE::InputInfo::CPtr>& p) {
        return std::make_pair(p.first, std::const_pointer_cast<IE::InputInfo>(p.second));
    };
    std::transform(inputs.begin(), inputs.end(), std::inserter(transformed, transformed.end()), convert);
    return transformed;
}

IE::OutputsDataMap giewrap::toOutputsDataMap (const IE::ConstOutputsDataMap& outputs) {
    IE::OutputsDataMap transformed;
    auto convert = [](const std::pair<std::string, IE::CDataPtr>& p) {
        return std::make_pair(p.first, std::const_pointer_cast<IE::Data>(p.second));
    };
    std::transform(outputs.begin(), outputs.end(), std::inserter(transformed, transformed.end()), convert);
    return transformed;
}

#if INF_ENGINE_RELEASE < 2020000000  // < 2020.1
// Load extensions (taken from DNN module)
std::vector<std::string> giewrap::getExtensions(const GIEParam& params) {
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

IE::CNNNetwork giewrap::readNetwork(const GIEParam& params) {
    IE::CNNNetReader reader;
    reader.ReadNetwork(params.model_path);
    reader.ReadWeights(params.weights_path);
    return reader.getNetwork();
}
#else // >= 2020.1
std::vector<std::string> giewrap::getExtensions(const GIEParam&) {
    return std::vector<std::string>();
}

IE::CNNNetwork giewrap::readNetwork(const GIEParam& params) {
    auto core = giewrap::getCore();
    return core.ReadNetwork(params.model_path, params.weights_path);
}
#endif // INF_ENGINE_RELEASE < 2020000000

#if INF_ENGINE_RELEASE < 2019020000  // < 2019.R2
IE::InferencePlugin giewrap::getPlugin(const GIEParam& params) {
    auto plugin = IE::PluginDispatcher().getPluginByDevice(params.device_id);
    if (params.device_id == "CPU" || params.device_id == "FPGA")
    {
        for (auto &&extlib : giewrap::getExtensions(params))
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
#else // >= 2019.R2
IE::Core giewrap::getCore() {
    static IE::Core core;
    return core;
}

IE::Core giewrap::getPlugin(const GIEParam& params) {
    auto plugin = giewrap::getCore();
    if (params.device_id == "CPU" || params.device_id == "FPGA")
    {
        for (auto &&extlib : giewrap::getExtensions(params))
        {
            try
            {
#if INF_ENGINE_RELEASE >= 2021040000
                plugin.AddExtension(std::make_shared<IE::Extension>(extlib), params.device_id);
#else
                plugin.AddExtension(IE::make_so_pointer<IE::IExtension>(extlib), params.device_id);
#endif
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
#endif // INF_ENGINE_RELEASE < 2019020000

#endif //HAVE_INF_ENGINE
