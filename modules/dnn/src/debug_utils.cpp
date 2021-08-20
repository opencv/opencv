// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <sstream>

#include <opencv2/dnn/layer_reg.private.hpp>
#include <opencv2/dnn/utils/debug_utils.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

bool DNN_DIAGNOSTICS_RUN = false;
bool DNN_SKIP_REAL_IMPORT = false;

void enableModelDiagnostics(bool isDiagnosticsMode)
{
    DNN_DIAGNOSTICS_RUN = isDiagnosticsMode;

    if (DNN_DIAGNOSTICS_RUN)
    {
        detail::NotImplemented::Register();
    }
    else
    {
        detail::NotImplemented::unRegister();
    }
}

void skipModelImport(bool skip)
{
    DNN_SKIP_REAL_IMPORT = skip;
}

void detail::LayerHandler::addMissing(const std::string& name, const std::string& type)
{
    cv::AutoLock lock(getLayerFactoryMutex());
    auto& registeredLayers = getLayerFactoryImpl();

    // If we didn't add it, but can create it, it's custom and not missing.
    if (layers.find(type) == layers.end() && registeredLayers.find(type) != registeredLayers.end())
    {
        return;
    }

    layers[type].insert(name);
}

bool detail::LayerHandler::contains(const std::string& type) const
{
    return layers.find(type) != layers.end();
}

void detail::LayerHandler::printMissing()
{
    if (layers.empty())
    {
        return;
    }

    std::stringstream ss;
    ss << "DNN: Not supported types:\n";
    for (const auto& type_names : layers)
    {
        const auto& type = type_names.first;
        ss << "Type='" << type << "', affected nodes:\n[";
        for (const auto& name : type_names.second)
        {
            ss << "'" << name << "', ";
        }
        ss.seekp(-2, std::ios_base::end);
        ss << "]\n";
    }
    CV_LOG_ERROR(NULL, ss.str());
}

LayerParams detail::LayerHandler::getNotImplementedParams(const std::string& name, const std::string& op)
{
    LayerParams lp;
    lp.name = name;
    lp.type = "NotImplemented";
    lp.set("type", op);

    return lp;
}

CV__DNN_INLINE_NS_END
}} // namespace
