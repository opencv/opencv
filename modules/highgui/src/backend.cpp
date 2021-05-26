// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "backend.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#ifdef NDEBUG
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#else
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#endif
#include <opencv2/core/utils/logger.hpp>


#include "registry.hpp"
#include "registry.impl.hpp"

#include "plugin_api.hpp"
#include "plugin_wrapper.impl.hpp"


namespace cv { namespace highgui_backend {

UIBackend::~UIBackend()
{
    // nothing
}

UIWindowBase::~UIWindowBase()
{
    // nothing
}

UIWindow::~UIWindow()
{
    // nothing
}

UITrackbar::~UITrackbar()
{
    // nothing
}

static
std::string& getUIBackendName()
{
    static std::string g_backendName = toUpperCase(cv::utils::getConfigurationParameterString("OPENCV_UI_BACKEND", ""));
    return g_backendName;
}

static bool g_initializedUIBackend = false;

static
std::shared_ptr<UIBackend> createUIBackend()
{
    const std::string& name = getUIBackendName();
    bool isKnown = false;
    const auto& backends = getBackendsInfo();
    if (!name.empty())
    {
        CV_LOG_INFO(NULL, "UI: requested backend name: " << name);
    }
    for (size_t i = 0; i < backends.size(); i++)
    {
        const auto& info = backends[i];
        if (!name.empty())
        {
            if (name != info.name)
            {
                continue;
            }
            isKnown = true;
        }
        try
        {
            CV_LOG_DEBUG(NULL, "UI: trying backend: " << info.name << " (priority=" << info.priority << ")");
            if (!info.backendFactory)
            {
                CV_LOG_DEBUG(NULL, "UI: factory is not available (plugins require filesystem support): " << info.name);
                continue;
            }
            std::shared_ptr<UIBackend> backend = info.backendFactory->create();
            if (!backend)
            {
                CV_LOG_VERBOSE(NULL, 0, "UI: not available: " << info.name);
                continue;
            }
            CV_LOG_INFO(NULL, "UI: using backend: " << info.name << " (priority=" << info.priority << ")");
            g_initializedUIBackend = true;
            getUIBackendName() = info.name;
            return backend;
        }
        catch (const std::exception& e)
        {
            CV_LOG_WARNING(NULL, "UI: can't initialize " << info.name << " backend: " << e.what());
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "UI: can't initialize " << info.name << " backend: Unknown C++ exception");
        }
    }
    if (name.empty())
    {
        CV_LOG_DEBUG(NULL, "UI: fallback on builtin code");
    }
    else
    {
        if (!isKnown)
            CV_LOG_INFO(NULL, "UI: unknown backend: " << name);
    }
    g_initializedUIBackend = true;
    return std::shared_ptr<UIBackend>();
}

static inline
std::shared_ptr<UIBackend> createDefaultUIBackend()
{
    CV_LOG_DEBUG(NULL, "UI: Initializing backend...");
    return createUIBackend();
}

std::shared_ptr<UIBackend>& getCurrentUIBackend()
{
    static std::shared_ptr<UIBackend> g_currentUIBackend = createDefaultUIBackend();
    return g_currentUIBackend;
}

void setUIBackend(const std::shared_ptr<UIBackend>& api)
{
    getCurrentUIBackend() = api;
}

bool setUIBackend(const std::string& backendName)
{
    CV_TRACE_FUNCTION();

    std::string backendName_u = toUpperCase(backendName);
    if (g_initializedUIBackend)
    {
        // ... already initialized
        if (getUIBackendName() == backendName_u)
        {
            CV_LOG_INFO(NULL, "UI: backend is already activated: " << (backendName.empty() ? "builtin(legacy)" : backendName));
            return true;
        }
        else
        {
            // ... re-create new
            CV_LOG_DEBUG(NULL, "UI: replacing backend...");
            getUIBackendName() = backendName_u;
            getCurrentUIBackend() = createUIBackend();
        }
    }
    else
    {
        // ... no backend exists, just specify the name (initialization is triggered by getCurrentUIBackend() call)
        getUIBackendName() = backendName_u;
    }
    std::shared_ptr<UIBackend> api = getCurrentUIBackend();
    if (!api)
    {
        if (!backendName.empty())
        {
            CV_LOG_WARNING(NULL, "UI: backend is not available: " << backendName << " (using builtin legacy code)");
            return false;
        }
        else
        {
            CV_LOG_WARNING(NULL, "UI: switched to builtin code (legacy)");
        }
    }
    if (!backendName_u.empty())
    {
        CV_Assert(backendName_u == getUIBackendName());  // data race?
    }
    return true;
}

}}  // namespace cv::highgui_backend
