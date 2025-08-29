// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../precomp.hpp"
#include "parallel.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#ifdef NDEBUG
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#else
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#endif
#include <opencv2/core/utils/logger.hpp>


#include "registry_parallel.hpp"
#include "registry_parallel.impl.hpp"

#include "plugin_parallel_api.hpp"
#include "plugin_parallel_wrapper.impl.hpp"


namespace cv { namespace parallel {

int numThreads = -1;

ParallelForAPI::~ParallelForAPI()
{
    // nothing
}

static
std::string& getParallelBackendName()
{
    static std::string g_backendName = toUpperCase(cv::utils::getConfigurationParameterString("OPENCV_PARALLEL_BACKEND", ""));
    return g_backendName;
}

static bool g_initializedParallelForAPI = false;

static
std::shared_ptr<ParallelForAPI> createParallelForAPI()
{
    const std::string& name = getParallelBackendName();
    bool isKnown = false;
    const auto& backends = getParallelBackendsInfo();
    if (!name.empty())
    {
        CV_LOG_INFO(NULL, "core(parallel): requested backend name: " << name);
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
            CV_LOG_DEBUG(NULL, "core(parallel): trying backend: " << info.name << " (priority=" << info.priority << ")");
            if (!info.backendFactory)
            {
                CV_LOG_DEBUG(NULL, "core(parallel): factory is not available (plugins require filesystem support): " << info.name);
                continue;
            }
            std::shared_ptr<ParallelForAPI> backend = info.backendFactory->create();
            if (!backend)
            {
                CV_LOG_VERBOSE(NULL, 0, "core(parallel): not available: " << info.name);
                continue;
            }
            CV_LOG_INFO(NULL, "core(parallel): using backend: " << info.name << " (priority=" << info.priority << ")");
            g_initializedParallelForAPI = true;
            getParallelBackendName() = info.name;
            return backend;
        }
        catch (const std::exception& e)
        {
            CV_LOG_WARNING(NULL, "core(parallel): can't initialize " << info.name << " backend: " << e.what());
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "core(parallel): can't initialize " << info.name << " backend: Unknown C++ exception");
        }
    }
    if (name.empty())
    {
        CV_LOG_DEBUG(NULL, "core(parallel): fallback on builtin code");
    }
    else
    {
        if (!isKnown)
            CV_LOG_INFO(NULL, "core(parallel): unknown backend: " << name);
    }
    // Note: when no backend is found we still mark initialized so that subsequent calls
    // don't repeatedly try to initialize on every getCurrentParallelForAPI().
    // The caller (setParallelForBackend) relies on validation and will react accordingly.
    g_initializedParallelForAPI = true;
    return std::shared_ptr<ParallelForAPI>();
}

static inline
std::shared_ptr<ParallelForAPI> createDefaultParallelForAPI()
{
    CV_LOG_DEBUG(NULL, "core(parallel): Initializing parallel backend...");
    return createParallelForAPI();
}

std::shared_ptr<ParallelForAPI>& getCurrentParallelForAPI()
{
    static std::shared_ptr<ParallelForAPI> g_currentParallelForAPI = createDefaultParallelForAPI();
    return g_currentParallelForAPI;
}

void setParallelForBackend(const std::shared_ptr<ParallelForAPI>& api, bool propagateNumThreads)
{
    getCurrentParallelForAPI() = api;
    if (propagateNumThreads && api)
    {
        setNumThreads(numThreads);
    }
}

bool setParallelForBackend(const std::string& backendName, bool propagateNumThreads)
{
    CV_TRACE_FUNCTION();

    std::string backendName_u = toUpperCase(backendName);
    // Helper: check whether backendName_u exists among registered backends
    auto isBackendKnown = [&](const std::string& name) -> bool {
        if (name.empty())
            return true; // builtin is always allowed
        const auto& backends = getParallelBackendsInfo();
        for (const auto& info : backends)
        {
            if (name == info.name)
                return true;
        }
        return false;
    };

    if (g_initializedParallelForAPI)
    {
        // ... already initialized
        if (getParallelBackendName() == backendName_u)
        {
            CV_LOG_INFO(NULL, "core(parallel): backend is already activated: " << (backendName.empty() ? "builtin(legacy)" : backendName));
            return true;
        }
        else
        {
            // If requested backend is unknown, refuse and don't overwrite current state
            if (!isBackendKnown(backendName_u))
            {
                CV_LOG_INFO(NULL, "core(parallel): unknown backend: " << backendName);
                return false;
            }

            // Try to initialize the requested backend before committing the change
            CV_LOG_DEBUG(NULL, "core(parallel): replacing parallel backend...");

            // Preserve old name in case initialization fails
            std::string oldName = getParallelBackendName();

            // Temporarily set the requested name so createParallelForAPI() will try it
            getParallelBackendName() = backendName_u;
            std::shared_ptr<ParallelForAPI> newApi = createParallelForAPI();

            if (!newApi)
            {
                // initialization failed -> restore previous name and keep previous backend
                CV_LOG_WARNING(NULL, "core(parallel): backend is not available: " << backendName << " (keeping previous backend: " << oldName << ")");
                getParallelBackendName() = oldName;
                // ensure that getCurrentParallelForAPI() remains unchanged
                return false;
            }

            // success -> commit new backend
            getCurrentParallelForAPI() = newApi;
        }
    }
    else
    {
        // ... no backend exists yet
        // If requested backend is unknown, refuse early
        if (!isBackendKnown(backendName_u))
        {
            CV_LOG_INFO(NULL, "core(parallel): unknown backend: " << backendName);
            return false;
        }

        // Just set the backend name — actual initialization will happen when getCurrentParallelForAPI() is used
        getParallelBackendName() = backendName_u;
    }

    std::shared_ptr<ParallelForAPI> api = getCurrentParallelForAPI();
    if (!api)
    {
        if (!backendName.empty())
        {
            CV_LOG_WARNING(NULL, "core(parallel): backend is not available: " << backendName << " (using builtin legacy code)");
            return false;
        }
        else
        {
            CV_LOG_WARNING(NULL, "core(parallel): switched to builtin code (legacy)");
        }
    }
    if (!backendName_u.empty())
    {
        CV_Assert(backendName_u == getParallelBackendName());  // data race?
    }

    if (propagateNumThreads)
    {
        setNumThreads(numThreads);
    }
    return true;
}

}}  // namespace
