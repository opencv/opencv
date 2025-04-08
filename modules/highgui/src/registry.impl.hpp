// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//
// Not a standalone header, part of backend.cpp
//

#include "opencv2/core/utils/filesystem.private.hpp"  // OPENCV_HAVE_FILESYSTEM_SUPPORT

namespace cv { namespace highgui_backend {

#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)
#define DECLARE_DYNAMIC_BACKEND(name) \
BackendInfo { \
    1000, name, createPluginUIBackendFactory(name) \
},
#else
#define DECLARE_DYNAMIC_BACKEND(name) /* nothing */
#endif

#define DECLARE_STATIC_BACKEND(name, createBackendAPI) \
BackendInfo { \
    1000, name, std::make_shared<cv::highgui_backend::StaticBackendFactory>([=] () -> std::shared_ptr<cv::highgui_backend::UIBackend> { return createBackendAPI(); }) \
},

static
std::vector<BackendInfo>& getBuiltinBackendsInfo()
{
    static std::vector<BackendInfo> g_backends
    {
#ifdef HAVE_GTK
        DECLARE_STATIC_BACKEND("GTK", createUIBackendGTK)
#if defined(HAVE_GTK3)
        DECLARE_STATIC_BACKEND("GTK3", createUIBackendGTK)
#elif defined(HAVE_GTK2)
        DECLARE_STATIC_BACKEND("GTK2", createUIBackendGTK)
#else
#warning "HAVE_GTK definition issue. Register new GTK backend"
#endif
#elif defined(ENABLE_PLUGINS)
        DECLARE_DYNAMIC_BACKEND("GTK")
        DECLARE_DYNAMIC_BACKEND("GTK3")
        DECLARE_DYNAMIC_BACKEND("GTK2")
#endif

#ifdef HAVE_FRAMEBUFFER
        DECLARE_STATIC_BACKEND("FB", createUIBackendFramebuffer)
#endif

#if 0  // TODO
#ifdef HAVE_QT
        DECLARE_STATIC_BACKEND("QT", createUIBackendQT)
#elif defined(ENABLE_PLUGINS)
        DECLARE_DYNAMIC_BACKEND("QT")
#endif
#endif

#ifdef _WIN32
#ifdef HAVE_WIN32UI
        DECLARE_STATIC_BACKEND("WIN32", createUIBackendWin32UI)
#elif defined(ENABLE_PLUGINS)
        DECLARE_DYNAMIC_BACKEND("WIN32")
#endif
#endif
    };
    return g_backends;
}

static
bool sortByPriority(const BackendInfo &lhs, const BackendInfo &rhs)
{
    return lhs.priority > rhs.priority;
}

/** @brief Manages list of enabled backends
 */
class UIBackendRegistry
{
protected:
    std::vector<BackendInfo> enabledBackends;
    UIBackendRegistry()
    {
        enabledBackends = getBuiltinBackendsInfo();
        int N = (int)enabledBackends.size();
        for (int i = 0; i < N; i++)
        {
            BackendInfo& info = enabledBackends[i];
            info.priority = 1000 - i * 10;
        }
        CV_LOG_DEBUG(NULL, "UI: Builtin backends(" << N << "): " << dumpBackends());
        if (readPrioritySettings())
        {
            CV_LOG_INFO(NULL, "UI: Updated backends priorities: " << dumpBackends());
            N = (int)enabledBackends.size();
        }
        int enabled = 0;
        for (int i = 0; i < N; i++)
        {
            BackendInfo& info = enabledBackends[enabled];
            if (enabled != i)
                info = enabledBackends[i];
            size_t param_priority = utils::getConfigurationParameterSizeT(cv::format("OPENCV_UI_PRIORITY_%s", info.name.c_str()).c_str(), (size_t)info.priority);
            CV_Assert(param_priority == (size_t)(int)param_priority); // overflow check
            if (param_priority > 0)
            {
                info.priority = (int)param_priority;
                enabled++;
            }
            else
            {
                CV_LOG_INFO(NULL, "UI: Disable backend: " << info.name);
            }
        }
        enabledBackends.resize(enabled);
        CV_LOG_DEBUG(NULL, "UI: Available backends(" << enabled << "): " << dumpBackends());
        std::sort(enabledBackends.begin(), enabledBackends.end(), sortByPriority);
        CV_LOG_INFO(NULL, "UI: Enabled backends(" << enabled << ", sorted by priority): " << (enabledBackends.empty() ? std::string("N/A") : dumpBackends()));
    }

    static std::vector<std::string> tokenize_string(const std::string& input, char token)
    {
        std::vector<std::string> result;
        std::string::size_type prev_pos = 0, pos = 0;
        while((pos = input.find(token, pos)) != std::string::npos)
        {
            result.push_back(input.substr(prev_pos, pos-prev_pos));
            prev_pos = ++pos;
        }
        result.push_back(input.substr(prev_pos));
        return result;
    }
    bool readPrioritySettings()
    {
        bool hasChanges = false;
        cv::String prioritized_backends = utils::getConfigurationParameterString("OPENCV_UI_PRIORITY_LIST");
        if (prioritized_backends.empty())
            return hasChanges;
        CV_LOG_INFO(NULL, "UI: Configured priority list (OPENCV_UI_PRIORITY_LIST): " << prioritized_backends);
        const std::vector<std::string> names = tokenize_string(prioritized_backends, ',');
        for (size_t i = 0; i < names.size(); i++)
        {
            const std::string& name = names[i];
            int priority = (int)(100000 + (names.size() - i) * 1000);
            bool found = false;
            for (size_t k = 0; k < enabledBackends.size(); k++)
            {
                BackendInfo& info = enabledBackends[k];
                if (name == info.name)
                {
                    info.priority = priority;
                    CV_LOG_DEBUG(NULL, "UI: New backend priority: '" << name << "' => " << info.priority);
                    found = true;
                    hasChanges = true;
                    break;
                }
            }
            if (!found)
            {
                CV_LOG_INFO(NULL, "UI: Adding backend (plugin): '" << name << "'");
                enabledBackends.push_back(BackendInfo{priority, name, createPluginUIBackendFactory(name)});
                hasChanges = true;
            }
        }
        return hasChanges;
    }
public:
    std::string dumpBackends() const
    {
        std::ostringstream os;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            if (i > 0) os << "; ";
            const BackendInfo& info = enabledBackends[i];
            os << info.name << '(' << info.priority << ')';
        }
#if !defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND)
        os << " + BUILTIN(" OPENCV_HIGHGUI_BUILTIN_BACKEND_STR ")";
#endif
        return os.str();
    }

    static UIBackendRegistry& getInstance()
    {
        static UIBackendRegistry g_instance;
        return g_instance;
    }

    inline const std::vector<BackendInfo>& getEnabledBackends() const { return enabledBackends; }
};


const std::vector<BackendInfo>& getBackendsInfo()
{
    return cv::highgui_backend::UIBackendRegistry::getInstance().getEnabledBackends();
}

}} // namespace
