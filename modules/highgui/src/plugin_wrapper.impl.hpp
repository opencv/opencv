// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//
// Not a standalone header, part of backend.cpp
//

//==================================================================================================
// Dynamic backend implementation

#include "opencv2/core/utils/plugin_loader.private.hpp"

namespace cv { namespace impl {

using namespace cv::highgui_backend;

#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)

using namespace cv::plugin::impl;  // plugin_loader.hpp

class PluginUIBackend CV_FINAL: public std::enable_shared_from_this<PluginUIBackend>
{
protected:
    void initPluginAPI()
    {
        const char* init_name = "opencv_ui_plugin_init_v0";
        FN_opencv_ui_plugin_init_t fn_init = reinterpret_cast<FN_opencv_ui_plugin_init_t>(lib_->getSymbol(init_name));
        if (fn_init)
        {
            CV_LOG_DEBUG(NULL, "Found entry: '" << init_name << "'");
            for (int supported_api_version = API_VERSION; supported_api_version >= 0; supported_api_version--)
            {
                plugin_api_ = fn_init(ABI_VERSION, supported_api_version, NULL);
                if (plugin_api_)
                    break;
            }
            if (!plugin_api_)
            {
                CV_LOG_INFO(NULL, "UI: plugin is incompatible (can't be initialized): " << lib_->getName());
                return;
            }
            // NB: force strict minor version check (ABI is not preserved for now)
            if (!checkCompatibility(plugin_api_->api_header, ABI_VERSION, API_VERSION, true))
            {
                plugin_api_ = NULL;
                return;
            }
            CV_LOG_INFO(NULL, "UI: plugin is ready to use '" << plugin_api_->api_header.api_description << "'");
        }
        else
        {
            CV_LOG_INFO(NULL, "UI: plugin is incompatible, missing init function: '" << init_name << "', file: " << lib_->getName());
        }
    }


    bool checkCompatibility(const OpenCV_API_Header& api_header, unsigned int abi_version, unsigned int api_version, bool checkMinorOpenCVVersion)
    {
        if (api_header.opencv_version_major != CV_VERSION_MAJOR)
        {
            CV_LOG_ERROR(NULL, "UI: wrong OpenCV major version used by plugin '" << api_header.api_description << "': " <<
                cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", api_header.opencv_version_major, api_header.opencv_version_minor))
            return false;
        }
        if (!checkMinorOpenCVVersion)
        {
            // no checks for OpenCV minor version
        }
        else if (api_header.opencv_version_minor != CV_VERSION_MINOR)
        {
            CV_LOG_ERROR(NULL, "UI: wrong OpenCV minor version used by plugin '" << api_header.api_description << "': " <<
                cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", api_header.opencv_version_major, api_header.opencv_version_minor))
            return false;
        }
        CV_LOG_DEBUG(NULL, "UI: initialized '" << api_header.api_description << "': built with "
            << cv::format("OpenCV %d.%d (ABI/API = %d/%d)",
                 api_header.opencv_version_major, api_header.opencv_version_minor,
                 api_header.min_api_version, api_header.api_version)
            << ", current OpenCV version is '" CV_VERSION "' (ABI/API = " << abi_version << "/" << api_version << ")"
        );
        if (api_header.min_api_version != abi_version)  // future: range can be here
        {
            // actually this should never happen due to checks in plugin's init() function
            CV_LOG_ERROR(NULL, "UI: plugin is not supported due to incompatible ABI = " << api_header.min_api_version);
            return false;
        }
        if (api_header.api_version != api_version)
        {
            CV_LOG_INFO(NULL, "UI: NOTE: plugin is supported, but there is API version mismath: "
                << cv::format("plugin API level (%d) != OpenCV API level (%d)", api_header.api_version, api_version));
            if (api_header.api_version < api_version)
            {
                CV_LOG_INFO(NULL, "UI: NOTE: some functionality may be unavailable due to lack of support by plugin implementation");
            }
        }
        return true;
    }

public:
    std::shared_ptr<cv::plugin::impl::DynamicLib> lib_;
    const OpenCV_UI_Plugin_API* plugin_api_;

    PluginUIBackend(const std::shared_ptr<cv::plugin::impl::DynamicLib>& lib)
        : lib_(lib)
        , plugin_api_(NULL)
    {
        initPluginAPI();
    }

    std::shared_ptr<cv::highgui_backend::UIBackend> create() const
    {
        CV_Assert(plugin_api_);

        CvPluginUIBackend instancePtr = NULL;

        if (plugin_api_->v0.getInstance)
        {
            if (CV_ERROR_OK == plugin_api_->v0.getInstance(&instancePtr))
            {
                CV_Assert(instancePtr);
                // TODO C++20 "aliasing constructor"
                return std::shared_ptr<cv::highgui_backend::UIBackend>(instancePtr, [](cv::highgui_backend::UIBackend*){});  // empty deleter
            }
        }
        return std::shared_ptr<cv::highgui_backend::UIBackend>();
    }
};


class PluginUIBackendFactory CV_FINAL: public IUIBackendFactory
{
public:
    std::string baseName_;
    std::shared_ptr<PluginUIBackend> backend;
    bool initialized;
public:
    PluginUIBackendFactory(const std::string& baseName)
        : baseName_(baseName)
        , initialized(false)
    {
        // nothing, plugins are loaded on demand
    }

    std::shared_ptr<cv::highgui_backend::UIBackend> create() const CV_OVERRIDE
    {
        if (!initialized)
        {
            const_cast<PluginUIBackendFactory*>(this)->initBackend();
        }
        if (backend)
            return backend->create();
        return std::shared_ptr<cv::highgui_backend::UIBackend>();
    }
protected:
    void initBackend()
    {
        AutoLock lock(getInitializationMutex());
        try
        {
            if (!initialized)
                loadPlugin();
        }
        catch (...)
        {
            CV_LOG_INFO(NULL, "UI: exception during plugin loading: " << baseName_ << ". SKIP");
        }
        initialized = true;
    }
    void loadPlugin();
};

static
std::vector<FileSystemPath_t> getPluginCandidates(const std::string& baseName)
{
    using namespace cv::utils;
    using namespace cv::utils::fs;
    const std::string baseName_l = toLowerCase(baseName);
    const std::string baseName_u = toUpperCase(baseName);
    const FileSystemPath_t baseName_l_fs = toFileSystemPath(baseName_l);
    std::vector<FileSystemPath_t> paths;
    // TODO OPENCV_PLUGIN_PATH
    const std::vector<std::string> paths_ = getConfigurationParameterPaths("OPENCV_CORE_PLUGIN_PATH", std::vector<std::string>());
    if (paths_.size() != 0)
    {
        for (size_t i = 0; i < paths_.size(); i++)
        {
            paths.push_back(toFileSystemPath(paths_[i]));
        }
    }
    else
    {
        FileSystemPath_t binaryLocation;
        if (getBinLocation(binaryLocation))
        {
            binaryLocation = getParent(binaryLocation);
#ifndef CV_UI_PLUGIN_SUBDIRECTORY
            paths.push_back(binaryLocation);
#else
            paths.push_back(binaryLocation + toFileSystemPath("/") + toFileSystemPath(CV_UI_PLUGIN_SUBDIRECTORY_STR));
#endif
        }
    }
    const std::string default_expr = libraryPrefix() + "opencv_highgui_" + baseName_l + "*" + librarySuffix();
    const std::string plugin_expr = getConfigurationParameterString((std::string("OPENCV_UI_PLUGIN_") + baseName_u).c_str(), default_expr.c_str());
    std::vector<FileSystemPath_t> results;
#ifdef _WIN32
    FileSystemPath_t moduleName = toFileSystemPath(libraryPrefix() + "opencv_highgui_" + baseName_l + librarySuffix());
    if (plugin_expr != default_expr)
    {
        moduleName = toFileSystemPath(plugin_expr);
        results.push_back(moduleName);
    }
    for (const FileSystemPath_t& path : paths)
    {
        results.push_back(path + L"\\" + moduleName);
    }
    results.push_back(moduleName);
#else
    CV_LOG_DEBUG(NULL, "UI: " << baseName << " plugin's glob is '" << plugin_expr << "', " << paths.size() << " location(s)");
    for (const std::string& path : paths)
    {
        if (path.empty())
            continue;
        std::vector<std::string> candidates;
        cv::glob(utils::fs::join(path, plugin_expr), candidates);
        CV_LOG_DEBUG(NULL, "    - " << path << ": " << candidates.size());
        copy(candidates.begin(), candidates.end(), back_inserter(results));
    }
#endif
    CV_LOG_DEBUG(NULL, "Found " << results.size() << " plugin(s) for " << baseName);
    return results;
}

// NB: require loading of imgcodecs module
static void* g_imwrite = (void*)imwrite;

void PluginUIBackendFactory::loadPlugin()
{
    CV_Assert(g_imwrite);
    for (const FileSystemPath_t& plugin : getPluginCandidates(baseName_))
    {
        auto lib = std::make_shared<cv::plugin::impl::DynamicLib>(plugin);
        if (!lib->isLoaded())
        {
            continue;
        }
        try
        {
            auto pluginBackend = std::make_shared<PluginUIBackend>(lib);
            if (!pluginBackend)
            {
                continue;
            }
            if (pluginBackend->plugin_api_ == NULL)
            {
                CV_LOG_ERROR(NULL, "UI: no compatible plugin API for backend: " << baseName_ << " in " << toPrintablePath(plugin));
                continue;
            }
            // NB: we are going to use UI backend, so prevent automatic library unloading
            lib->disableAutomaticLibraryUnloading();
            backend = pluginBackend;
            return;
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "UI: exception during plugin initialization: " << toPrintablePath(plugin) << ". SKIP");
        }
    }
}

#endif  // OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)

}  // namespace

namespace highgui_backend {

std::shared_ptr<IUIBackendFactory> createPluginUIBackendFactory(const std::string& baseName)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)
    return std::make_shared<impl::PluginUIBackendFactory>(baseName);
#else
    CV_UNUSED(baseName);
    return std::shared_ptr<IUIBackendFactory>();
#endif
}

}}  // namespace
