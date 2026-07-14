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

using namespace cv::dnn_backend;

#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)

using namespace cv::plugin::impl;  // plugin_loader.hpp

class PluginDNNBackend CV_FINAL: public std::enable_shared_from_this<PluginDNNBackend>
{
protected:
    void initPluginAPI()
    {
        const char* init_name = "opencv_dnn_plugin_init_v0";
        FN_opencv_dnn_plugin_init_t fn_init = reinterpret_cast<FN_opencv_dnn_plugin_init_t>(lib_->getSymbol(init_name));
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
                CV_LOG_INFO(NULL, "DNN: plugin is incompatible (can't be initialized): " << lib_->getName());
                return;
            }
            // NB: force strict minor version check (ABI is not preserved for now)
            if (!checkCompatibility(plugin_api_->api_header, ABI_VERSION, API_VERSION, true))
            {
                plugin_api_ = NULL;
                return;
            }
            CV_LOG_INFO(NULL, "DNN: plugin is ready to use '" << plugin_api_->api_header.api_description << "'");
        }
        else
        {
            CV_LOG_INFO(NULL, "DNN: plugin is incompatible, missing init function: '" << init_name << "', file: " << lib_->getName());
        }
    }


    bool checkCompatibility(const OpenCV_API_Header& api_header, unsigned int abi_version, unsigned int api_version, bool checkMinorOpenCVVersion)
    {
        if (api_header.opencv_version_major != CV_VERSION_MAJOR)
        {
            CV_LOG_ERROR(NULL, "DNN: wrong OpenCV major version used by plugin '" << api_header.api_description << "': " <<
                cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", api_header.opencv_version_major, api_header.opencv_version_minor))
            return false;
        }
        if (!checkMinorOpenCVVersion)
        {
            // no checks for OpenCV minor version
        }
        else if (api_header.opencv_version_minor != CV_VERSION_MINOR)
        {
            CV_LOG_ERROR(NULL, "DNN: wrong OpenCV minor version used by plugin '" << api_header.api_description << "': " <<
                cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", api_header.opencv_version_major, api_header.opencv_version_minor))
            return false;
        }
        CV_LOG_DEBUG(NULL, "DNN: initialized '" << api_header.api_description << "': built with "
            << cv::format("OpenCV %d.%d (ABI/API = %d/%d)",
                 api_header.opencv_version_major, api_header.opencv_version_minor,
                 api_header.min_api_version, api_header.api_version)
            << ", current OpenCV version is '" CV_VERSION "' (ABI/API = " << abi_version << "/" << api_version << ")"
        );
        if (api_header.min_api_version != abi_version)  // future: range can be here
        {
            // actually this should never happen due to checks in plugin's init() function
            CV_LOG_ERROR(NULL, "DNN: plugin is not supported due to incompatible ABI = " << api_header.min_api_version);
            return false;
        }
        if (api_header.api_version != api_version)
        {
            CV_LOG_INFO(NULL, "DNN: NOTE: plugin is supported, but there is API version mismath: "
                << cv::format("plugin API level (%d) != OpenCV API level (%d)", api_header.api_version, api_version));
            if (api_header.api_version < api_version)
            {
                CV_LOG_INFO(NULL, "DNN: NOTE: some functionality may be unavailable due to lack of support by plugin implementation");
            }
        }
        return true;
    }

public:
    std::shared_ptr<cv::plugin::impl::DynamicLib> lib_;
    const OpenCV_DNN_Plugin_API* plugin_api_;

    PluginDNNBackend(const std::shared_ptr<cv::plugin::impl::DynamicLib>& lib)
        : lib_(lib)
        , plugin_api_(NULL)
    {
        initPluginAPI();
    }

    std::shared_ptr<cv::dnn_backend::NetworkBackend> createNetworkBackend() const
    {
        CV_Assert(plugin_api_);

        CvPluginDNNNetworkBackend instancePtr = NULL;

        if (plugin_api_->v0.getInstance)
        {
            if (CV_ERROR_OK == plugin_api_->v0.getInstance(&instancePtr))
            {
                CV_Assert(instancePtr);
                // TODO C++20 "aliasing constructor"
                return std::shared_ptr<cv::dnn_backend::NetworkBackend>(instancePtr, [](cv::dnn_backend::NetworkBackend*){});  // empty deleter
            }
        }
        return std::shared_ptr<cv::dnn_backend::NetworkBackend>();
    }

};  // class PluginDNNBackend


class PluginDNNBackendFactory CV_FINAL: public IDNNBackendFactory
{
public:
    std::string baseName_;
    std::shared_ptr<PluginDNNBackend> backend;
    bool initialized;
public:
    PluginDNNBackendFactory(const std::string& baseName)
        : baseName_(baseName)
        , initialized(false)
    {
        // nothing, plugins are loaded on demand
    }

    std::shared_ptr<cv::dnn_backend::NetworkBackend> createNetworkBackend() const CV_OVERRIDE
    {
        if (!initialized)
        {
            const_cast<PluginDNNBackendFactory*>(this)->initBackend();
        }
        if (backend)
            return backend->createNetworkBackend();
        return std::shared_ptr<cv::dnn_backend::NetworkBackend>();
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
            CV_LOG_INFO(NULL, "DNN: exception during plugin loading: " << baseName_ << ". SKIP");
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
    const std::vector<std::string> paths_ = getConfigurationParameterPaths("OPENCV_DNN_PLUGIN_PATH", std::vector<std::string>());
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
#ifndef CV_DNN_PLUGIN_SUBDIRECTORY
            paths.push_back(binaryLocation);
#else
            paths.push_back(binaryLocation + toFileSystemPath("/") + toFileSystemPath(CV_DNN_PLUGIN_SUBDIRECTORY_STR));
#endif
        }
    }
    const std::string default_expr = libraryPrefix() + "opencv_dnn_" + baseName_l + "*" + librarySuffix();
    const std::string plugin_expr = getConfigurationParameterString((std::string("OPENCV_DNN_PLUGIN_") + baseName_u).c_str(), default_expr.c_str());
    std::vector<FileSystemPath_t> results;
#ifdef _WIN32
    FileSystemPath_t moduleName = toFileSystemPath(libraryPrefix() + "opencv_dnn_" + baseName_l + librarySuffix());
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
    CV_LOG_DEBUG(NULL, "DNN: " << baseName << " plugin's glob is '" << plugin_expr << "', " << paths.size() << " location(s)");
    for (const std::string& path : paths)
    {
        if (path.empty())
            continue;
        std::vector<std::string> candidates;
        cv::glob(utils::fs::join(path, plugin_expr), candidates);
        // Prefer candidates with higher versions
        // TODO: implemented accurate versions-based comparator
        std::sort(candidates.begin(), candidates.end(), std::greater<std::string>());
        CV_LOG_DEBUG(NULL, "    - " << path << ": " << candidates.size());
        copy(candidates.begin(), candidates.end(), back_inserter(results));
    }
#endif
    CV_LOG_DEBUG(NULL, "Found " << results.size() << " plugin(s) for " << baseName);
    return results;
}

void PluginDNNBackendFactory::loadPlugin()
{
    for (const FileSystemPath_t& plugin : getPluginCandidates(baseName_))
    {
        auto lib = std::make_shared<cv::plugin::impl::DynamicLib>(plugin);
        if (!lib->isLoaded())
        {
            continue;
        }
        try
        {
            auto pluginBackend = std::make_shared<PluginDNNBackend>(lib);
            if (!pluginBackend)
            {
                continue;
            }
            if (pluginBackend->plugin_api_ == NULL)
            {
                CV_LOG_ERROR(NULL, "DNN: no compatible plugin API for backend: " << baseName_ << " in " << toPrintablePath(plugin));
                continue;
            }
            // NB: we are going to use backend, so prevent automatic library unloading
            lib->disableAutomaticLibraryUnloading();
            backend = pluginBackend;
            return;
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "DNN: exception during plugin initialization: " << toPrintablePath(plugin) << ". SKIP");
        }
    }
}

#endif  // OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)

}  // namespace



namespace dnn_backend {


std::shared_ptr<IDNNBackendFactory> createPluginDNNBackendFactory(const std::string& baseName)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)
    const std::string baseName_u = toUpperCase(baseName);
    AutoLock lock(getInitializationMutex());
    static std::map<std::string, std::shared_ptr<IDNNBackendFactory>> g_plugins_cache;
    auto it = g_plugins_cache.find(baseName_u);
    if (it == g_plugins_cache.end())
    {
        auto factory = std::make_shared<impl::PluginDNNBackendFactory>(baseName);
        g_plugins_cache.insert(std::pair<std::string, std::shared_ptr<IDNNBackendFactory>>(baseName_u, factory));
        return factory;
    }
    return it->second;
#else
    CV_UNUSED(baseName);
    return std::shared_ptr<IDNNBackendFactory>();
#endif
}


cv::dnn_backend::NetworkBackend& createPluginDNNNetworkBackend(const std::string& baseName)
{
    auto factory = dnn_backend::createPluginDNNBackendFactory(baseName);
    if (!factory)
    {
        CV_Error(Error::StsNotImplemented, cv::format("Plugin factory is not available: '%s'", baseName.c_str()));
    }
    auto backend = factory->createNetworkBackend();
    if (!backend)
    {
        CV_Error(Error::StsNotImplemented, cv::format("Backend (plugin) is not available: '%s'", baseName.c_str()));
    }
    return *backend;
}


}}  // namespace
