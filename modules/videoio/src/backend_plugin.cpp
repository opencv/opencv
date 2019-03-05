// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "backend.hpp"
#include "plugin_api.hpp"

#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/private.hpp"
#include "videoio_registry.hpp"

//==================================================================================================
// Dynamic backend implementation

#include "opencv2/core/utils/logger.hpp"
#include <sstream>
using namespace std;

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace cv { namespace impl {

#if defined(_WIN32)
typedef HMODULE LibHandle_t;
#elif defined(__linux__) || defined(__APPLE__)
typedef void* LibHandle_t;
#endif

static Mutex& getInitializationMutex()
{
    static Mutex initializationMutex;
    return initializationMutex;
}

static inline
void* getSymbol_(LibHandle_t h, const char* symbolName)
{
#if defined(_WIN32)
    return (void*)GetProcAddress(h, symbolName);
#elif defined(__linux__) || defined(__APPLE__)
    return dlsym(h, symbolName);
#endif
}

static inline
LibHandle_t libraryLoad_(const char* filename)
{
#if defined(_WIN32)
    return LoadLibraryA(filename);
#elif defined(__linux__) || defined(__APPLE__)
    return dlopen(filename, RTLD_LAZY);
#endif
}

static inline
void libraryRelease_(LibHandle_t h)
{
#if defined(_WIN32)
    FreeLibrary(h);
#elif defined(__linux__) || defined(__APPLE__)
    dlclose(h);
#endif
}

static inline
std::string libraryPrefix()
{
#if defined(_WIN32)
    return string();
#else
    return "lib";
#endif
}
static inline
std::string librarySuffix()
{
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

//============================

class DynamicLib
{
private:
    LibHandle_t handle;
    const std::string fname;

public:
    DynamicLib(const std::string &filename)
        : handle(0), fname(filename)
    {
        libraryLoad(filename);
    }
    ~DynamicLib()
    {
        libraryRelease();
    }
    bool isLoaded() const
    {
        return handle != NULL;
    }
    void* getSymbol(const char* symbolName) const
    {
        if (!handle)
        {
            return 0;
        }
        void * res = getSymbol_(handle, symbolName);
        if (!res)
            CV_LOG_ERROR(NULL, "No symbol '" << symbolName << "' in " << fname);
        return res;
    }
    const std::string& getName() const { return fname; }
private:
    void libraryLoad(const std::string &filename)
    {
        handle = libraryLoad_(filename.c_str());
        CV_LOG_INFO(NULL, "load " << filename << " => " << (handle ? "OK" : "FAILED"));
    }
    void libraryRelease()
    {
        CV_LOG_INFO(NULL, "unload "<< fname);
        if (handle)
        {
            libraryRelease_(handle);
            handle = 0;
        }
    }

private:
    DynamicLib(const DynamicLib &);
    DynamicLib &operator=(const DynamicLib &);
};


//============================

class PluginBackend: public IBackend
{
public:
    Ptr<DynamicLib> lib_;
    const OpenCV_VideoIO_Plugin_API_preview* plugin_api_;

    PluginBackend(const Ptr<DynamicLib>& lib) :
        lib_(lib), plugin_api_(NULL)
    {
        const char* init_name = "opencv_videoio_plugin_init_v0";
        FN_opencv_videoio_plugin_init_t fn_init = reinterpret_cast<FN_opencv_videoio_plugin_init_t>(lib_->getSymbol(init_name));
        if (fn_init)
        {
            plugin_api_ = fn_init(ABI_VERSION, API_VERSION, NULL);
            if (!plugin_api_)
            {
                CV_LOG_INFO(NULL, "Video I/O: plugin is incompatible: " << lib->getName());
                return;
            }
            if (plugin_api_->api_header.opencv_version_major != CV_VERSION_MAJOR ||
                plugin_api_->api_header.opencv_version_minor != CV_VERSION_MINOR)
            {
                CV_LOG_ERROR(NULL, "Video I/O: wrong OpenCV version used by plugin '" << plugin_api_->api_header.api_description << "': " <<
                    cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", plugin_api_->api_header.opencv_version_major, plugin_api_->api_header.opencv_version_minor))
                plugin_api_ = NULL;
                return;
            }
            // TODO Preview: add compatibility API/ABI checks
            CV_LOG_INFO(NULL, "Video I/O: loaded plugin '" << plugin_api_->api_header.api_description << "'");
        }
        else
        {
            CV_LOG_INFO(NULL, "Video I/O: plugin is incompatible, missing init function: '" << init_name << "', file: " << lib->getName());
        }
    }

    Ptr<IVideoCapture> createCapture(int camera) const CV_OVERRIDE;
    Ptr<IVideoCapture> createCapture(const std::string &filename) const CV_OVERRIDE;
    Ptr<IVideoWriter>  createWriter(const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor) const CV_OVERRIDE;
};

class PluginBackendFactory : public IBackendFactory
{
public:
    VideoCaptureAPIs id_;
    const char* baseName_;
    Ptr<PluginBackend> backend;
    bool initialized;
public:
    PluginBackendFactory(VideoCaptureAPIs id, const char* baseName) :
        id_(id), baseName_(baseName),
        initialized(false)
    {
        // nothing, plugins are loaded on demand
    }

    Ptr<IBackend> getBackend() const CV_OVERRIDE
    {
        if (!initialized)
        {
            const_cast<PluginBackendFactory*>(this)->initBackend();
        }
        return backend.staticCast<IBackend>();
    }
protected:
    void initBackend()
    {
        AutoLock lock(getInitializationMutex());
        try {
            if (!initialized)
                loadPlugin();
        }
        catch (...)
        {
            CV_LOG_INFO(NULL, "Video I/O: exception during plugin loading: " << baseName_ << ". SKIP");
        }
        initialized = true;
    }
    void loadPlugin();
};

static
std::vector<string> getPluginCandidates(const std::string& baseName)
{
    using namespace cv::utils;
    using namespace cv::utils::fs;
#ifndef CV_VIDEOIO_PLUGIN_SUBDIRECTORY
#define CV_VIDEOIO_PLUGIN_SUBDIRECTORY_STR ""
#else
#define CV_VIDEOIO_PLUGIN_SUBDIRECTORY_STR CVAUX_STR(CV_VIDEOIO_PLUGIN_SUBDIRECTORY)
#endif
    const vector<string> default_paths = { utils::fs::join(getParent(getBinLocation()), CV_VIDEOIO_PLUGIN_SUBDIRECTORY_STR) };
    const vector<string> paths = getConfigurationParameterPaths("OPENCV_VIDEOIO_PLUGIN_PATH", default_paths);
    const string baseName_l = toLowerCase(baseName);
    const string baseName_u = toUpperCase(baseName);
    const string default_expr = libraryPrefix() + "opencv_videoio_" + baseName_l + "*" + librarySuffix();
    const string expr = getConfigurationParameterString((std::string("OPENCV_VIDEOIO_PLUGIN_") + baseName_u).c_str(), default_expr.c_str());
    CV_LOG_INFO(NULL, "VideoIO pluigin (" << baseName << "): glob is '" << expr << "', " << paths.size() << " location(s)");
    vector<string> results;
    for(const string & path : paths)
    {
        if (path.empty())
            continue;
        vector<string> candidates;
        cv::glob(utils::fs::join(path, expr), candidates);
        CV_LOG_INFO(NULL, "    - " << path << ": " << candidates.size());
        copy(candidates.begin(), candidates.end(), back_inserter(results));
    }
    CV_LOG_INFO(NULL, "Found " << results.size() << " plugin(s) for " << baseName);
    return results;
}

void PluginBackendFactory::loadPlugin()
{
    for(const std::string & plugin : getPluginCandidates(baseName_))
    {
        Ptr<DynamicLib> lib = makePtr<DynamicLib>(plugin);
        if (!lib->isLoaded())
            continue;
        try
        {
            Ptr<PluginBackend> pluginBackend = makePtr<PluginBackend>(lib);
            if (pluginBackend && pluginBackend->plugin_api_)
            {
                if (pluginBackend->plugin_api_->captureAPI != id_)
                {
                    CV_LOG_ERROR(NULL, "Video I/O: plugin '" << pluginBackend->plugin_api_->api_header.api_description <<
                                       "': unexpected backend ID: " <<
                                       pluginBackend->plugin_api_->captureAPI << " vs " << (int)id_ << " (expected)");
                }
                else
                {
                    backend = pluginBackend;
                    return;
                }
            }
        }
        catch (...)
        {
            CV_LOG_INFO(NULL, "Video I/O: exception during plugin initialization: " << plugin << ". SKIP");
        }
    }
}


//==================================================================================================

class PluginCapture : public cv::IVideoCapture
{
    const OpenCV_VideoIO_Plugin_API_preview* plugin_api_;
    CvPluginCapture capture_;

public:
    static
    Ptr<PluginCapture> create(const OpenCV_VideoIO_Plugin_API_preview* plugin_api,
            const std::string &filename, int camera)
    {
        CV_Assert(plugin_api);
        CvPluginCapture capture = NULL;
        if (plugin_api->Capture_open)
        {
            CV_Assert(plugin_api->Capture_release);
            if (CV_ERROR_OK == plugin_api->Capture_open(filename.empty() ? 0 : filename.c_str(), camera, &capture))
            {
                CV_Assert(capture);
                return makePtr<PluginCapture>(plugin_api, capture);
            }
        }
        return Ptr<PluginCapture>();
    }

    PluginCapture(const OpenCV_VideoIO_Plugin_API_preview* plugin_api, CvPluginCapture capture)
        : plugin_api_(plugin_api), capture_(capture)
    {
        CV_Assert(plugin_api_); CV_Assert(capture_);
    }

    ~PluginCapture()
    {
        CV_DbgAssert(plugin_api_->Capture_release);
        if (CV_ERROR_OK != plugin_api_->Capture_release(capture_))
            CV_LOG_ERROR(NULL, "Video I/O: Can't release capture by plugin '" << plugin_api_->api_header.api_description << "'");
        capture_ = NULL;
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        double val = -1;
        if (plugin_api_->Capture_getProperty)
            if (CV_ERROR_OK != plugin_api_->Capture_getProperty(capture_, prop, &val))
                val = -1;
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        if (plugin_api_->Capture_setProperty)
            if (CV_ERROR_OK == plugin_api_->Capture_setProperty(capture_, prop, val))
                return true;
        return false;
    }
    bool grabFrame() CV_OVERRIDE
    {
        if (plugin_api_->Capture_grab)
            if (CV_ERROR_OK == plugin_api_->Capture_grab(capture_))
                return true;
        return false;
    }
    static CvResult CV_API_CALL retrieve_callback(int stream_idx, const unsigned char* data, int step, int width, int height, int cn, void* userdata)
    {
        CV_UNUSED(stream_idx);
        cv::_OutputArray* dst = static_cast<cv::_OutputArray*>(userdata);
        if (!dst)
            return CV_ERROR_FAIL;
        cv::Mat(cv::Size(width, height), CV_MAKETYPE(CV_8U, cn), (void*)data, step).copyTo(*dst);
        return CV_ERROR_OK;
    }
    bool retrieveFrame(int idx, cv::OutputArray img) CV_OVERRIDE
    {
        bool res = false;
        if (plugin_api_->Capture_retreive)
            if (CV_ERROR_OK == plugin_api_->Capture_retreive(capture_, idx, retrieve_callback, (cv::_OutputArray*)&img))
                res = true;
        return res;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return capture_ != NULL;  // TODO always true
    }
    int getCaptureDomain() CV_OVERRIDE
    {
        return plugin_api_->captureAPI;
    }
};


//==================================================================================================

class PluginWriter : public cv::IVideoWriter
{
    const OpenCV_VideoIO_Plugin_API_preview* plugin_api_;
    CvPluginWriter writer_;

public:
    static
    Ptr<PluginWriter> create(const OpenCV_VideoIO_Plugin_API_preview* plugin_api,
            const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor)
    {
        CV_Assert(plugin_api);
        CvPluginWriter writer = NULL;
        if (plugin_api->Writer_open)
        {
            CV_Assert(plugin_api->Writer_release);
            if (CV_ERROR_OK == plugin_api->Writer_open(filename.empty() ? 0 : filename.c_str(), fourcc, fps, sz.width, sz.height, isColor, &writer))
            {
                CV_Assert(writer);
                return makePtr<PluginWriter>(plugin_api, writer);
            }
        }
        return Ptr<PluginWriter>();
    }

    PluginWriter(const OpenCV_VideoIO_Plugin_API_preview* plugin_api, CvPluginWriter writer)
        : plugin_api_(plugin_api), writer_(writer)
    {
        CV_Assert(plugin_api_); CV_Assert(writer_);
    }

    ~PluginWriter()
    {
        CV_DbgAssert(plugin_api_->Writer_release);
        if (CV_ERROR_OK != plugin_api_->Writer_release(writer_))
            CV_LOG_ERROR(NULL, "Video I/O: Can't release writer by plugin '" << plugin_api_->api_header.api_description << "'");
        writer_ = NULL;
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        double val = -1;
        if (plugin_api_->Writer_getProperty)
            if (CV_ERROR_OK != plugin_api_->Writer_getProperty(writer_, prop, &val))
                val = -1;
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        if (plugin_api_->Writer_setProperty)
            if (CV_ERROR_OK == plugin_api_->Writer_setProperty(writer_, prop, val))
                return true;
        return false;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return writer_ != NULL;  // TODO always true
    }
    void write(cv::InputArray arr) CV_OVERRIDE
    {
        cv::Mat img = arr.getMat();
        CV_DbgAssert(writer_);
        CV_Assert(plugin_api_->Writer_write);
        if (CV_ERROR_OK != plugin_api_->Writer_write(writer_, img.data, (int)img.step[0], img.cols, img.rows, img.channels()))
        {
            CV_LOG_DEBUG(NULL, "Video I/O: Can't write frame by plugin '" << plugin_api_->api_header.api_description << "'");
        }
        // TODO return bool result?
    }
    int getCaptureDomain() const CV_OVERRIDE
    {
        return plugin_api_->captureAPI;
    }
};


Ptr<IVideoCapture> PluginBackend::createCapture(int camera) const
{
    try
    {
        if (plugin_api_)
            return PluginCapture::create(plugin_api_, std::string(), camera); //.staticCast<IVideoCapture>();
    }
    catch (...)
    {
        CV_LOG_DEBUG(NULL, "Video I/O: can't create camera capture: " << camera);
    }
    return Ptr<IVideoCapture>();
}

Ptr<IVideoCapture> PluginBackend::createCapture(const std::string &filename) const
{
    try
    {
        if (plugin_api_)
            return PluginCapture::create(plugin_api_, filename, 0); //.staticCast<IVideoCapture>();
    }
    catch (...)
    {
        CV_LOG_DEBUG(NULL, "Video I/O: can't open file capture: " << filename);
    }
    return Ptr<IVideoCapture>();
}

Ptr<IVideoWriter> PluginBackend::createWriter(const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor) const
{
    try
    {
        if (plugin_api_)
            return PluginWriter::create(plugin_api_, filename, fourcc, fps, sz, isColor); //.staticCast<IVideoWriter>();
    }
    catch (...)
    {
        CV_LOG_DEBUG(NULL, "Video I/O: can't open writer: " << filename);
    }
    return Ptr<IVideoWriter>();
}

}  // namespace

Ptr<IBackendFactory> createPluginBackendFactory(VideoCaptureAPIs id, const char* baseName)
{
    return makePtr<impl::PluginBackendFactory>(id, baseName); //.staticCast<IBackendFactory>();
}

}  // namespace
