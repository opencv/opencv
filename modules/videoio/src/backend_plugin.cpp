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
#elif defined(__linux__) || defined(__APPLE__) || defined(__OpenBSD__) || defined(__FreeBSD__) || defined(__GLIBC__)
#include <dlfcn.h>
#endif

namespace cv { namespace impl {

#if defined(_WIN32)
typedef HMODULE LibHandle_t;
typedef wchar_t FileSystemChar_t;
typedef std::wstring FileSystemPath_t;

static
FileSystemPath_t toFileSystemPath(const std::string& p)
{
    FileSystemPath_t result;
    result.resize(p.size());
    for (size_t i = 0; i < p.size(); i++)
        result[i] = (wchar_t)p[i];
    return result;
}
static
std::string toPrintablePath(const FileSystemPath_t& p)
{
    std::string result;
    result.resize(p.size());
    for (size_t i = 0; i < p.size(); i++)
    {
        wchar_t ch = p[i];
        if ((int)ch >= ' ' && (int)ch < 128)
            result[i] = (char)ch;
        else
            result[i] = '?';
    }
    return result;
}
#else  // !_WIN32
typedef void* LibHandle_t;
typedef char FileSystemChar_t;
typedef std::string FileSystemPath_t;

static inline FileSystemPath_t toFileSystemPath(const std::string& p) { return p; }
static inline std::string toPrintablePath(const FileSystemPath_t& p) { return p; }
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
#elif defined(__linux__) || defined(__APPLE__) || defined(__OpenBSD__) || defined(__FreeBSD__) || defined(__GLIBC__)
    return dlsym(h, symbolName);
#endif
}

static inline
LibHandle_t libraryLoad_(const FileSystemPath_t& filename)
{
#if defined(_WIN32)
# ifdef WINRT
    return LoadPackagedLibrary(filename.c_str(), 0);
# else
    return LoadLibraryW(filename.c_str());
#endif
#elif defined(__linux__) || defined(__APPLE__) || defined(__OpenBSD__) || defined(__FreeBSD__) || defined(__GLIBC__)
    return dlopen(filename.c_str(), RTLD_LAZY);
#endif
}

static inline
void libraryRelease_(LibHandle_t h)
{
#if defined(_WIN32)
    FreeLibrary(h);
#elif defined(__linux__) || defined(__APPLE__) || defined(__OpenBSD__) || defined(__FreeBSD__) || defined(__GLIBC__)
    dlclose(h);
#endif
}

static inline
std::string libraryPrefix()
{
#if defined(_WIN32)
    return "";
#else
    return "lib";
#endif
}
static inline
std::string librarySuffix()
{
#if defined(_WIN32)
    const char* suffix = ""
        CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
    #if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__)
        "_64"
    #endif
    #if defined(_DEBUG) && defined(DEBUG_POSTFIX)
        CVAUX_STR(DEBUG_POSTFIX)
    #endif
        ".dll";
    return suffix;
#else
    return ".so";
#endif
}

//============================

class DynamicLib
{
private:
    LibHandle_t handle;
    const FileSystemPath_t fname;

public:
    DynamicLib(const FileSystemPath_t& filename)
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
            CV_LOG_ERROR(NULL, "No symbol '" << symbolName << "' in " << toPrintablePath(fname));
        return res;
    }
    const std::string getName() const { return toPrintablePath(fname); }
private:
    void libraryLoad(const FileSystemPath_t& filename)
    {
        handle = libraryLoad_(filename);
        CV_LOG_INFO(NULL, "load " << toPrintablePath(filename) << " => " << (handle ? "OK" : "FAILED"));
    }
    void libraryRelease()
    {
        if (handle)
        {
            CV_LOG_INFO(NULL, "unload "<< toPrintablePath(fname));
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
            if (plugin_api_->api_header.opencv_version_major != CV_VERSION_MAJOR)
            {
                CV_LOG_ERROR(NULL, "Video I/O: wrong OpenCV major version used by plugin '" << plugin_api_->api_header.api_description << "': " <<
                    cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", plugin_api_->api_header.opencv_version_major, plugin_api_->api_header.opencv_version_minor))
                plugin_api_ = NULL;
                return;
            }
#ifdef HAVE_FFMPEG_WRAPPER
            if (plugin_api_->captureAPI == CAP_FFMPEG)
            {
                // no checks for OpenCV minor version
            }
            else
#endif
            if (plugin_api_->api_header.opencv_version_minor != CV_VERSION_MINOR)
            {
                CV_LOG_ERROR(NULL, "Video I/O: wrong OpenCV minor version used by plugin '" << plugin_api_->api_header.api_description << "': " <<
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
    Ptr<IVideoWriter> createWriter(const std::string& filename, int fourcc, double fps,
                                   const cv::Size& sz, const VideoWriterParameters& params) const CV_OVERRIDE;
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
std::vector<FileSystemPath_t> getPluginCandidates(const std::string& baseName)
{
    using namespace cv::utils;
    using namespace cv::utils::fs;
    const string baseName_l = toLowerCase(baseName);
    const string baseName_u = toUpperCase(baseName);
    const FileSystemPath_t baseName_l_fs = toFileSystemPath(baseName_l);
    vector<FileSystemPath_t> paths;
    const vector<string> paths_ = getConfigurationParameterPaths("OPENCV_VIDEOIO_PLUGIN_PATH", vector<string>());
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
#ifndef CV_VIDEOIO_PLUGIN_SUBDIRECTORY
            paths.push_back(binaryLocation);
#else
            paths.push_back(binaryLocation + toFileSystemPath("/") + toFileSystemPath(CV_VIDEOIO_PLUGIN_SUBDIRECTORY_STR));
#endif
        }
    }
    const string default_expr = libraryPrefix() + "opencv_videoio_" + baseName_l + "*" + librarySuffix();
    const string plugin_expr = getConfigurationParameterString((std::string("OPENCV_VIDEOIO_PLUGIN_") + baseName_u).c_str(), default_expr.c_str());
    vector<FileSystemPath_t> results;
#ifdef _WIN32
    FileSystemPath_t moduleName = toFileSystemPath(libraryPrefix() + "opencv_videoio_" + baseName_l + librarySuffix());
#ifndef WINRT
    if (baseName_u == "FFMPEG")  // backward compatibility
    {
        const wchar_t* ffmpeg_env_path = _wgetenv(L"OPENCV_FFMPEG_DLL_DIR");
        if (ffmpeg_env_path)
        {
            results.push_back(FileSystemPath_t(ffmpeg_env_path) + L"\\" + moduleName);
        }
    }
#endif
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
#if defined(_DEBUG) && defined(DEBUG_POSTFIX)
    if (baseName_u == "FFMPEG")  // backward compatibility
    {
        const FileSystemPath_t templ = toFileSystemPath(CVAUX_STR(DEBUG_POSTFIX) ".dll");
        FileSystemPath_t nonDebugName(moduleName);
        size_t suf = nonDebugName.rfind(templ);
        if (suf != FileSystemPath_t::npos)
        {
            nonDebugName.replace(suf, suf + templ.size(), L".dll");
            results.push_back(nonDebugName);
        }
    }
#endif // _DEBUG && DEBUG_POSTFIX
#else
    CV_LOG_INFO(NULL, "VideoIO pluigin (" << baseName << "): glob is '" << plugin_expr << "', " << paths.size() << " location(s)");
    for (const string & path : paths)
    {
        if (path.empty())
            continue;
        vector<string> candidates;
        cv::glob(utils::fs::join(path, plugin_expr), candidates);
        CV_LOG_INFO(NULL, "    - " << path << ": " << candidates.size());
        copy(candidates.begin(), candidates.end(), back_inserter(results));
    }
#endif
    CV_LOG_INFO(NULL, "Found " << results.size() << " plugin(s) for " << baseName);
    return results;
}

void PluginBackendFactory::loadPlugin()
{
    for (const FileSystemPath_t& plugin : getPluginCandidates(baseName_))
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
            CV_LOG_WARNING(NULL, "Video I/O: exception during plugin initialization: " << toPrintablePath(plugin) << ". SKIP");
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
            const std::string& filename, int fourcc, double fps, const cv::Size& sz,
            const VideoWriterParameters& params)
    {
        CV_Assert(plugin_api);
        CvPluginWriter writer = NULL;
        if (plugin_api->Writer_open)
        {
            CV_Assert(plugin_api->Writer_release);
            CV_Assert(!filename.empty());
            const bool isColor = params.get(VIDEOWRITER_PROP_IS_COLOR, true);
            if (CV_ERROR_OK == plugin_api->Writer_open(filename.c_str(), fourcc, fps, sz.width, sz.height, isColor, &writer))
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

Ptr<IVideoWriter> PluginBackend::createWriter(const std::string& filename, int fourcc, double fps,
                                              const cv::Size& sz, const VideoWriterParameters& params) const
{
    try
    {
        if (plugin_api_)
            return PluginWriter::create(plugin_api_, filename, fourcc, fps, sz, params); //.staticCast<IVideoWriter>();
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
