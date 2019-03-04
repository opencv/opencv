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
// IBackend implementation

namespace cv {

static bool param_VIDEOIO_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOIO_DEBUG", false);
static bool param_VIDEOCAPTURE_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOCAPTURE_DEBUG", false);
static bool param_VIDEOWRITER_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOWRITER_DEBUG", false);

Ptr<IVideoCapture> IBackend::tryOpenCapture(const std::string & backendName, const std::string & filename, int cameraNum) const
{
    try
    {
        if (param_VIDEOIO_DEBUG || param_VIDEOCAPTURE_DEBUG)
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): trying ...\n", backendName.c_str()));
        Ptr<IVideoCapture> icap = createCapture(filename, cameraNum);
        if (param_VIDEOIO_DEBUG ||param_VIDEOCAPTURE_DEBUG)
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): result=%p isOpened=%d ...\n", backendName.c_str(), icap.empty() ? NULL : icap.get(), icap.empty() ? -1: icap->isOpened()));
        return icap;
    }
    catch(const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n", backendName.c_str(), e.what()));
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n", backendName.c_str(), e.what()));
    }
    catch(...)
    {
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n", backendName.c_str()));
    }
    return 0;
}

Ptr<IVideoWriter> IBackend::tryOpenWriter(const std::string & backendName, const std::string& filename, int _fourcc, double fps, const Size &frameSize, bool isColor) const
{
    try
    {
        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG)
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): trying ...\n", backendName.c_str()));
        Ptr<IVideoWriter> iwriter = createWriter(filename, _fourcc, fps, frameSize, isColor);
        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG)
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): result=%p  isOpened=%d...\n", backendName.c_str(), iwriter.empty() ? NULL : iwriter.get(), iwriter.empty() ? iwriter->isOpened() : -1));
        return iwriter;
    }
    catch(const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n", backendName.c_str(), e.what()));
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n", backendName.c_str(), e.what()));
    }
    catch(...)
    {
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n", backendName.c_str()));
    }
    return 0;
}

} // cv::

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

inline static void * getSymbol_(void *h, const std::string &symbolName)
{
#if defined(_WIN32)
    return (void*)GetProcAddress(static_cast<HMODULE>(h), symbolName.c_str());
#elif defined(__linux__) || defined(__APPLE__)
    return dlsym(h, symbolName.c_str());
#endif
}

inline static void * libraryLoad_(const std::string &filename)
{
#if defined(_WIN32)
    return static_cast<HMODULE>(LoadLibraryA(filename.c_str()));
#elif defined(__linux__) || defined(__APPLE__)
    return dlopen(filename.c_str(), RTLD_LAZY);
#endif
}

inline static void libraryRelease_(void *h)
{
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(h));
#elif defined(__linux__) || defined(__APPLE__)
    dlclose(h);
#endif
}

inline static std::string libraryPrefix()
{
#if defined(_WIN32)
    return string();
#else
    return "lib";
#endif
}
inline static std::string librarySuffix()
{
#if defined(_WIN32)
    return "dll";
#elif defined(__APPLE__)
    return "dylib";
#else
    return "so";
#endif
}

//============================

class cv::DynamicBackend::DynamicLib
{
private:
    void * handle;
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
    void* getSymbol(const std::string & symbolName) const
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

private:
    void libraryLoad(const std::string &filename)
    {
        handle = libraryLoad_(filename);
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

// Utility function
static bool verifyVersion(cv_get_version_t * fun)
{
    if (!fun)
        return false;
    int major, minor, patch, api, abi;
    fun(major, minor, patch, api, abi);
    if (api < API_VERSION || abi != ABI_VERSION)
    {
        CV_LOG_ERROR(NULL, "Bad plugin API/ABI (" << api << "/" << abi << "), expected " << API_VERSION << "/" << ABI_VERSION);
        return false;
    }
#ifdef STRICT_PLUGIN_CHECK
    if (major != CV_MAJOR_VERSION || minor != CV_MINOR_VERSION)
    {
        CV_LOG_ERROR(NULL, "Bad plugin version (" << major << "." << minor << "), expected " << CV_MAJOR_VERSION << "/" << CV_MINOR_VERSION);
        return false;
    }
#endif
    return true;
}

//============================

class cv::DynamicBackend::CaptureTable
{
public:
    cv_get_version_t *cv_get_version;
    cv_domain_t *cv_domain;
    cv_open_capture_t *cv_open_capture;
    cv_get_cap_prop_t *cv_get_cap_prop;
    cv_set_cap_prop_t *cv_set_cap_prop;
    cv_grab_t *cv_grab;
    cv_retrieve_t *cv_retrieve;
    cv_release_capture_t *cv_release_capture;
    bool isComplete;
public:
    CaptureTable(const cv::DynamicBackend::DynamicLib & p)
        : isComplete(true)
    {
    #define READ_FUN(name) \
        name = reinterpret_cast<name##_t*>(p.getSymbol(#name)); \
        isComplete = isComplete && (name)

        READ_FUN(cv_get_version);
        READ_FUN(cv_domain);
        READ_FUN(cv_open_capture);
        READ_FUN(cv_get_cap_prop);
        READ_FUN(cv_set_cap_prop);
        READ_FUN(cv_grab);
        READ_FUN(cv_retrieve);
        READ_FUN(cv_release_capture);

    #undef READ_FUN
    }
};

class cv::DynamicBackend::WriterTable
{
public:
    cv_get_version_t *cv_get_version;
    cv_domain_t *cv_domain;
    cv_open_writer_t *cv_open_writer;
    cv_get_wri_prop_t *cv_get_wri_prop;
    cv_set_wri_prop_t *cv_set_wri_prop;
    cv_write_t *cv_write;
    cv_release_writer_t *cv_release_writer;
    bool isComplete;
public:
    WriterTable(const cv::DynamicBackend::DynamicLib & p)
        : isComplete(true)
    {
    #define READ_FUN(name) \
        name = reinterpret_cast<name##_t*>(p.getSymbol(#name)); \
        isComplete = isComplete && (name)

        READ_FUN(cv_get_version);
        READ_FUN(cv_domain);
        READ_FUN(cv_open_writer);
        READ_FUN(cv_get_wri_prop);
        READ_FUN(cv_set_wri_prop);
        READ_FUN(cv_write);
        READ_FUN(cv_release_writer);

    #undef READ_FUN
    }
};

//============================

class DynamicCapture;
class DynamicWriter;

cv::DynamicBackend::DynamicBackend(const std::string &filename)
    : lib(0), cap_tbl(0), wri_tbl(0)
{
    lib = new DynamicLib(filename);
    if (lib->isLoaded())
    {
        cap_tbl = new CaptureTable(*lib);
        wri_tbl = new WriterTable(*lib);
    }
}

cv::DynamicBackend::~DynamicBackend()
{
    if (cap_tbl)
        delete cap_tbl;
    if (wri_tbl)
        delete wri_tbl;
    if (lib)
        delete lib;
}

bool cv::DynamicBackend::canCreateCapture(VideoCaptureAPIs api) const
{
    return lib && lib->isLoaded() && cap_tbl && cap_tbl->isComplete && verifyVersion(cap_tbl->cv_get_version) && (cap_tbl->cv_domain() == api);
}

bool cv::DynamicBackend::canCreateWriter(VideoCaptureAPIs api) const
{
    return lib && lib->isLoaded() && wri_tbl && wri_tbl->isComplete && verifyVersion(wri_tbl->cv_get_version) && (wri_tbl->cv_domain() == api);
}

cv::Ptr<cv::IVideoCapture> cv::DynamicBackend::createCapture(const std::string & filename, int camera) const
{
    return makePtr<DynamicCapture>(cap_tbl, filename, camera).staticCast<IVideoCapture>();
}

cv::Ptr<cv::IVideoWriter> cv::DynamicBackend::createWriter(const std::string & filename, int fourcc, double fps, const cv::Size &sz, bool isColor) const
{
    return makePtr<DynamicWriter>(wri_tbl, filename, fourcc, fps, sz, isColor).staticCast<IVideoWriter>();
}

inline static std::vector<string> getPluginCandidates()
{
    using namespace cv::utils;
    using namespace cv::utils::fs;
    const vector<string> default_paths = { getParent(getBinLocation()) };
    const vector<string> paths = getConfigurationParameterPaths("OPENCV_VIDEOIO_PLUGIN_PATH", default_paths);
    const string default_expr = libraryPrefix() + "opencv_videoio_*." + librarySuffix();
    const string expr = getConfigurationParameterString("OPENCV_VIDEOIO_PLUGIN_NAME", default_expr.c_str());
    CV_LOG_INFO(NULL, "VideoIO pluigins: glob is '" << expr << "', " << paths.size() << " location(s)");
    vector<string> results;
    for(const string & path : paths)
    {
        if (path.empty())
            continue;
        vector<string> candidates;
        cv::glob(join(path, expr), candidates);
        CV_LOG_INFO(NULL, "VideoIO pluigins in " << path << ": " << candidates.size());
        copy(candidates.begin(), candidates.end(), back_inserter(results));
    }
    CV_LOG_INFO(NULL, "Found " << results.size() << " plugin(s)");
    return results;
}

cv::Ptr<cv::DynamicBackend> cv::DynamicBackend::load(cv::VideoCaptureAPIs api, int mode)
{
    for(const std::string & plugin : getPluginCandidates())
    {
        bool res = true;
        Ptr<DynamicBackend> factory = makePtr<DynamicBackend>(plugin);
        if (factory)
            if (mode & cv::MODE_CAPTURE_BY_INDEX || mode & cv::MODE_CAPTURE_BY_FILENAME)
            {
                res = res && factory->canCreateCapture(api);
            }
        if (mode & cv::MODE_WRITER)
        {
            res = res && factory->canCreateWriter(api);
        }
        if (res)
            return factory;
    }
    return 0;
}

//==================================================================================================
// DynamicCapture

class DynamicCapture : public cv::IVideoCapture
{
    const cv::DynamicBackend::CaptureTable * tbl;
    void * capture;
public:
    DynamicCapture(const cv::DynamicBackend::CaptureTable * tbl_, const std::string &filename, int camera)
        : tbl(tbl_), capture(0)
    {
        CV_Assert(!capture);
        if (tbl->cv_open_capture)
            tbl->cv_open_capture(filename.empty() ? 0 : filename.c_str(), camera, capture);
    }
    ~DynamicCapture()
    {
        if (capture)
        {
            CV_Assert(tbl->cv_release_capture);
            tbl->cv_release_capture(capture);
            capture = 0;
        }
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        CV_Assert(capture);
        double val = -1;
        tbl->cv_get_cap_prop(capture, prop, val);
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        CV_Assert(capture);
        return tbl->cv_set_cap_prop(capture, prop, val);
    }
    bool grabFrame() CV_OVERRIDE
    {
        CV_Assert(capture);
        return tbl->cv_grab(capture);
    }
    static bool local_retrieve(unsigned char * data, int step, int width, int height, int cn, void * userdata)
    {
        cv::Mat * img = static_cast<cv::Mat*>(userdata);
        if (!img)
            return false;
        cv::Mat(cv::Size(width, height), CV_MAKETYPE(CV_8U, cn), data, step).copyTo(*img);
        return true;
    }
    bool retrieveFrame(int idx, cv::OutputArray img) CV_OVERRIDE
    {
        CV_Assert(capture);
        cv::Mat frame;
        bool res = tbl->cv_retrieve(capture, idx, &local_retrieve, &frame);
        if (res)
            frame.copyTo(img);
        return res;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return capture != NULL;
    }
    int getCaptureDomain() CV_OVERRIDE
    {
        return tbl->cv_domain();
    }
};

//==================================================================================================
// DynamicWriter

class DynamicWriter : public cv::IVideoWriter
{
    const cv::DynamicBackend::WriterTable * tbl;
    void * writer;
public:
    DynamicWriter(const cv::DynamicBackend::WriterTable * tbl_, const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor)
        : tbl(tbl_), writer(0)
    {
        CV_Assert(!writer);
        if(tbl->cv_open_writer)
            tbl->cv_open_writer(filename.empty() ? 0 : filename.c_str(), fourcc, fps, sz.width, sz.height, isColor, writer);
    }
    ~DynamicWriter()
    {
        if (writer)
        {
            CV_Assert(tbl->cv_release_writer);
            tbl->cv_release_writer(writer);
            writer = 0;
        }
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        CV_Assert(writer);
        double val = -1;
        tbl->cv_get_wri_prop(writer, prop, val);
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        CV_Assert(writer);
        return tbl->cv_set_wri_prop(writer, prop, val);
    }
    bool isOpened() const CV_OVERRIDE
    {
        return writer != NULL;
    }
    void write(cv::InputArray arr) CV_OVERRIDE
    {
        cv::Mat img = arr.getMat();
        CV_Assert(writer);
        tbl->cv_write(writer, img.data, (int)img.step[0], img.cols, img.rows, img.channels());
    }
    int getCaptureDomain() const CV_OVERRIDE
    {
        return tbl->cv_domain();
    }
};
