// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include <opencv2/core/utils/configuration.private.hpp>

#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/utils/filesystem.private.hpp"
#include "opencv2/core/utils/filesystem.hpp"

//#define DEBUG_FS_UTILS

#ifdef DEBUG_FS_UTILS
#include <iostream>
#define DBG(...) __VA_ARGS__
#else
#define DBG(...)
#endif


#if OPENCV_HAVE_FILESYSTEM_SUPPORT

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#undef NOMINMAX
#define NOMINMAX
#include <windows.h>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <io.h>
#include <stdio.h>
#elif defined __linux__ || defined __APPLE__ || defined __HAIKU__ || defined __FreeBSD__ || defined __GNU__ || defined __EMSCRIPTEN__ || defined __QNX__
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#endif

#endif // OPENCV_HAVE_FILESYSTEM_SUPPORT

namespace cv { namespace utils { namespace fs {

#ifdef _WIN32
static const char native_separator = '\\';
#else
static const char native_separator = '/';
#endif

static inline
bool isPathSeparator(char c)
{
    return c == '/' || c == '\\';
}

cv::String join(const cv::String& base, const cv::String& path)
{
    if (base.empty())
        return path;
    if (path.empty())
        return base;

    bool baseSep = isPathSeparator(base[base.size() - 1]);
    bool pathSep = isPathSeparator(path[0]);
    String result;
    if (baseSep && pathSep)
    {
        result = base + path.substr(1);
    }
    else if (!baseSep && !pathSep)
    {
        result = base + native_separator + path;
    }
    else
    {
        result = base + path;
    }
    return result;
}

CV_EXPORTS cv::String getParent(const cv::String &path)
{
    std::wstring::size_type loc = path.find_last_of(L"\\");
    if (loc == std::wstring::npos)
        return std::wstring();
    return std::wstring(path, 0, loc);    
}

CV_EXPORTS std::wstring getParent(const std::wstring& path)
{
    std::wstring::size_type loc = path.find_last_of(L"/\\");
    if (loc == std::wstring::npos)
        return std::wstring();
    return std::wstring(path, 0, loc);
}

#if OPENCV_HAVE_FILESYSTEM_SUPPORT

cv::String canonical(const cv::String& path)
{
    cv::String result;
#ifdef _WIN32
    const char* result_str = _fullpath(NULL, path.c_str(), 0);
#else
    const char* result_str = realpath(path.c_str(), NULL);
#endif
    if (result_str)
    {
        result = cv::String(result_str);
        free((void*)result_str);
    }
    return result.empty() ? path : result;
}


bool exists(const cv::String& path)
{
    CV_INSTRUMENT_REGION();

#if defined _WIN32 || defined WINCE
    BOOL status = TRUE;
    {
        WIN32_FILE_ATTRIBUTE_DATA all_attrs;
#ifdef WINRT
        wchar_t wpath[MAX_PATH];
        size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
        CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
        status = ::GetFileAttributesExW(wpath, GetFileExInfoStandard, &all_attrs);
#else
        status = ::GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &all_attrs);
#endif
    }

    return !!status;
#else
    struct stat stat_buf;
    return (0 == stat(path.c_str(), &stat_buf));
#endif
}

CV_EXPORTS void remove_all(const cv::String& path)
{
    if (!exists(path))
        return;
    if (isDirectory(path))
    {
        std::vector<String> entries;
        utils::fs::glob(path, cv::String(), entries, false, true);
        for (size_t i = 0; i < entries.size(); i++)
        {
            const String& e = entries[i];
            remove_all(e);
        }
#ifdef _MSC_VER
        bool result = _rmdir(path.c_str()) == 0;
#else
        bool result = rmdir(path.c_str()) == 0;
#endif
        if (!result)
        {
            CV_LOG_ERROR(NULL, "Can't remove directory: " << path);
        }
    }
    else
    {
#ifdef _MSC_VER
        bool result = _unlink(path.c_str()) == 0;
#else
        bool result = unlink(path.c_str()) == 0;
#endif
        if (!result)
        {
            CV_LOG_ERROR(NULL, "Can't remove file: " << path);
        }
    }
}


cv::String getcwd()
{
    CV_INSTRUMENT_REGION();
    cv::AutoBuffer<char, 4096> buf;
#if defined WIN32 || defined _WIN32 || defined WINCE
#ifdef WINRT
    return cv::String();
#else
    DWORD sz = GetCurrentDirectoryA(0, NULL);
    buf.allocate((size_t)sz);
    sz = GetCurrentDirectoryA((DWORD)buf.size(), buf.data());
    return cv::String(buf.data(), (size_t)sz);
#endif
#elif defined __linux__ || defined __APPLE__ || defined __HAIKU__ || defined __FreeBSD__ || defined __EMSCRIPTEN__ || defined __QNX__
    for(;;)
    {
        char* p = ::getcwd(buf.data(), buf.size());
        if (p == NULL)
        {
            if (errno == ERANGE)
            {
                buf.allocate(buf.size() * 2);
                continue;
            }
            return cv::String();
        }
        break;
    }
    return cv::String(buf.data(), (size_t)strlen(buf.data()));
#else
    return cv::String();
#endif
}


bool createDirectory(const cv::String& path)
{
    CV_INSTRUMENT_REGION();
#if defined WIN32 || defined _WIN32 || defined WINCE
#ifdef WINRT
    wchar_t wpath[MAX_PATH];
    size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
    CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
    int result = CreateDirectoryA(wpath, NULL) ? 0 : -1;
#else
    int result = _mkdir(path.c_str());
#endif
#elif defined __linux__ || defined __APPLE__ || defined __HAIKU__ || defined __FreeBSD__ || defined __EMSCRIPTEN__ || defined __QNX__
    int result = mkdir(path.c_str(), 0777);
#else
    int result = -1;
#endif

    if (result == -1)
    {
        return isDirectory(path);
    }
    return true;
}

bool createDirectories(const cv::String& path_)
{
    cv::String path = path_;
    for (;;)
    {
        char last_char = path.empty() ? 0 : path[path.length() - 1];
        if (isPathSeparator(last_char))
        {
            path = path.substr(0, path.length() - 1);
            continue;
        }
        break;
    }

    if (path.empty() || path == "./" || path == ".\\" || path == ".")
        return true;
    if (isDirectory(path))
        return true;

    size_t pos = path.rfind('/');
    if (pos == cv::String::npos)
        pos = path.rfind('\\');
    if (pos != cv::String::npos)
    {
        cv::String parent_directory = path.substr(0, pos);
        if (!parent_directory.empty())
        {
            if (!createDirectories(parent_directory))
                return false;
        }
    }

    return createDirectory(path);
}

#ifdef _WIN32

struct FileLock::Impl
{
    Impl(const char* fname)
    {
        // http://support.microsoft.com/kb/316609
        int numRetries = 5;
        do
        {
            handle = ::CreateFileA(fname, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (INVALID_HANDLE_VALUE == handle)
            {
                if (ERROR_SHARING_VIOLATION == GetLastError())
                {
                    numRetries--;
                    Sleep(250);
                    continue;
                }
                else
                {
                    CV_Error_(Error::StsAssert, ("Can't open lock file: %s", fname));
                }
            }
            break;
        } while (numRetries > 0);
    }
    ~Impl()
    {
        if (INVALID_HANDLE_VALUE != handle)
        {
            ::CloseHandle(handle);
        }
    }

    bool lock()
    {
        OVERLAPPED overlapped;
        std::memset(&overlapped, 0, sizeof(overlapped));
        return !!::LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &overlapped);
    }
    bool unlock()
    {
        OVERLAPPED overlapped;
        std::memset(&overlapped, 0, sizeof(overlapped));
        return !!::UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &overlapped);
    }

    bool lock_shared()
    {
        OVERLAPPED overlapped;
        std::memset(&overlapped, 0, sizeof(overlapped));
        return !!::LockFileEx(handle, 0, 0, MAXDWORD, MAXDWORD, &overlapped);
    }
    bool unlock_shared()
    {
        return unlock();
    }

    HANDLE handle;

private:
    Impl(const Impl&); // disabled
    Impl& operator=(const Impl&); // disabled
};

#elif defined __linux__ || defined __APPLE__ || defined __HAIKU__ || defined __FreeBSD__ || defined __GNU__ || defined __EMSCRIPTEN__ || defined __QNX__

struct FileLock::Impl
{
    Impl(const char* fname)
    {
        handle = ::open(fname, O_RDWR);
        CV_Assert(handle != -1);
    }
    ~Impl()
    {
        if (handle >= 0)
            ::close(handle);
    }

    bool lock()
    {
        struct ::flock l;
        std::memset(&l, 0, sizeof(l));
        l.l_type = F_WRLCK;
        l.l_whence = SEEK_SET;
        l.l_start = 0;
        l.l_len = 0;
        DBG(std::cout << "Lock..." << std::endl);
        bool res = -1 != ::fcntl(handle, F_SETLKW, &l);
        return res;
    }
    bool unlock()
    {
        struct ::flock l;
        std::memset(&l, 0, sizeof(l));
        l.l_type = F_UNLCK;
        l.l_whence = SEEK_SET;
        l.l_start = 0;
        l.l_len = 0;
        DBG(std::cout << "Unlock..." << std::endl);
        bool res = -1 != ::fcntl(handle, F_SETLK, &l);
        return res;
    }

    bool lock_shared()
    {
        struct ::flock l;
        std::memset(&l, 0, sizeof(l));
        l.l_type = F_RDLCK;
        l.l_whence = SEEK_SET;
        l.l_start = 0;
        l.l_len = 0;
        DBG(std::cout << "Lock read..." << std::endl);
        bool res = -1 != ::fcntl(handle, F_SETLKW, &l);
        return res;
    }
    bool unlock_shared()
    {
        return unlock();
    }

    int handle;

private:
    Impl(const Impl&); // disabled
    Impl& operator=(const Impl&); // disabled
};

#endif

FileLock::FileLock(const char* fname)
    : pImpl(new Impl(fname))
{
    // nothing
}
FileLock::~FileLock()
{
    delete pImpl;
    pImpl = NULL;
}

void FileLock::lock() { CV_Assert(pImpl->lock()); }
void FileLock::unlock() { CV_Assert(pImpl->unlock()); }
void FileLock::lock_shared() { CV_Assert(pImpl->lock_shared()); }
void FileLock::unlock_shared() { CV_Assert(pImpl->unlock_shared()); }



cv::String getCacheDirectory(const char* sub_directory_name, const char* configuration_name)
{
    String cache_path;
    if (configuration_name)
    {
        cache_path = utils::getConfigurationParameterString(configuration_name, "");
    }
    if (cache_path.empty())
    {
        cv::String default_cache_path;
#ifdef _WIN32
        char tmp_path_buf[MAX_PATH+1] = {0};
        DWORD res = GetTempPath(MAX_PATH, tmp_path_buf);
        if (res > 0 && res <= MAX_PATH)
        {
            default_cache_path = tmp_path_buf;
        }
#elif defined __ANDROID__
        // no defaults
#elif defined __APPLE__
        const std::wstring tmpdir_env = utils::getConfigurationParameterString("TMPDIR");
        if (!tmpdir_env.empty() && utils::fs::isDirectory(tmpdir_env))
        {
            default_cache_path = tmpdir_env;
        }
        else
        {
            default_cache_path = "/tmp/";
            CV_LOG_WARNING(NULL, "Using world accessible cache directory. This may be not secure: " << default_cache_path);
        }
#elif defined __linux__ || defined __HAIKU__ || defined __FreeBSD__ || defined __GNU__
        // https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
        if (default_cache_path.empty())
        {
            const std::wstring xdg_cache_env = utils::getConfigurationParameterString("XDG_CACHE_HOME");
            if (!xdg_cache_env.empty() && utils::fs::isDirectory(xdg_cache_env))
            {
                default_cache_path = xdg_cache_env;
            }
        }
        if (default_cache_path.empty())
        {
            const std::wstring home_env = utils::getConfigurationParameterString("HOME");
            if (!home_env.empty() && utils::fs::isDirectory(home_env))
            {
                cv::String home_path = home_env;
                cv::String home_cache_path = utils::fs::join(home_path, ".cache/");
                if (utils::fs::isDirectory(home_cache_path))
                {
                    default_cache_path = home_cache_path;
                }
            }
        }
        if (default_cache_path.empty())
        {
            const char* temp_path = "/var/tmp/";
            if (utils::fs::isDirectory(temp_path))
            {
                default_cache_path = temp_path;
                CV_LOG_WARNING(NULL, "Using world accessible cache directory. This may be not secure: " << default_cache_path);
            }
        }
        if (default_cache_path.empty())
        {
            default_cache_path = "/tmp/";
            CV_LOG_WARNING(NULL, "Using world accessible cache directory. This may be not secure: " << default_cache_path);
        }
#else
        // no defaults
#endif
        CV_LOG_VERBOSE(NULL, 0, "default_cache_path = " << default_cache_path);
        if (!default_cache_path.empty())
        {
            if (utils::fs::isDirectory(default_cache_path))
            {
                cv::String default_cache_path_base = utils::fs::join(default_cache_path, "opencv");
                default_cache_path = utils::fs::join(default_cache_path_base, "4.x" CV_VERSION_STATUS);
                if (utils::getConfigurationParameterBool("OPENCV_CACHE_SHOW_CLEANUP_MESSAGE", true)
                    && !utils::fs::isDirectory(default_cache_path))
                {
                    std::vector<cv::String> existedCacheDirs;
                    try
                    {
                        utils::fs::glob_relative(default_cache_path_base, "*", existedCacheDirs, false, true);
                    }
                    catch (...)
                    {
                        // ignore
                    }
                    if (!existedCacheDirs.empty())
                    {
                        CV_LOG_WARNING(NULL, "Creating new OpenCV cache directory: " << default_cache_path);
                        CV_LOG_WARNING(NULL, "There are several neighbour directories, probably created by old OpenCV versions.");
                        CV_LOG_WARNING(NULL, "Feel free to cleanup these unused directories:");
                        for (size_t i = 0; i < existedCacheDirs.size(); i++)
                        {
                            CV_LOG_WARNING(NULL, "  - " << existedCacheDirs[i]);
                        }
                        CV_LOG_WARNING(NULL, "Note: This message is showed only once.");
                    }
                }
                if (sub_directory_name && sub_directory_name[0] != '\0')
                    default_cache_path = utils::fs::join(default_cache_path, cv::String(sub_directory_name) + native_separator);
                if (!utils::fs::createDirectories(default_cache_path))
                {
                    CV_LOG_DEBUG(NULL, "Can't create OpenCV cache sub-directory: " << default_cache_path);
                }
                else
                {
                    cache_path = default_cache_path;
                }
            }
            else
            {
                CV_LOG_INFO(NULL, "Can't find default cache directory (does it exist?): " << default_cache_path);
            }
        }
        else
        {
            CV_LOG_DEBUG(NULL, "OpenCV has no support to discover default cache directory on the current platform");
        }
    }
    else
    {
        if (cache_path == "disabled")
            return cache_path;
        if (!isDirectory(cache_path))
        {
            CV_LOG_WARNING(NULL, "Specified non-existed directory, creating OpenCV sub-directory for caching purposes: " << cache_path);
            if (!createDirectories(cache_path))
            {
                CV_LOG_ERROR(NULL, "Can't create OpenCV cache sub-directory: " << cache_path);
                cache_path.clear();
            }
        }
    }
    CV_Assert(cache_path.empty() || utils::fs::isDirectory(cache_path));
    if (!cache_path.empty())
    {
        if (!isPathSeparator(cache_path[cache_path.size() - 1]))
        {
            cache_path += native_separator;
        }
    }
    return cache_path;
}

#else
#define NOT_IMPLEMENTED CV_Error(Error::StsNotImplemented, "");
cv::String canonical(const cv::String& /*path*/) { NOT_IMPLEMENTED }
bool exists(const cv::String& /*path*/) { NOT_IMPLEMENTED }
void remove_all(const cv::String& /*path*/) { NOT_IMPLEMENTED }
cv::String getcwd() { NOT_IMPLEMENTED }
bool createDirectory(const cv::String& /*path*/) { NOT_IMPLEMENTED }
bool createDirectories(const cv::String& /*path*/) { NOT_IMPLEMENTED }
cv::String getCacheDirectory(const char* /*sub_directory_name*/, const char* /*configuration_name = NULL*/) { NOT_IMPLEMENTED }
#undef NOT_IMPLEMENTED
#endif // OPENCV_HAVE_FILESYSTEM_SUPPORT

}}} // namespace


#if OPENCV_HAVE_FILESYSTEM_SUPPORT
#include "plugin_loader.impl.hpp"
#endif
