// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UTILS_FILESYSTEM_PRIVATE_HPP
#define OPENCV_UTILS_FILESYSTEM_PRIVATE_HPP

// TODO Move to CMake?
#ifndef OPENCV_HAVE_FILESYSTEM_SUPPORT
#  if defined(__EMSCRIPTEN__) || defined(__native_client__)
     /* no support */
#  elif defined WINRT || defined _WIN32_WCE
     /* not supported */
#  elif defined __ANDROID__ || defined __linux__ || defined _WIN32 || \
        defined __FreeBSD__ || defined __bsdi__ || defined __HAIKU__ || \
        defined __GNU__ || defined __QNX__
#      define OPENCV_HAVE_FILESYSTEM_SUPPORT 1
#  elif defined(__APPLE__)
#    include <TargetConditionals.h>
#    if (defined(TARGET_OS_OSX) && TARGET_OS_OSX) || (defined(TARGET_OS_IOS) && TARGET_OS_IOS)
#      define OPENCV_HAVE_FILESYSTEM_SUPPORT 1 // OSX, iOS only
#    endif
#  else
     /* unknown */
#  endif
#  ifndef OPENCV_HAVE_FILESYSTEM_SUPPORT
#    define OPENCV_HAVE_FILESYSTEM_SUPPORT 0
#  endif
#endif

#if OPENCV_HAVE_FILESYSTEM_SUPPORT
namespace cv { namespace utils { namespace fs {

/**
 * File-based lock object.
 *
 * Provides interprocess synchronization mechanism.
 * Platform dependent.
 *
 * Supports multiple readers / single writer access pattern (RW / readers-writer / shared-exclusive lock).
 *
 * File must exist.
 * File can't be re-used (for example, I/O operations via std::fstream is not safe)
 */
class CV_EXPORTS FileLock {
public:
    explicit FileLock(const char* fname);
    ~FileLock();

    void lock(); ///< acquire exclusive (writer) lock
    void unlock(); ///< release exclusive (writer) lock

    void lock_shared(); ///< acquire shareable (reader) lock
    void unlock_shared(); ///< release shareable (reader) lock

    struct Impl;
protected:
    Impl* pImpl;

private:
    FileLock(const FileLock&); // disabled
    FileLock& operator=(const FileLock&); // disabled
};

}}} // namespace
#endif
#endif // OPENCV_UTILS_FILESYSTEM_PRIVATE_HPP
