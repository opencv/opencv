// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_STITCHING_UTIL_LOG_HPP__
#define __OPENCV_STITCHING_UTIL_LOG_HPP__

#ifndef ENABLE_LOG
#define ENABLE_LOG 0
#endif

// TODO remove LOG macros, add logging class
#if ENABLE_LOG
#ifdef __ANDROID__
  #include <iostream>
  #include <sstream>
  #include <android/log.h>
  #define LOG_STITCHING_MSG(msg) \
    do { \
        Stringstream _os; \
        _os << msg; \
       __android_log_print(ANDROID_LOG_DEBUG, "STITCHING", "%s", _os.str().c_str()); \
    } while(0);
#else
  #include <iostream>
  #define LOG_STITCHING_MSG(msg) for(;;) { std::cout << msg; std::cout.flush(); break; }
#endif
#else
  #define LOG_STITCHING_MSG(msg)
#endif

#define LOG_(_level, _msg)                     \
    for(;;)                                    \
    {                                          \
        using namespace std;                   \
        if ((_level) >= ::cv::detail::stitchingLogLevel()) \
        {                                      \
            LOG_STITCHING_MSG(_msg);           \
        }                                      \
    break;                                 \
    }


#define LOG(msg) LOG_(1, msg)
#define LOG_CHAT(msg) LOG_(0, msg)

#define LOGLN(msg) LOG(msg << std::endl)
#define LOGLN_CHAT(msg) LOG_CHAT(msg << std::endl)

//#if DEBUG_LOG_CHAT
//  #define LOG_CHAT(msg) LOG(msg)
//  #define LOGLN_CHAT(msg) LOGLN(msg)
//#else
//  #define LOG_CHAT(msg) do{}while(0)
//  #define LOGLN_CHAT(msg) do{}while(0)
//#endif

#endif // __OPENCV_STITCHING_UTIL_LOG_HPP__
