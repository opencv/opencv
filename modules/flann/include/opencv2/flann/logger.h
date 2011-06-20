/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef _OPENCV_LOGGER_H_
#define _OPENCV_LOGGER_H_

#include <cstdio>
#include <stdarg.h>

namespace cvflann
{
    
enum flann_log_level_t {
    FLANN_LOG_NONE = 0,
    FLANN_LOG_FATAL = 1,
    FLANN_LOG_ERROR = 2,
    FLANN_LOG_WARN = 3,
    FLANN_LOG_INFO = 4
};
    
class CV_EXPORTS Logger
{
    FILE* stream;
    int logLevel;

public:

    Logger() : stream(stdout), logLevel(FLANN_LOG_WARN) {};

    ~Logger()
    {
        if (stream!=NULL && stream!=stdout) {
            fclose(stream);
        }
    }

    void setDestination(const char* name)
    {
        if (name==NULL) {
            stream = stdout;
        }
        else {
            stream = fopen(name,"w");
            if (stream == NULL) {
                stream = stdout;
            }
        }
    }

    void setLevel(int level) { logLevel = level; }

    int log(int level, const char* fmt, ...);

    int log(int level, const char* fmt, va_list arglist);

    int fatal(const char* fmt, ...);

    int error(const char* fmt, ...);

    int warn(const char* fmt, ...);

    int info(const char* fmt, ...);
};

CV_EXPORTS Logger& logger();

} // namespace cvflann

#endif //_OPENCV_LOGGER_H_
