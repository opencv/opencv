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

#ifndef FLANN_LOGGER_H
#define FLANN_LOGGER_H

#include <stdio.h>
#include <stdarg.h>

#include "defines.h"


namespace cvflann
{

class Logger
{
    Logger() : stream(stdout), logLevel(FLANN_LOG_WARN) {}

    ~Logger()
    {
        if ((stream!=NULL)&&(stream!=stdout)) {
            fclose(stream);
        }
    }

    static Logger& instance()
    {
        static Logger logger;
        return logger;
    }

    void _setDestination(const char* name)
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

    int _log(int level, const char* fmt, va_list arglist)
    {
        if (level > logLevel ) return -1;
        int ret = vfprintf(stream, fmt, arglist);
        return ret;
    }

public:
    /**
     * Sets the logging level. All messages with lower priority will be ignored.
     * @param level Logging level
     */
    static void setLevel(int level) { instance().logLevel = level; }

    /**
     * Sets the logging destination
     * @param name Filename or NULL for console
     */
    static void setDestination(const char* name) { instance()._setDestination(name); }

    /**
     * Print log message
     * @param level Log level
     * @param fmt Message format
     * @return
     */
    static int log(int level, const char* fmt, ...)
    {
        va_list arglist;
        va_start(arglist, fmt);
        int ret = instance()._log(level,fmt,arglist);
        va_end(arglist);
        return ret;
    }

#define LOG_METHOD(NAME,LEVEL) \
    static int NAME(const char* fmt, ...) \
    { \
        va_list ap; \
        va_start(ap, fmt); \
        int ret = instance()._log(LEVEL, fmt, ap); \
        va_end(ap); \
        return ret; \
    }

    LOG_METHOD(fatal, FLANN_LOG_FATAL)
    LOG_METHOD(error, FLANN_LOG_ERROR)
    LOG_METHOD(warn, FLANN_LOG_WARN)
    LOG_METHOD(info, FLANN_LOG_INFO)

private:
    FILE* stream;
    int logLevel;
};

}

#endif //FLANN_LOGGER_H
