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

#include "logger.h"

#include <cstdio>
#include <cstdarg>
#include <sstream>

using namespace std;

namespace cvflann
{

Logger logger;

int Logger::log(int level, const char* fmt, ...)
{
    if (level > logLevel ) return -1;

    int ret;
    va_list arglist;
    va_start(arglist, fmt);
    ret = vfprintf(stream, fmt, arglist);
    va_end(arglist);

    return ret;
}

int Logger::log(int level, const char* fmt, va_list arglist)
{
    if (level > logLevel ) return -1;

    int ret;
    ret = vfprintf(stream, fmt, arglist);

    return ret;
}


#define LOG_METHOD(NAME,LEVEL) \
    int Logger::NAME(const char* fmt, ...) \
    { \
        int ret; \
        va_list ap; \
        va_start(ap, fmt); \
        ret = log(LEVEL, fmt, ap); \
        va_end(ap); \
        return ret; \
    }


LOG_METHOD(fatal, LOG_FATAL)
LOG_METHOD(error, LOG_ERROR)
LOG_METHOD(warn, LOG_WARN)
LOG_METHOD(info, LOG_INFO)

}
