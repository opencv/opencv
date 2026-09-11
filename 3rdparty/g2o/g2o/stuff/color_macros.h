// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_COLOR_MACROS_H
#define G2O_COLOR_MACROS_H

// font attributes
#define FT_BOLD      "\033[1m"
#define FT_UNDERLINE "\033[4m"

//background color
#define BG_BLACK     "\033[40m"
#define BG_RED       "\033[41m"
#define BG_GREEN     "\033[42m"
#define BG_YELLOW    "\033[43m"
#define BG_LIGHTBLUE "\033[44m"
#define BG_MAGENTA   "\033[45m"
#define BG_BLUE      "\033[46m"
#define BG_WHITE     "\033[47m"

// font color
#define CL_BLACK(s)     "\033[30m" << s << "\033[0m"
#define CL_RED(s)       "\033[31m" << s << "\033[0m"
#define CL_GREEN(s)     "\033[32m" << s << "\033[0m"
#define CL_YELLOW(s)    "\033[33m" << s << "\033[0m"
#define CL_LIGHTBLUE(s) "\033[34m" << s << "\033[0m"
#define CL_MAGENTA(s)   "\033[35m" << s << "\033[0m"
#define CL_BLUE(s)      "\033[36m" << s << "\033[0m"
#define CL_WHITE(s)     "\033[37m" << s << "\033[0m"

#define FG_BLACK     "\033[30m"
#define FG_RED       "\033[31m"
#define FG_GREEN     "\033[32m"
#define FG_YELLOW    "\033[33m"
#define FG_LIGHTBLUE "\033[34m"
#define FG_MAGENTA   "\033[35m"
#define FG_BLUE      "\033[36m"
#define FG_WHITE     "\033[37m"

#define FG_NORM      "\033[0m"

#endif
