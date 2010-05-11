///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////



#ifndef INCLUDED_IMATHPLATFORM_H
#define INCLUDED_IMATHPLATFORM_H

//----------------------------------------------------------------------------
//
//	ImathPlatform.h
//
//	This file contains functions and constants which aren't 
//	provided by the system libraries, compilers, or includes on 
//	certain platforms.
//----------------------------------------------------------------------------

#include <math.h>

#if defined _WIN32 || defined _WIN64
    #ifndef M_PI
        #define M_PI 3.14159265358979323846
    #endif
#endif

//-----------------------------------------------------------------------------
//
//    Fixes for the "restrict" keyword.  These #ifdef's for detecting 
//    compiler versions courtesy of Boost's select_compiler_config.hpp; 
//    here is the copyright notice for that file:
//
//    (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//    and distribute this software is granted provided this copyright notice
//    appears in all copies. This software is provided "as is" without express 
//    or implied warranty, and with no claim as to its suitability for any 
//    purpose.
//
//    Some compilers support "restrict", in which case we do nothing.
//    Other compilers support some variant of it (e.g. "__restrict").
//    If we don't know anything about the compiler, we define "restrict"
//    to be a no-op.
//
//-----------------------------------------------------------------------------

#if defined __GNUC__
    #if !defined(restrict)
    	#define restrict __restrict
    #endif

#elif defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)
    // supports restrict, do nothing.

#elif defined __sgi
    // supports restrict, do nothing.

#else
    #define restrict

#endif

#endif // INCLUDED_IMATHPLATFORM_H
