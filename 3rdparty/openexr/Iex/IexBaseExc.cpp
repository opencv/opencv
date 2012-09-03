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



//---------------------------------------------------------------------
//
//	Constructors and destructors for our exception base class.
//
//---------------------------------------------------------------------

#include "IexBaseExc.h"

namespace Iex {
namespace {


StackTracer currentStackTracer = 0;


} // namespace


void	
setStackTracer (StackTracer stackTracer)
{
    currentStackTracer = stackTracer;
}


StackTracer
stackTracer ()
{
    return currentStackTracer;
}


BaseExc::BaseExc (const char* s) throw () :
    std::string (s? s: ""),
    _stackTrace (currentStackTracer? currentStackTracer(): "")
{
    // empty
}


BaseExc::BaseExc (const std::string &s) throw () :
    std::string (s),
    _stackTrace (currentStackTracer? currentStackTracer(): "")
{
    // empty
}


BaseExc::BaseExc (std::stringstream &s) throw () :
    std::string (s.str()),
    _stackTrace (currentStackTracer? currentStackTracer(): "")
{
    // empty
}


BaseExc::BaseExc (const BaseExc &be) throw () :
    std::string (be),
    _stackTrace (be._stackTrace)
{
    // empty
}


BaseExc::~BaseExc () throw ()
{
    // empty
}


const char *
BaseExc::what () const throw ()
{
    return c_str();
}


BaseExc &
BaseExc::assign (std::stringstream &s)
{
    std::string::assign (s.str());
    return *this;
}

BaseExc &
BaseExc::append (std::stringstream &s)
{
    std::string::append (s.str());
    return *this;
}


} // namespace Iex
