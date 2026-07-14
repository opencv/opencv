///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
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

#include "ImfDeepFrameBuffer.h"
#include "Iex.h"


using namespace std;
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepSlice::DeepSlice (PixelType t,
                      char *b,
                      size_t xst,
                      size_t yst,
                      size_t spst,
                      int xsm,
                      int ysm,
                      double fv,
                      bool xtc,
                      bool ytc)
:
    Slice (t, b, xst, yst, xsm, ysm, fv, xtc, ytc),
    sampleStride (spst)
{
    // empty
}


void
DeepFrameBuffer::insert (const char name[], const DeepSlice &slice)
{
    if (name[0] == 0)
    {
        THROW (IEX_NAMESPACE::ArgExc,
               "Frame buffer slice name cannot be an empty string.");
    }

    _map[name] = slice;
}


void
DeepFrameBuffer::insert (const string &name, const DeepSlice &slice)
{
    insert (name.c_str(), slice);
}


DeepSlice &
DeepFrameBuffer::operator [] (const char name[])
{
    SliceMap::iterator i = _map.find (name);

    if (i == _map.end())
    {
        THROW (IEX_NAMESPACE::ArgExc,
               "Cannot find frame buffer slice \"" << name << "\".");
    }

    return i->second;
}


const DeepSlice &
DeepFrameBuffer::operator [] (const char name[]) const
{
    SliceMap::const_iterator i = _map.find (name);

    if (i == _map.end())
    {
        THROW (IEX_NAMESPACE::ArgExc,
               "Cannot find frame buffer slice \"" << name << "\".");
    }

    return i->second;
}


DeepSlice &
DeepFrameBuffer::operator [] (const string &name)
{
    return this->operator[] (name.c_str());
}


const DeepSlice &
DeepFrameBuffer::operator [] (const string &name) const
{
    return this->operator[] (name.c_str());
}


DeepSlice *
DeepFrameBuffer::findSlice (const char name[])
{
    SliceMap::iterator i = _map.find (name);
    return (i == _map.end())? 0: &i->second;
}


const DeepSlice *
DeepFrameBuffer::findSlice (const char name[]) const
{
    SliceMap::const_iterator i = _map.find (name);
    return (i == _map.end())? 0: &i->second;
}


DeepSlice *
DeepFrameBuffer::findSlice (const string &name)
{
    return findSlice (name.c_str());
}


const DeepSlice *
DeepFrameBuffer::findSlice (const string &name) const
{
    return findSlice (name.c_str());
}


DeepFrameBuffer::Iterator
DeepFrameBuffer::begin ()
{
    return _map.begin();
}


DeepFrameBuffer::ConstIterator
DeepFrameBuffer::begin () const
{
    return _map.begin();
}


DeepFrameBuffer::Iterator
DeepFrameBuffer::end ()
{
    return _map.end();
}


DeepFrameBuffer::ConstIterator
DeepFrameBuffer::end () const
{
    return _map.end();
}


DeepFrameBuffer::Iterator
DeepFrameBuffer::find (const char name[])
{
    return _map.find (name);
}


DeepFrameBuffer::ConstIterator
DeepFrameBuffer::find (const char name[]) const
{
    return _map.find (name);
}


DeepFrameBuffer::Iterator
DeepFrameBuffer::find (const string &name)
{
    return find (name.c_str());
}


DeepFrameBuffer::ConstIterator
DeepFrameBuffer::find (const string &name) const
{
    return find (name.c_str());
}


void
DeepFrameBuffer::insertSampleCountSlice(const Slice & slice)
{
    if (slice.type != UINT)
    {
        throw IEX_NAMESPACE::ArgExc("The type of sample count slice should be UINT.");
    }

    _sampleCounts = slice;
}


const Slice &
DeepFrameBuffer::getSampleCountSlice() const
{
    return _sampleCounts;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
