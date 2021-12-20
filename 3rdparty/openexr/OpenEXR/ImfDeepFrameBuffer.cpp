//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

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
    sampleStride (static_cast<int>(spst))
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
