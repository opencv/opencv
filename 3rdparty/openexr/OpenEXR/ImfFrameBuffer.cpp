//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//      class Slice
//      class FrameBuffer
//
//-----------------------------------------------------------------------------

#include <ImfFrameBuffer.h>
#include "Iex.h"


using namespace std;

#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

Slice::Slice (PixelType t,
              char *b,
              size_t xst,
              size_t yst,
              int xsm,
              int ysm,
              double fv,
              bool xtc,
              bool ytc)
:
    type (t),
    base (b),
    xStride (xst),
    yStride (yst),
    xSampling (xsm),
    ySampling (ysm),
    fillValue (fv),
    xTileCoords (xtc),
    yTileCoords (ytc)
{
    // empty
}

Slice
Slice::Make (
    PixelType                   type,
    const void*                 ptr,
    const IMATH_NAMESPACE::V2i& origin,
    int64_t                     w,
    int64_t                     h,
    size_t                      xStride,
    size_t                      yStride,
    int                         xSampling,
    int                         ySampling,
    double                      fillValue,
    bool                        xTileCoords,
    bool                        yTileCoords)
{
    intptr_t base = reinterpret_cast<intptr_t> (const_cast<void *> (ptr));
    if (xStride == 0)
    {
        switch (type)
        {
            case UINT: xStride = sizeof (uint32_t); break;
            case HALF: xStride = sizeof (uint16_t); break;
            case FLOAT: xStride = sizeof (float); break;
            case NUM_PIXELTYPES:
                THROW (IEX_NAMESPACE::ArgExc, "Invalid pixel type.");
        }
    }
    if (yStride == 0)
        yStride = static_cast<size_t> (w / xSampling) * xStride;

    // data window is an int, so force promote to higher type to avoid
    // overflow for off y (degenerate size checks should be in
    // ImfHeader::sanityCheck, but offset can be large-ish)
    int64_t offx = (static_cast<int64_t> (origin.x) /
                    static_cast<int64_t> (xSampling));
    offx *= static_cast<int64_t> (xStride);

    int64_t offy = (static_cast<int64_t> (origin.y) /
                    static_cast<int64_t> (ySampling));
    offy *= static_cast<int64_t> (yStride);

    return Slice (
        type,
        reinterpret_cast<char*>(base - offx - offy),
        xStride,
        yStride,
        xSampling,
        ySampling,
        fillValue,
        xTileCoords,
        yTileCoords);
}

Slice
Slice::Make (
    PixelType                     type,
    const void*                   ptr,
    const IMATH_NAMESPACE::Box2i& dataWindow,
    size_t                        xStride,
    size_t                        yStride,
    int                           xSampling,
    int                           ySampling,
    double                        fillValue,
    bool                          xTileCoords,
    bool                          yTileCoords)
{
    return Make (
        type,
        ptr,
        dataWindow.min,
        static_cast<int64_t> (dataWindow.max.x) -
            static_cast<int64_t> (dataWindow.min.x) + 1,
        static_cast<int64_t> (dataWindow.max.y) -
            static_cast<int64_t> (dataWindow.min.y) + 1,
        xStride,
        yStride,
        xSampling,
        ySampling,
        fillValue,
        xTileCoords,
        yTileCoords);
}

void
FrameBuffer::insert (const char name[], const Slice &slice)
{
    if (name[0] == 0)
    {
        THROW (IEX_NAMESPACE::ArgExc,
               "Frame buffer slice name cannot be an empty string.");
    }

    _map[name] = slice;
}


void
FrameBuffer::insert (const string &name, const Slice &slice)
{
    insert (name.c_str(), slice);
}


Slice &
FrameBuffer::operator [] (const char name[])
{
    SliceMap::iterator i = _map.find (name);

    if (i == _map.end())
    {
        THROW (IEX_NAMESPACE::ArgExc,
               "Cannot find frame buffer slice \"" << name << "\".");
    }

    return i->second;
}


const Slice &
FrameBuffer::operator [] (const char name[]) const
{
    SliceMap::const_iterator i = _map.find (name);

    if (i == _map.end())
    {
        THROW (IEX_NAMESPACE::ArgExc,
               "Cannot find frame buffer slice \"" << name << "\".");
    }

    return i->second;
}


Slice &
FrameBuffer::operator [] (const string &name)
{
    return this->operator[] (name.c_str());
}


const Slice &
FrameBuffer::operator [] (const string &name) const
{
    return this->operator[] (name.c_str());
}


Slice *
FrameBuffer::findSlice (const char name[])
{
    SliceMap::iterator i = _map.find (name);
    return (i == _map.end())? 0: &i->second;
}


const Slice *
FrameBuffer::findSlice (const char name[]) const
{
    SliceMap::const_iterator i = _map.find (name);
    return (i == _map.end())? 0: &i->second;
}


Slice *
FrameBuffer::findSlice (const string &name)
{
    return findSlice (name.c_str());
}


const Slice *
FrameBuffer::findSlice (const string &name) const
{
    return findSlice (name.c_str());
}


FrameBuffer::Iterator
FrameBuffer::begin ()
{
    return _map.begin();
}


FrameBuffer::ConstIterator
FrameBuffer::begin () const
{
    return _map.begin();
}


FrameBuffer::Iterator
FrameBuffer::end ()
{
    return _map.end();
}


FrameBuffer::ConstIterator
FrameBuffer::end () const
{
    return _map.end();
}


FrameBuffer::Iterator
FrameBuffer::find (const char name[])
{
    return _map.find (name);
}


FrameBuffer::ConstIterator
FrameBuffer::find (const char name[]) const
{
    return _map.find (name);
}


FrameBuffer::Iterator
FrameBuffer::find (const string &name)
{
    return find (name.c_str());
}


FrameBuffer::ConstIterator
FrameBuffer::find (const string &name) const
{
    return find (name.c_str());
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
