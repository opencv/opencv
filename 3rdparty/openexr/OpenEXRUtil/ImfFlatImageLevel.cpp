//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class FlatImageLevel
//
//----------------------------------------------------------------------------

#include "ImfFlatImageLevel.h"
#include "ImfFlatImage.h"
#include <Iex.h>
#include <cassert>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

FlatImageLevel::FlatImageLevel (
    FlatImage&   image,
    int          xLevelNumber,
    int          yLevelNumber,
    const Box2i& dataWindow)
    : ImageLevel (image, xLevelNumber, yLevelNumber)
{
    resize (dataWindow);
}

FlatImage&
FlatImageLevel::flatImage ()
{
    return static_cast<FlatImage&> (image ());
}

const FlatImage&
FlatImageLevel::flatImage () const
{
    return static_cast<const FlatImage&> (image ());
}

FlatImageLevel::~FlatImageLevel ()
{
    clearChannels ();
}

void
FlatImageLevel::resize (const Box2i& dataWindow)
{
    //
    // Note: if the following code throws an exception, then the image level
    // may be left in an inconsistent state where some channels have been
    // resized, but others have not.  However, the image to which this level
    // belongs will catch the exception and clean up the mess.
    //

    ImageLevel::resize (dataWindow);

    for (ChannelMap::iterator i = _channels.begin (); i != _channels.end ();
         ++i)
        i->second->resize ();
}

void
FlatImageLevel::shiftPixels (int dx, int dy)
{
    ImageLevel::shiftPixels (dx, dy);

    for (ChannelMap::iterator i = _channels.begin (); i != _channels.end ();
         ++i)
        i->second->resetBasePointer ();
}

void
FlatImageLevel::insertChannel (
    const string& name,
    PixelType     type,
    int           xSampling,
    int           ySampling,
    bool          pLinear)
{
    if (_channels.find (name) != _channels.end ()) throwChannelExists (name);

    switch (type)
    {
        case HALF:
            _channels[name] =
                new FlatHalfChannel (*this, xSampling, ySampling, pLinear);
            break;

        case FLOAT:
            _channels[name] =
                new FlatFloatChannel (*this, xSampling, ySampling, pLinear);
            break;

        case UINT:
            _channels[name] =
                new FlatUIntChannel (*this, xSampling, ySampling, pLinear);
            break;

        default: assert (false);
    }
}

void
FlatImageLevel::eraseChannel (const string& name)
{
    ChannelMap::iterator i = _channels.find (name);

    if (i != _channels.end ())
    {
        delete i->second;
        _channels.erase (i);
    }
}

void
FlatImageLevel::clearChannels ()
{
    for (ChannelMap::iterator i = _channels.begin (); i != _channels.end ();
         ++i)
        delete i->second;

    _channels.clear ();
}

void
FlatImageLevel::renameChannel (const string& oldName, const string& newName)
{
    ChannelMap::iterator oldChannel = _channels.find (oldName);

    assert (oldChannel != _channels.end ());
    assert (_channels.find (newName) == _channels.end ());

    _channels[newName] = oldChannel->second;
    _channels.erase (oldChannel);
}

void
FlatImageLevel::renameChannels (const RenamingMap& oldToNewNames)
{
    renameChannelsInMap (oldToNewNames, _channels);
}

FlatImageChannel*
FlatImageLevel::findChannel (const string& name)
{
    ChannelMap::iterator i = _channels.find (name);

    if (i != _channels.end ())
        return i->second;
    else
        return 0;
}

const FlatImageChannel*
FlatImageLevel::findChannel (const string& name) const
{
    ChannelMap::const_iterator i = _channels.find (name);

    if (i != _channels.end ())
        return i->second;
    else
        return 0;
}

FlatImageChannel&
FlatImageLevel::channel (const string& name)
{
    ChannelMap::iterator i = _channels.find (name);

    if (i == _channels.end ()) throwBadChannelName (name);

    return *i->second;
}

const FlatImageChannel&
FlatImageLevel::channel (const string& name) const
{
    ChannelMap::const_iterator i = _channels.find (name);

    if (i == _channels.end ()) throwBadChannelName (name);

    return *i->second;
}

FlatImageLevel::Iterator
FlatImageLevel::begin ()
{
    return _channels.begin ();
}

FlatImageLevel::ConstIterator
FlatImageLevel::begin () const
{
    return _channels.begin ();
}

FlatImageLevel::Iterator
FlatImageLevel::end ()
{
    return _channels.end ();
}

FlatImageLevel::ConstIterator
FlatImageLevel::end () const
{
    return _channels.end ();
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
