//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class DeepImageLevel
//
//----------------------------------------------------------------------------

#include "ImfDeepImageLevel.h"
#include "ImfDeepImage.h"
#include <Iex.h>
#include <cassert>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepImageLevel::DeepImageLevel (
    DeepImage&   image,
    int          xLevelNumber,
    int          yLevelNumber,
    const Box2i& dataWindow)
    : ImageLevel (image, xLevelNumber, yLevelNumber), _sampleCounts (*this)
{
    resize (dataWindow);
}

DeepImage&
DeepImageLevel::deepImage ()
{
    return static_cast<DeepImage&> (image ());
}

const DeepImage&
DeepImageLevel::deepImage () const
{
    return static_cast<const DeepImage&> (image ());
}

DeepImageLevel::~DeepImageLevel ()
{
    clearChannels ();
}

void
DeepImageLevel::setSamplesToZero (
    size_t i, unsigned int oldNumSamples, unsigned int newNumSamples)
{
    for (ChannelMap::iterator j = _channels.begin (); j != _channels.end ();
         ++j)
    {
        j->second->setSamplesToZero (i, oldNumSamples, newNumSamples);
    }
}

void
DeepImageLevel::moveSampleList (
    size_t       i,
    unsigned int oldNumSamples,
    unsigned int newNumSamples,
    size_t       newSampleListPosition)
{
    for (ChannelMap::iterator j = _channels.begin (); j != _channels.end ();
         ++j)
    {
        j->second->moveSampleList (
            i, oldNumSamples, newNumSamples, newSampleListPosition);
    }
}

void
DeepImageLevel::moveSamplesToNewBuffer (
    const unsigned int* oldNumSamples,
    const unsigned int* newNumSamples,
    const size_t*       newSampleListPositions)
{
    for (ChannelMap::iterator j = _channels.begin (); j != _channels.end ();
         ++j)
    {
        j->second->moveSamplesToNewBuffer (
            oldNumSamples, newNumSamples, newSampleListPositions);
    }
}

void
DeepImageLevel::initializeSampleLists ()
{
    for (ChannelMap::iterator j = _channels.begin (); j != _channels.end ();
         ++j)
        j->second->initializeSampleLists ();
}

void
DeepImageLevel::resize (const Box2i& dataWindow)
{
    //
    // Note: if the following code throws an exception, then the image level
    // may be left in an inconsistent state where some channels have been
    // resized, but others have not.  However, the image to which this level
    // belongs will catch the exception and clean up the mess.
    //

    ImageLevel::resize (dataWindow);
    _sampleCounts.resize ();

    for (ChannelMap::iterator i = _channels.begin (); i != _channels.end ();
         ++i)
        i->second->resize ();
}

void
DeepImageLevel::shiftPixels (int dx, int dy)
{
    ImageLevel::shiftPixels (dx, dy);

    _sampleCounts.resetBasePointer ();

    for (ChannelMap::iterator i = _channels.begin (); i != _channels.end ();
         ++i)
        i->second->resetBasePointer ();
}

void
DeepImageLevel::insertChannel (
    const string& name,
    PixelType     type,
    int           xSampling,
    int           ySampling,
    bool          pLinear)
{
    if (xSampling != 1 && ySampling != 1)
    {
        THROW (
            ArgExc,
            "Cannot create deep image channel "
                << name
                << " "
                   "with x sampling rate "
                << xSampling
                << " and "
                   "and y sampling rate "
                << ySampling
                << ". X and y "
                   "sampling rates for deep channels must be 1.");
    }

    if (_channels.find (name) != _channels.end ()) throwChannelExists (name);

    switch (type)
    {
        case HALF:
            _channels[name] = new DeepHalfChannel (*this, pLinear);
            break;

        case FLOAT:
            _channels[name] = new DeepFloatChannel (*this, pLinear);
            break;

        case UINT:
            _channels[name] = new DeepUIntChannel (*this, pLinear);
            break;

        default: assert (false);
    }
}

void
DeepImageLevel::eraseChannel (const string& name)
{
    ChannelMap::iterator i = _channels.find (name);

    if (i != _channels.end ())
    {
        delete i->second;
        _channels.erase (i);
    }
}

void
DeepImageLevel::clearChannels ()
{
    for (ChannelMap::iterator i = _channels.begin (); i != _channels.end ();
         ++i)
        delete i->second;

    _channels.clear ();
}

void
DeepImageLevel::renameChannel (const string& oldName, const string& newName)
{
    ChannelMap::iterator oldChannel = _channels.find (oldName);

    assert (oldChannel != _channels.end ());
    assert (_channels.find (newName) == _channels.end ());

    _channels[newName] = oldChannel->second;
    _channels.erase (oldChannel);
}

void
DeepImageLevel::renameChannels (const RenamingMap& oldToNewNames)
{
    renameChannelsInMap (oldToNewNames, _channels);
}

DeepImageChannel*
DeepImageLevel::findChannel (const string& name)
{
    ChannelMap::iterator i = _channels.find (name);

    if (i != _channels.end ())
        return i->second;
    else
        return 0;
}

const DeepImageChannel*
DeepImageLevel::findChannel (const string& name) const
{
    ChannelMap::const_iterator i = _channels.find (name);

    if (i != _channels.end ())
        return i->second;
    else
        return 0;
}

DeepImageChannel&
DeepImageLevel::channel (const string& name)
{
    ChannelMap::iterator i = _channels.find (name);

    if (i == _channels.end ()) throwBadChannelName (name);

    return *i->second;
}

const DeepImageChannel&
DeepImageLevel::channel (const string& name) const
{
    ChannelMap::const_iterator i = _channels.find (name);

    if (i == _channels.end ()) throwBadChannelName (name);

    return *i->second;
}

DeepImageLevel::Iterator
DeepImageLevel::begin ()
{
    return _channels.begin ();
}

DeepImageLevel::ConstIterator
DeepImageLevel::begin () const
{
    return _channels.begin ();
}

DeepImageLevel::Iterator
DeepImageLevel::end ()
{
    return _channels.end ();
}

DeepImageLevel::ConstIterator
DeepImageLevel::end () const
{
    return _channels.end ();
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
