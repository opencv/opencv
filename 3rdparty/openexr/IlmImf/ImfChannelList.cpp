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



//-----------------------------------------------------------------------------
//
//	class Channel
//	class ChannelList
//
//-----------------------------------------------------------------------------

#include <ImfChannelList.h>
#include <Iex.h>


using std::string;
using std::set;

namespace Imf {


Channel::Channel (PixelType t, int xs, int ys, bool pl):
    type (t),
    xSampling (xs),
    ySampling (ys),
    pLinear (pl)
{
    // empty
}


bool
Channel::operator == (const Channel &other) const
{
    return type == other.type &&
       xSampling == other.xSampling &&
       ySampling == other.ySampling &&
       pLinear == other.pLinear;
}


void
ChannelList::insert (const char name[], const Channel &channel)
{
    if (name[0] == 0)
    THROW (Iex::ArgExc, "Image channel name cannot be an empty string.");

    _map[name] = channel;
}


void
ChannelList::insert (const string &name, const Channel &channel)
{
    insert (name.c_str(), channel);
}


Channel &
ChannelList::operator [] (const char name[])
{
    ChannelMap::iterator i = _map.find (name);

    if (i == _map.end())
    THROW (Iex::ArgExc, "Cannot find image channel \"" << name << "\".");

    return i->second;
}


const Channel &
ChannelList::operator [] (const char name[]) const
{
    ChannelMap::const_iterator i = _map.find (name);

    if (i == _map.end())
    THROW (Iex::ArgExc, "Cannot find image channel \"" << name << "\".");

    return i->second;
}


Channel &
ChannelList::operator [] (const string &name)
{
    return this->operator[] (name.c_str());
}


const Channel &
ChannelList::operator [] (const string &name) const
{
    return this->operator[] (name.c_str());
}


Channel *
ChannelList::findChannel (const char name[])
{
    ChannelMap::iterator i = _map.find (name);
    return (i == _map.end())? 0: &i->second;
}


const Channel *
ChannelList::findChannel (const char name[]) const
{
    ChannelMap::const_iterator i = _map.find (name);
    return (i == _map.end())? 0: &i->second;
}


Channel *
ChannelList::findChannel (const string &name)
{
    return findChannel (name.c_str());
}


const Channel *
ChannelList::findChannel (const string &name) const
{
    return findChannel (name.c_str());
}


ChannelList::Iterator
ChannelList::begin ()
{
    return _map.begin();
}


ChannelList::ConstIterator
ChannelList::begin () const
{
    return _map.begin();
}


ChannelList::Iterator
ChannelList::end ()
{
    return _map.end();
}


ChannelList::ConstIterator
ChannelList::end () const
{
    return _map.end();
}


ChannelList::Iterator
ChannelList::find (const char name[])
{
    return _map.find (name);
}


ChannelList::ConstIterator
ChannelList::find (const char name[]) const
{
    return _map.find (name);
}


ChannelList::Iterator
ChannelList::find (const string &name)
{
    return find (name.c_str());
}


ChannelList::ConstIterator
ChannelList::find (const string &name) const
{
    return find (name.c_str());
}


void
ChannelList::layers (set <string> &layerNames) const
{
    layerNames.clear();

    for (ConstIterator i = begin(); i != end(); ++i)
    {
    string layerName = i.name();
    size_t pos = layerName.rfind ('.');

    if (pos != string::npos && pos != 0 && pos + 1 < layerName.size())
    {
        layerName.erase (pos);
        layerNames.insert (layerName);
    }
    }
}


void
ChannelList::channelsInLayer (const string &layerName,
                  Iterator &first,
                  Iterator &last)
{
    channelsWithPrefix (layerName + '.', first, last);
}


void
ChannelList::channelsInLayer (const string &layerName,
                  ConstIterator &first,
                  ConstIterator &last) const
{
    channelsWithPrefix (layerName + '.', first, last);
}


void
ChannelList::channelsWithPrefix (const char prefix[],
                 Iterator &first,
                 Iterator &last)
{
    first = last = _map.lower_bound (prefix);
    int n = strlen (prefix);

    while (last != Iterator (_map.end()) &&
       strncmp (last.name(), prefix, n) <= 0)
    {
    ++last;
    }
}


void
ChannelList::channelsWithPrefix (const char prefix[],
                 ConstIterator &first,
                 ConstIterator &last) const
{
    first = last = _map.lower_bound (prefix);
    int n = strlen (prefix);

    while (last != ConstIterator (_map.end()) &&
       strncmp (last.name(), prefix, n) <= 0)
    {
    ++last;
    }
}


void
ChannelList::channelsWithPrefix (const string &prefix,
                 Iterator &first,
                 Iterator &last)
{
    return channelsWithPrefix (prefix.c_str(), first, last);
}


void
ChannelList::channelsWithPrefix (const string &prefix,
                 ConstIterator &first,
                 ConstIterator &last) const
{
    return channelsWithPrefix (prefix.c_str(), first, last);
}


bool
ChannelList::operator == (const ChannelList &other) const
{
    ConstIterator i = begin();
    ConstIterator j = other.begin();

    while (i != end() && j != other.end())
    {
    if (!(i.channel() == j.channel()))
        return false;

    ++i;
    ++j;
    }

    return i == end() && j == other.end();
}


} // namespace Imf
