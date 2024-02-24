//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_IMAGE_CHANNEL_RENAMING_H
#define INCLUDED_IMF_IMAGE_CHANNEL_RENAMING_H

//----------------------------------------------------------------------------
//
//      typedef RenamingMap,
//      helper functions for image channel renaming.
//
//----------------------------------------------------------------------------

#include "ImfNamespace.h"
#include <map>
#include <string>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Given a map from old channel names to new channel names,
// rename the channels in an image or image level.
// This function assumes that the channel old-to-new-names
// map has already been checked for collisions.
//

typedef std::map<std::string, std::string> RenamingMap;

template <class ChannelMap>
inline void
renameChannelsInMap (const RenamingMap& oldToNewNames, ChannelMap& channels)
{
    ChannelMap renamedChannels;

    for (typename ChannelMap::const_iterator i = channels.begin ();
         i != channels.end ();
         ++i)
    {
        RenamingMap::const_iterator j = oldToNewNames.find (i->first);
        std::string newName           = (j == oldToNewNames.end ()) ? i->first
                                                                    : j->second;
        renamedChannels[newName]      = i->second;
    }

    channels = renamedChannels;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
