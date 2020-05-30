///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2007, Weta Digital Ltd
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
// *       Neither the name of Weta Digital nor the names of
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


#ifndef INCLUDED_IMF_MULTIVIEW_H
#define INCLUDED_IMF_MULTIVIEW_H

#include "ImfChannelList.h"
#include "ImfStringVectorAttribute.h"
#include "ImfExport.h"
#include "ImfNamespace.h"

//-----------------------------------------------------------------------------
//
//	Functions related to accessing channels and views in multi-view
//	OpenEXR files.
//
//	A multi-view image file contains two or more views of the same
//	scene, as seen from different viewpoints, for example, a left-eye
//	and a right-eye view for stereo displays.  Each view has its own
//	set of image channels.  A naming convention identifies the channels
//	that belong to a given view.
//
//	A "multiView" attribute in the file header lists the names of the
//	views in an image (see ImfStandardAttributes.h), and channel names
//	of the form
//
//		layer.view.channel
//
//	allow channels to be matched with views.
//
//	For compatibility with singe-view images, the first view listed in
//	the multiView attribute is the "default view", and channels that
//	have no periods in their names are considered part of the default
//	view.
//
//	For example, if a file's multiView attribute lists the views
//	"left" and "right", in that order, then "left" is the default
//	view.  Channels
//
//		"R", "left.Z", "diffuse.left.R"
//
//	are part of the "left" view; channels
//
//		"right.R", "right.Z", "diffuse.right.R"
//
//	are part of the "right" view; and channels
//
//		"tmp.R", "right.diffuse.R", "diffuse.tmp.R"
//
//	belong to no view at all.
//
//-----------------------------------------------------------------------------

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Return the name of the default view given a multi-view string vector,
// that is, return the first element of the string vector.  If the string
// vector is empty, return "".
//

IMF_EXPORT
std::string defaultViewName (const StringVector &multiView);


//
// Given the name of a channel, return the name of the view to
// which it belongs.  Returns the empty string ("") if the channel
// is not a member of any named view.
//

IMF_EXPORT
std::string viewFromChannelName (const std::string &channel,
                                 const StringVector &multiView);


//
// Return whether channel1 and channel2 are the same channel but
// viewed in different views.  (Return false if either channel
// belongs to no view or if both channels belong to the same view.)
//

IMF_EXPORT
bool areCounterparts (const std::string &channel1,
                      const std::string &channel2,
                      const StringVector &multiView);

//
// Return a list of all channels belonging to view viewName.
//

IMF_EXPORT
ChannelList channelsInView (const std::string &viewName,
                            const ChannelList &channelList,
                            const StringVector &multiView);

//
// Return a list of channels not associated with any view.
//

IMF_EXPORT
ChannelList channelsInNoView (const ChannelList &channelList,
                              const StringVector &multiView);

//
// Given the name of a channel, return a list of the same channel
// in all views (for example, given X.left.Y return X.left.Y,
// X.right.Y, X.centre.Y, etc.).
//

IMF_EXPORT
ChannelList channelInAllViews (const std::string &channame,
                               const ChannelList &channelList,
                               const StringVector &multiView);

//
// Given the name of a channel in one view, return the corresponding
// channel name for view otherViewName.  Return "" if no corresponding
// channel exists in view otherViewName, or if view otherViewName doesn't
// exist.
//

IMF_EXPORT
std::string channelInOtherView (const std::string &channel,
                                const ChannelList &channelList,
                                const StringVector &multiView,
                                const std::string &otherViewName);

//
// Given a channel name that does not include a view name, insert
// multiView[i] into the channel name at the appropriate location.
// If i is zero and the channel name contains no periods, then do
// not insert the view name.
//

IMF_EXPORT
std::string insertViewName (const std::string &channel,
			    const StringVector &multiView,
			    int i);

//
// Given a channel name that does may include a view name, return
// string without the view name. If the string does not contain
// the view name, return the string unaltered.
// (Will only remove the viewname if it is in the correct position 
//  in the string)
//

IMF_EXPORT
std::string removeViewName (const std::string &channel,
		            const std::string &view);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
