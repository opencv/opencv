//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_CHANNEL_LIST_ATTRIBUTE_H
#define INCLUDED_IMF_CHANNEL_LIST_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class ChannelListAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include "ImfChannelList.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::ChannelList> ChannelListAttribute;

#ifndef COMPILING_IMF_CHANNEL_LIST_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::ChannelList>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif

