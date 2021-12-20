//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Magic and version number.
//
//-----------------------------------------------------------------------------


#include <ImfVersion.h>
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


bool
isImfMagic (const char bytes[4])
{
    return bytes[0] == ((MAGIC >>  0) & 0x00ff) &&
	   bytes[1] == ((MAGIC >>  8) & 0x00ff) &&
	   bytes[2] == ((MAGIC >> 16) & 0x00ff) &&
	   bytes[3] == ((MAGIC >> 24) & 0x00ff);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

