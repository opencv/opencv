//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_TEST_FILE_H
#define INCLUDED_IMF_TEST_FILE_H

//-----------------------------------------------------------------------------
//
//	Utility routines to test quickly if a given
//	file is an OpenEXR file, and whether the
//	file is scanline-based or tiled.
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


IMF_EXPORT bool isOpenExrFile (const char fileName[]);

IMF_EXPORT bool isOpenExrFile (const char fileName[],
                               bool &isTiled);

IMF_EXPORT bool isOpenExrFile (const char fileName[],
                               bool &isTiled,
                               bool &isDeep);

IMF_EXPORT bool isOpenExrFile (const char fileName[],
                               bool &isTiled,
                               bool &isDeep,
                               bool &isMultiPart);

IMF_EXPORT bool isTiledOpenExrFile (const char fileName[]);

IMF_EXPORT bool isDeepOpenExrFile (const char fileName[]);

IMF_EXPORT bool isMultiPartOpenExrFile (const char fileName[]);

IMF_EXPORT bool isOpenExrFile (IStream &is);

IMF_EXPORT bool isOpenExrFile (IStream &is,
                               bool &isTiled);

IMF_EXPORT bool isOpenExrFile (IStream &is,
                               bool &isTiled,
                               bool &isDeep);

IMF_EXPORT bool isOpenExrFile (IStream &is,
                               bool &isTiled,
                               bool &isDeep,
                               bool &isMultiPart);

IMF_EXPORT bool isTiledOpenExrFile (IStream &is);

IMF_EXPORT bool isDeepOpenExrFile (IStream &is);

IMF_EXPORT bool isMultiPartOpenExrFile (IStream &is);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
