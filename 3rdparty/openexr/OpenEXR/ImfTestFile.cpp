//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Utility routines to test quickly if a given
//	file is an OpenEXR file, and whether the
//	file is scanline-based or tiled.
//
//-----------------------------------------------------------------------------


#include <ImfTestFile.h>
#include <ImfStdIO.h>
#include <ImfXdr.h>
#include <ImfVersion.h>
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


bool
isOpenExrFile
    (const char fileName[],
     bool &tiled,
     bool &deep,
     bool &multiPart)
{
    try
    {
	StdIFStream is (fileName);

	int magic, version;
	Xdr::read <StreamIO> (is, magic);
	Xdr::read <StreamIO> (is, version);

	tiled = isTiled (version);
        deep = isNonImage (version);
        multiPart = isMultiPart (version);
	return magic == MAGIC;
    }
    catch (...)
    {
	tiled = false;
	return false;
    }
}


bool
isOpenExrFile (const char fileName[], bool &tiled, bool &deep)
{
    bool multiPart;
    return isOpenExrFile (fileName, tiled, deep, multiPart);
}


bool
isOpenExrFile (const char fileName[], bool &tiled)
{
    bool deep, multiPart;
    return isOpenExrFile (fileName, tiled, deep, multiPart);
}


bool
isOpenExrFile (const char fileName[])
{
    bool tiled, deep, multiPart;
    return isOpenExrFile (fileName, tiled, deep, multiPart);
}


bool
isTiledOpenExrFile (const char fileName[])
{
    bool exr, tiled, deep, multiPart;
    exr = isOpenExrFile (fileName, tiled, deep, multiPart);
    return exr && tiled;
}


bool
isDeepOpenExrFile (const char fileName[])
{
    bool exr, tiled, deep, multiPart;
    exr = isOpenExrFile (fileName, tiled, deep, multiPart);
    return exr && deep;
}


bool
isMultiPartOpenExrFile (const char fileName[])
{
    bool exr, tiled, deep, multiPart;
    exr = isOpenExrFile (fileName, tiled, deep, multiPart);
    return exr && multiPart;
}


bool
isOpenExrFile
    (IStream &is,
     bool &tiled,
     bool &deep,
     bool &multiPart)
{
    try
    {
	uint64_t pos = is.tellg();

	if (pos != 0)
	    is.seekg (0);

	int magic, version;
	Xdr::read <StreamIO> (is, magic);
	Xdr::read <StreamIO> (is, version);

	is.seekg (pos);

	tiled = isTiled (version);
	deep = isNonImage (version);
	multiPart = isMultiPart (version);
	return magic == MAGIC;
    }
    catch (...)
    {
	is.clear();
	tiled = false;
	return false;
    }
}


bool
isOpenExrFile (IStream &is, bool &tiled, bool &deep)
{
    bool multiPart;
    return isOpenExrFile (is, tiled, deep, multiPart);
}


bool
isOpenExrFile (IStream &is, bool &tiled)
{
    bool deep, multiPart;
    return isOpenExrFile (is, tiled, deep, multiPart);
}


bool
isOpenExrFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is)
{
    bool tiled, deep, multiPart;
    return isOpenExrFile (is, tiled, deep, multiPart);
}


bool
isTiledOpenExrFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is)
{
    bool exr, tiled, deep, multiPart;
    exr = isOpenExrFile (is, tiled, deep, multiPart);
    return exr && tiled;
}


bool
isDeepOpenExrFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is)
{
    bool exr, tiled, deep, multiPart;
    exr = isOpenExrFile (is, tiled, deep, multiPart);
    return exr && deep;
}


bool
isMultiPartOpenExrFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is)
{
    bool exr, tiled, deep, multiPart;
    exr = isOpenExrFile (is, tiled, deep, multiPart);
    return exr && multiPart;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
