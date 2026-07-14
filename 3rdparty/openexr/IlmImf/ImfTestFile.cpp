///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
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
	Int64 pos = is.tellg();

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
