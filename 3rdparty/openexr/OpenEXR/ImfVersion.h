//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_VERSION_H
#define INCLUDED_IMF_VERSION_H

//-----------------------------------------------------------------------------
//
//	Magic and version number.
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// The MAGIC number is stored in the first four bytes of every
// OpenEXR image file.  This can be used to quickly test whether
// a given file is an OpenEXR image file (see isImfMagic(), below).
//

static const int MAGIC = 20000630;


//
// The second item in each OpenEXR image file, right after the
// magic number, is a four-byte file version identifier.  Depending
// on a file's version identifier, a file reader can enable various
// backwards-compatibility switches, or it can quickly reject files
// that it cannot read.
//
// The version identifier is split into an 8-bit version number,
// and a 24-bit flags field.
//

static const int VERSION_NUMBER_FIELD	= 0x000000ff;
static const int VERSION_FLAGS_FIELD	= 0xffffff00;


//
// Value that goes into VERSION_NUMBER_FIELD.
//

static const int EXR_VERSION		= 2;


//
// Flags that can go into VERSION_FLAGS_FIELD.
// Flags can only occupy the 1 bits in VERSION_FLAGS_FIELD.
//

static const int TILED_FLAG		= 0x00000200;   // File is tiled
static
const int LONG_NAMES_FLAG       = 0x00000400;   // File contains long
                                                // attribute or channel
                                                // names
static
const int NON_IMAGE_FLAG        = 0x00000800;   // File has at least one part
                                                // which is not a regular
                                                // scanline image or regular tiled image
                                                // (that is, it is a deep format)
static
const int MULTI_PART_FILE_FLAG  = 0x00001000;   // File has multiple parts

//
// Bitwise OR of all known flags.
//
static
const int ALL_FLAGS		= TILED_FLAG | LONG_NAMES_FLAG |
                                  NON_IMAGE_FLAG | MULTI_PART_FILE_FLAG;


//
// Utility functions
//

inline bool  isTiled (int version)	{return !!(version & TILED_FLAG);}
inline bool  isMultiPart (int version)  {return !!(version & MULTI_PART_FILE_FLAG); }
inline bool  isNonImage(int version)    {return !!(version & NON_IMAGE_FLAG); }
inline int   makeTiled (int version)	{return version | TILED_FLAG;}
inline int   makeNotTiled (int version) {return version & ~TILED_FLAG;}
inline int   getVersion (int version)	{return version & VERSION_NUMBER_FIELD;}
inline int   getFlags (int version)	{return version & VERSION_FLAGS_FIELD;}
inline bool  supportsFlags (int flags)	{return !(flags & ~ALL_FLAGS);}


//
// Given the first four bytes of a file, returns true if the
// file is probably an OpenEXR image file, false if not.
//

IMF_EXPORT 
bool	     isImfMagic (const char bytes[4]);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
