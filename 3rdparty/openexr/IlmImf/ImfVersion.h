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

const int MAGIC = 20000630;


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

const int VERSION_NUMBER_FIELD	= 0x000000ff;
const int VERSION_FLAGS_FIELD	= 0xffffff00;


//
// Value that goes into VERSION_NUMBER_FIELD.
//

const int EXR_VERSION		= 2;


//
// Flags that can go into VERSION_FLAGS_FIELD.
// Flags can only occupy the 1 bits in VERSION_FLAGS_FIELD.
//

const int TILED_FLAG		= 0x00000200;   // File is tiled

const int LONG_NAMES_FLAG       = 0x00000400;   // File contains long
                                                // attribute or channel
                                                // names

const int NON_IMAGE_FLAG        = 0x00000800;   // File has at least one part
                                                // which is not a regular
                                                // scanline image or regular tiled image
                                                // (that is, it is a deep format)

const int MULTI_PART_FILE_FLAG  = 0x00001000;   // File has multiple parts

//
// Bitwise OR of all known flags.
//

const int ALL_FLAGS		= TILED_FLAG | LONG_NAMES_FLAG |
                                  NON_IMAGE_FLAG | MULTI_PART_FILE_FLAG;


//
// Utility functions
//

inline bool  isTiled (int version)	{return !!(version & TILED_FLAG);}
inline bool  isMultiPart (int version)  {return version & MULTI_PART_FILE_FLAG; }
inline bool  isNonImage(int version)    {return version & NON_IMAGE_FLAG; }
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
