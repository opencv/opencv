

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// Portions (c) 2012 Weta Digital Ltd
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

#ifndef INCLUDED_IMF_FORWARD_H
#define INCLUDED_IMF_FORWARD_H

////////////////////////////////////////////////////////////////////
//
// Forward declarations for OpenEXR - correctly declares namespace
//
////////////////////////////////////////////////////////////////////

#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


// classes for basic types;
template<class T> class Array;
template<class T> class Array2D;
struct Channel;
class  ChannelList;
struct Chromaticities;

// attributes used in headers are TypedAttributes
class Attribute;

class Header;

// file handling classes
class OutputFile;
class TiledInputFile;
class ScanLineInputFile;
class InputFile;
class TiledOutputFile;
class DeepScanLineInputFile;
class DeepScanLineOutputFile;
class DeepTiledInputFile;
class DeepTiledOutputFile;
class AcesInputFile;
class AcesOutputFile;
class TiledInputPart;
class TiledInputFile;
class TileOffsets;

// multipart file handling
class GenericInputFile;
class GenericOutputFile;
class MultiPartInputFile;
class MultiPartOutputFile;

class InputPart;
class TiledInputPart;
class DeepScanLineInputPart;
class DeepTiledInputPart;

class OutputPart;
class ScanLineOutputPart;
class TiledOutputPart;
class DeepScanLineOutputPart;
class DeepTiledOutputPart;


// internal use only
struct InputPartData;
struct OutputStreamMutex;
struct OutputPartData;
struct InputStreamMutex;

// frame buffers

class  FrameBuffer;
class  DeepFrameBuffer;
struct DeepSlice;

// compositing
class DeepCompositing;
class CompositeDeepScanLine;

// preview image
class PreviewImage;
struct PreviewRgba;

// streams
class OStream;
class IStream;


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif // include guard
