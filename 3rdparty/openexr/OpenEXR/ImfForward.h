//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_FORWARD_H
#define INCLUDED_IMF_FORWARD_H

////////////////////////////////////////////////////////////////////
//
// Forward declarations for OpenEXR - correctly declares namespace
//
////////////////////////////////////////////////////////////////////

#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


// classes for basic types;
template<class T> class IMF_EXPORT_TEMPLATE_TYPE Array;
template<class T> class IMF_EXPORT_TEMPLATE_TYPE Array2D;
struct IMF_EXPORT_TYPE Channel;
class  IMF_EXPORT_TYPE ChannelList;
struct IMF_EXPORT_TYPE Chromaticities;

// attributes used in headers are TypedAttributes
class IMF_EXPORT_TYPE Attribute;

class IMF_EXPORT_TYPE Header;

// file handling classes
class IMF_EXPORT_TYPE OutputFile;
class IMF_EXPORT_TYPE TiledInputFile;
class IMF_EXPORT_TYPE ScanLineInputFile;
class IMF_EXPORT_TYPE InputFile;
class IMF_EXPORT_TYPE TiledOutputFile;
class IMF_EXPORT_TYPE DeepScanLineInputFile;
class IMF_EXPORT_TYPE DeepScanLineOutputFile;
class IMF_EXPORT_TYPE DeepTiledInputFile;
class IMF_EXPORT_TYPE DeepTiledOutputFile;
class IMF_EXPORT_TYPE AcesInputFile;
class IMF_EXPORT_TYPE AcesOutputFile;
class IMF_EXPORT_TYPE TiledInputPart;
class IMF_EXPORT_TYPE TiledInputFile;
class IMF_EXPORT_TYPE TileOffsets;

// multipart file handling
class IMF_EXPORT_TYPE GenericInputFile;
class IMF_EXPORT_TYPE GenericOutputFile;
class IMF_EXPORT_TYPE MultiPartInputFile;
class IMF_EXPORT_TYPE MultiPartOutputFile;

class IMF_EXPORT_TYPE InputPart;
class IMF_EXPORT_TYPE TiledInputPart;
class IMF_EXPORT_TYPE DeepScanLineInputPart;
class IMF_EXPORT_TYPE DeepTiledInputPart;

class IMF_EXPORT_TYPE OutputPart;
class IMF_EXPORT_TYPE ScanLineOutputPart;
class IMF_EXPORT_TYPE TiledOutputPart;
class IMF_EXPORT_TYPE DeepScanLineOutputPart;
class IMF_EXPORT_TYPE DeepTiledOutputPart;


// internal use only
struct InputPartData;
struct OutputStreamMutex;
struct OutputPartData;
struct InputStreamMutex;

// frame buffers

class  IMF_EXPORT_TYPE FrameBuffer;
class  IMF_EXPORT_TYPE DeepFrameBuffer;
struct IMF_EXPORT_TYPE DeepSlice;

// compositing
class IMF_EXPORT_TYPE DeepCompositing;
class IMF_EXPORT_TYPE CompositeDeepScanLine;

// preview image
class IMF_EXPORT_TYPE PreviewImage;
struct IMF_EXPORT_TYPE PreviewRgba;

// streams
class IMF_EXPORT_TYPE OStream;
class IMF_EXPORT_TYPE IStream;

class IMF_EXPORT_TYPE IDManifest;
class IMF_EXPORT_TYPE CompressedIDManifest;


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif // include guard
