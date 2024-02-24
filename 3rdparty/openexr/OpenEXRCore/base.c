/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "openexr_base.h"
#include "openexr_errors.h"
#include "openexr_version.h"

/**************************************/

void
exr_get_library_version (int* maj, int* min, int* patch, const char** extra)
{
    if (maj) *maj = OPENEXR_VERSION_MAJOR;
    if (min) *min = OPENEXR_VERSION_MINOR;
    if (patch) *patch = OPENEXR_VERSION_PATCH;
#ifdef OPENEXR_VERSION_EXTRA
    if (extra) *extra = OPENEXR_VERSION_EXTRA;
#else
    if (extra) *extra = "";
#endif
}

/**************************************/

static const char* the_error_code_names[] = {
    "EXR_ERR_SUCCESS",
    "EXR_ERR_OUT_OF_MEMORY",
    "EXR_ERR_MISSING_CONTEXT_ARG",
    "EXR_ERR_INVALID_ARGUMENT",
    "EXR_ERR_ARGUMENT_OUT_OF_RANGE",
    "EXR_ERR_FILE_ACCESS",
    "EXR_ERR_FILE_BAD_HEADER",
    "EXR_ERR_NOT_OPEN_READ",
    "EXR_ERR_NOT_OPEN_WRITE",
    "EXR_ERR_HEADER_NOT_WRITTEN",
    "EXR_ERR_READ_IO",
    "EXR_ERR_WRITE_IO",
    "EXR_ERR_NAME_TOO_LONG",
    "EXR_ERR_MISSING_REQ_ATTR",
    "EXR_ERR_INVALID_ATTR",
    "EXR_ERR_NO_ATTR_BY_NAME",
    "EXR_ERR_ATTR_TYPE_MISMATCH",
    "EXR_ERR_ATTR_SIZE_MISMATCH",
    "EXR_ERR_SCAN_TILE_MIXEDAPI",
    "EXR_ERR_TILE_SCAN_MIXEDAPI",
    "EXR_ERR_MODIFY_SIZE_CHANGE",
    "EXR_ERR_ALREADY_WROTE_ATTRS",
    "EXR_ERR_BAD_CHUNK_LEADER",
    "EXR_ERR_CORRUPT_CHUNK",
    "EXR_ERR_INCORRECT_PART",
    "EXR_ERR_INCORRECT_CHUNK",
    "EXR_ERR_USE_SCAN_DEEP_WRITE",
    "EXR_ERR_USE_TILE_DEEP_WRITE",
    "EXR_ERR_USE_SCAN_NONDEEP_WRITE",
    "EXR_ERR_USE_TILE_NONDEEP_WRITE",
    "EXR_ERR_INVALID_SAMPLE_DATA",
    "EXR_ERR_FEATURE_NOT_IMPLEMENTED",
    "EXR_ERR_UNKNOWN"};
static int the_error_code_count =
    sizeof (the_error_code_names) / sizeof (const char*);

/**************************************/

static const char* the_default_errors[] = {
    "Success",
    "Unable to allocate memory",
    "Context argument to function is not valid",
    "Invalid argument to function",
    "Argument to function out of valid range",
    "Unable to open file (path does not exist or permission denied)",
    "File is not an OpenEXR file or has a bad header value",
    "File not opened for read",
    "File not opened for write",
    "File opened for write, but header not yet written",
    "Error reading from stream",
    "Error writing to stream",
    "Text too long for file flags",
    "Missing required attribute in part header",
    "Invalid attribute in part header",
    "No attribute by that name in part header",
    "Attribute type mismatch",
    "Attribute type vs. size mismatch",
    "Attempt to use a scanline accessor function for a tiled image",
    "Attempt to use a tiled accessor function for a scanline image",
    "Attempt to modify a value when in update mode with different size",
    "File in write mode, but header already written, can no longer edit attributes",
    "Unexpected or corrupt values in data block leader vs computed value",
    "Corrupt data block data, unable to decode",
    "Previous part not yet finished writing",
    "Invalid data block to write at this point",
    "Use deep scanline write with the sample count table arguments",
    "Use deep tile write with the sample count table arguments",
    "Use non-deep scanline write (sample count table invalid for this part type)",
    "Use non-deep tile write (sample count table invalid for this part type)",
    "Invalid sample data table value",
    "Feature not yet implemented, please use C++ library",
    "Unknown error code"};
static int the_default_error_count =
    sizeof (the_default_errors) / sizeof (const char*);

/**************************************/

const char*
exr_get_default_error_message (exr_result_t code)
{
    int idx = (int) code;
    if (idx < 0 || idx >= the_default_error_count)
        idx = the_default_error_count - 1;
    return the_default_errors[idx];
}

/**************************************/

const char*
exr_get_error_code_as_string (exr_result_t code)
{
    int idx = (int) code;
    if (idx < 0 || idx >= the_error_code_count) idx = the_error_code_count - 1;
    return the_error_code_names[idx];
}

/**************************************/

static int sMaxW = 0;
static int sMaxH = 0;

void
exr_set_default_maximum_image_size (int w, int h)
{
    if (w >= 0 && h >= 0)
    {
        sMaxW = w;
        sMaxH = h;
    }
}

/**************************************/

void
exr_get_default_maximum_image_size (int* w, int* h)
{
    if (w) *w = sMaxW;
    if (h) *h = sMaxH;
}

/**************************************/

static int sTileMaxW = 0;
static int sTileMaxH = 0;

void
exr_set_default_maximum_tile_size (int w, int h)
{
    if (w >= 0 && h >= 0)
    {
        sTileMaxW = w;
        sTileMaxH = h;
    }
}

/**************************************/

void
exr_get_default_maximum_tile_size (int* w, int* h)
{
    if (w) *w = sTileMaxW;
    if (h) *h = sTileMaxH;
}

/**************************************/

static int sDefaultZipLevel = -1;

void
exr_set_default_zip_compression_level (int l)
{
    if (l < 0) l = -1;
    if (l > 9) l = 9;
    sDefaultZipLevel = l;
}

/**************************************/

void
exr_get_default_zip_compression_level (int* l)
{
    if (l) *l = sDefaultZipLevel;
}

/**************************************/

static float sDefaultDwaLevel = 45.f;

void
exr_set_default_dwa_compression_quality (float q)
{
    if (q < 0.f) q = 0.f;
    if (q > 100.f) q = 100.f;
    sDefaultDwaLevel = q;
}

/**************************************/

void
exr_get_default_dwa_compression_quality (float* q)
{
    if (q) *q = sDefaultDwaLevel;
}
