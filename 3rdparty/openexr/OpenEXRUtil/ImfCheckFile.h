// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.

#ifndef INCLUDED_IMF_CHECKFILE_H
#define INCLUDED_IMF_CHECKFILE_H

#include "ImfNamespace.h"
#include "ImfUtilExport.h"

#include <cstddef>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// attempt to read the given file as an OpenEXR, using various OpenEXR read paths.
// This can be used to validate correctness of the library, when running the library
// with a sanitizer or memory checker, as well as checking that a file is a correct OpenEXR
//
// returns true if the file reads correctly using expected API calls, or false
// if an exception was thrown that indicates the file is invalid
//
// if reduceMemory is true, will avoid tests or inputs that are known to
// take large amounts of memory. This may hide errors within the file or library.
//
// if reduceTime is true and an error is found within the file, then future tests are reduced for speed.
// This may hide errors within the library.
//
// if runCoreCheck is true, only uses the OpenEXRCore (C) API, otherwise uses the OpenEXR (C++) API
//

IMFUTIL_EXPORT bool checkOpenEXRFile (
    const char* fileName,
    bool        reduceMemory    = false,
    bool        reduceTime      = false,
    bool        runCoreCheck = false);

//
// overloaded version of checkOpenEXRFile that takes a pointer to in-memory data
//

IMFUTIL_EXPORT bool checkOpenEXRFile (
    const char* data,
    size_t      numBytes,
    bool        reduceMemory    = false,
    bool        reduceTime      = false,
    bool        runCoreCheck = false);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
