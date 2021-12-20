//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfGenericInputFile.h"

#include <ImfIO.h>
#include <ImfVersion.h>
#include <ImfXdr.h>
#include <Iex.h>
#include <OpenEXRConfig.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

GenericInputFile::~GenericInputFile ()
{}

GenericInputFile::GenericInputFile ()
{}

void GenericInputFile::readMagicNumberAndVersionField(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is, int& version)
{
    //
    // Read the magic number and the file format version number.
    // Then check if we can read the rest of this file.
    //

    int magic;

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, magic);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, version);

    if (magic != MAGIC)
    {
        throw IEX_NAMESPACE::InputExc ("File is not an image file.");
    }

    if (getVersion (version) != EXR_VERSION)
    {
        THROW (IEX_NAMESPACE::InputExc, "Cannot read "
                              "version " << getVersion (version) << " "
                              "image files.  Current file format version "
                              "is " << EXR_VERSION << ".");
    }

    if (!supportsFlags (getFlags (version)))
    {
        THROW (IEX_NAMESPACE::InputExc, "The file format version number's flag field "
                              "contains unrecognized flags.");
    }
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
