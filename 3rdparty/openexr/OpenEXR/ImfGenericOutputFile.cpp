//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfGenericOutputFile.h"

#include "ImfVersion.h"
#include "ImfIO.h"
#include "ImfXdr.h"
#include "ImfHeader.h"
#include "ImfBoxAttribute.h"
#include "ImfFloatAttribute.h"
#include "ImfTimeCodeAttribute.h"
#include "ImfChromaticitiesAttribute.h"

#include "ImfMisc.h"
#include "ImfPartType.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

    
using namespace std;

GenericOutputFile::~GenericOutputFile ()
{}

GenericOutputFile::GenericOutputFile ()
{}

void
GenericOutputFile::writeMagicNumberAndVersionField (
    OPENEXR_IMF_INTERNAL_NAMESPACE::OStream& os, const Header& header)
{
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, MAGIC);

    int version = EXR_VERSION;

    if (header.hasType() && isDeepData(header.type()))
    {
        version |= NON_IMAGE_FLAG;
    }
    else
    {
        // (TODO) we may want to check something else in function signature
        // instead of hasTileDescription()?
        if (header.hasTileDescription())
            version |= TILED_FLAG;
    }

    if (usesLongNames (header))
        version |= LONG_NAMES_FLAG;

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, version);
}

void
GenericOutputFile::writeMagicNumberAndVersionField (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream& os,
                                                    const Header * headers,
                                                    int parts)
{
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, MAGIC);

    int version = EXR_VERSION;

    if (parts == 1)
    {
        if (headers[0].type() == TILEDIMAGE)
            version |= TILED_FLAG;
    }
    else
    {
        version |= MULTI_PART_FILE_FLAG;
    }
    
    for (int i = 0; i < parts; i++)
    {
        if (usesLongNames (headers[i]))
            version |= LONG_NAMES_FLAG;

        if (headers[i].hasType() && isImage(headers[i].type()) == false)
            version |= NON_IMAGE_FLAG;
    }

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, version);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
