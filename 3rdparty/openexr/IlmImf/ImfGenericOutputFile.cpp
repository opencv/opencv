///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
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

#include "ImfGenericOutputFile.h"

#include <ImfBoxAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfTimeCodeAttribute.h>
#include <ImfChromaticitiesAttribute.h>

#include <ImfMisc.h>
#include <ImfPartType.h>

#include "ImfNamespace.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

    
using namespace std;


    
void
GenericOutputFile::writeMagicNumberAndVersionField (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream& os,
                                                    const Header& header)
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
