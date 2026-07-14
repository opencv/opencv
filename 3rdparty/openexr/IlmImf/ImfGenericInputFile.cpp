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

#include "ImfGenericInputFile.h"

#include <ImfVersion.h>
#include <ImfXdr.h>
#include <Iex.h>
#include <OpenEXRConfig.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

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
