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



//-----------------------------------------------------------------------------
//
//	class OpaqueAttribute
//
//-----------------------------------------------------------------------------

#include <ImfOpaqueAttribute.h>
#include "Iex.h"
#include <string.h>

namespace Imf {


OpaqueAttribute::OpaqueAttribute (const char typeName[]):
    _typeName (strlen (typeName) + 1),
    _dataSize (0)
{
    strcpy (_typeName, typeName);
}


OpaqueAttribute::OpaqueAttribute (const OpaqueAttribute &other):
    _typeName (strlen (other._typeName) + 1),
    _dataSize (other._dataSize),
    _data (other._dataSize)
{
    strcpy (_typeName, other._typeName);
    _data.resizeErase (other._dataSize);
    memcpy ((char *) _data, (const char *) other._data, other._dataSize);
}


OpaqueAttribute::~OpaqueAttribute ()
{
    // empty
}


const char *
OpaqueAttribute::typeName () const
{
    return _typeName;
}


Attribute *
OpaqueAttribute::copy () const
{
    return new OpaqueAttribute (*this);
}


void
OpaqueAttribute::writeValueTo (OStream &os, int) const
{
    Xdr::write <StreamIO> (os, _data, _dataSize);
}


void
OpaqueAttribute::readValueFrom (IStream &is, int size, int)
{
    _data.resizeErase (size);
    _dataSize = size;
    Xdr::read <StreamIO> (is, _data, size);
}


void
OpaqueAttribute::copyValueFrom (const Attribute &other)
{
    const OpaqueAttribute *oa = dynamic_cast <const OpaqueAttribute *> (&other);

    if (oa == 0 || strcmp (_typeName, oa->_typeName))
    {
    THROW (Iex::TypeExc, "Cannot copy the value of an "
                 "image file attribute of type "
                 "\"" << other.typeName() << "\" "
                 "to an attribute of type "
                 "\"" << _typeName << "\".");
    }

    _data.resizeErase (oa->_dataSize);
    _dataSize = oa->_dataSize;
    memcpy ((char *) _data, (const char *) oa->_data, oa->_dataSize);
}


} // namespace Imf
