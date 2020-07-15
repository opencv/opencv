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
//	class V2iAttribute
//	class V2fAttribute
//	class V2dAttribute
//	class V3iAttribute
//	class V3fAttribute
//	class V3dAttribute
//
//-----------------------------------------------------------------------------

#include <ImfVecAttribute.h>


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
const char *
V2iAttribute::staticTypeName ()
{
    return "v2i";
}


template <>
void
V2iAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
}


template <>
void
V2iAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
}


template <>
const char *
V2fAttribute::staticTypeName ()
{
    return "v2f";
}


template <>
void
V2fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
}


template <>
void
V2fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
}


template <>
const char *
V2dAttribute::staticTypeName ()
{
    return "v2d";
}


template <>
void
V2dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
}


template <>
void
V2dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
}


template <>
const char *
V3iAttribute::staticTypeName ()
{
    return "v3i";
}


template <>
void
V3iAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
    Xdr::write <StreamIO> (os, _value.z);
}


template <>
void
V3iAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
    Xdr::read <StreamIO> (is, _value.z);
}


template <>
const char *
V3fAttribute::staticTypeName ()
{
    return "v3f";
}


template <>
void
V3fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
    Xdr::write <StreamIO> (os, _value.z);
}


template <>
void
V3fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
    Xdr::read <StreamIO> (is, _value.z);
}


template <>
const char *
V3dAttribute::staticTypeName ()
{
    return "v3d";
}


template <>
void
V3dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
    Xdr::write <StreamIO> (os, _value.z);
}


template <>
void
V3dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
    Xdr::read <StreamIO> (is, _value.z);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
