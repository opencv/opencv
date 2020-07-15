///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
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
//	class M33fAttribute
//	class M33dAttribute
//	class M44fAttribute
//	class M44dAttribute
//
//-----------------------------------------------------------------------------

#include <ImfMatrixAttribute.h>


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;


template <>
const char *
M33fAttribute::staticTypeName ()
{
    return "m33f";
}


template <>
void
M33fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
}


template <>
void
M33fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
}


template <>
const char *
M33dAttribute::staticTypeName ()
{
    return "m33d";
}


template <>
void
M33dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
}


template <>
void
M33dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
}


template <>
const char *
M44fAttribute::staticTypeName ()
{
    return "m44f";
}


template <>
void
M44fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);
    Xdr::write <StreamIO> (os, _value[0][3]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);
    Xdr::write <StreamIO> (os, _value[1][3]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
    Xdr::write <StreamIO> (os, _value[2][3]);

    Xdr::write <StreamIO> (os, _value[3][0]);
    Xdr::write <StreamIO> (os, _value[3][1]);
    Xdr::write <StreamIO> (os, _value[3][2]);
    Xdr::write <StreamIO> (os, _value[3][3]);
}


template <>
void
M44fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);
    Xdr::read <StreamIO> (is, _value[0][3]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);
    Xdr::read <StreamIO> (is, _value[1][3]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
    Xdr::read <StreamIO> (is, _value[2][3]);

    Xdr::read <StreamIO> (is, _value[3][0]);
    Xdr::read <StreamIO> (is, _value[3][1]);
    Xdr::read <StreamIO> (is, _value[3][2]);
    Xdr::read <StreamIO> (is, _value[3][3]);
}


template <>
const char *
M44dAttribute::staticTypeName ()
{
    return "m44d";
}


template <>
void
M44dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);
    Xdr::write <StreamIO> (os, _value[0][3]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);
    Xdr::write <StreamIO> (os, _value[1][3]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
    Xdr::write <StreamIO> (os, _value[2][3]);

    Xdr::write <StreamIO> (os, _value[3][0]);
    Xdr::write <StreamIO> (os, _value[3][1]);
    Xdr::write <StreamIO> (os, _value[3][2]);
    Xdr::write <StreamIO> (os, _value[3][3]);
}


template <>
void
M44dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);
    Xdr::read <StreamIO> (is, _value[0][3]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);
    Xdr::read <StreamIO> (is, _value[1][3]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
    Xdr::read <StreamIO> (is, _value[2][3]);

    Xdr::read <StreamIO> (is, _value[3][0]);
    Xdr::read <StreamIO> (is, _value[3][1]);
    Xdr::read <StreamIO> (is, _value[3][2]);
    Xdr::read <StreamIO> (is, _value[3][3]);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
