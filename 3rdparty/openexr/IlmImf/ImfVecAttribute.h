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



#ifndef INCLUDED_IMF_VEC_ATTRIBUTE_H
#define INCLUDED_IMF_VEC_ATTRIBUTE_H

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

#include <ImfAttribute.h>
#include "ImathVec.h"


namespace Imf {


typedef TypedAttribute<Imath::V2i> V2iAttribute;
template <> const char *V2iAttribute::staticTypeName ();
template <> void V2iAttribute::writeValueTo (OStream &, int) const;
template <> void V2iAttribute::readValueFrom (IStream &, int, int);


typedef TypedAttribute<Imath::V2f> V2fAttribute;
template <> const char *V2fAttribute::staticTypeName ();
template <> void V2fAttribute::writeValueTo (OStream &, int) const;
template <> void V2fAttribute::readValueFrom (IStream &, int, int);


typedef TypedAttribute<Imath::V2d> V2dAttribute;
template <> const char *V2dAttribute::staticTypeName ();
template <> void V2dAttribute::writeValueTo (OStream &, int) const;
template <> void V2dAttribute::readValueFrom (IStream &, int, int);


typedef TypedAttribute<Imath::V3i> V3iAttribute;
template <> const char *V3iAttribute::staticTypeName ();
template <> void V3iAttribute::writeValueTo (OStream &, int) const;
template <> void V3iAttribute::readValueFrom (IStream &, int, int);


typedef TypedAttribute<Imath::V3f> V3fAttribute;
template <> const char *V3fAttribute::staticTypeName ();
template <> void V3fAttribute::writeValueTo (OStream &, int) const;
template <> void V3fAttribute::readValueFrom (IStream &, int, int);


typedef TypedAttribute<Imath::V3d> V3dAttribute;
template <> const char *V3dAttribute::staticTypeName ();
template <> void V3dAttribute::writeValueTo (OStream &, int) const;
template <> void V3dAttribute::readValueFrom (IStream &, int, int);


} // namespace Imf

// Metrowerks compiler wants the .cpp file inlined, too
#ifdef __MWERKS__
#include <ImfVecAttribute.cpp>
#endif

#endif
