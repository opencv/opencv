///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2007, Weta Digital Ltd
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
// *       Neither the name of Weta Digital nor the names of
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
//	class StringAttribute
//
//-----------------------------------------------------------------------------

#include <ImfStringVectorAttribute.h>


namespace Imf {


template <>
const char *
StringVectorAttribute::staticTypeName ()
{
    return "stringvector";
}


template <>
void
StringVectorAttribute::writeValueTo (OStream &os, int version) const
{
    int size = _value.size();

    for (int i = 0; i < size; i++)
    {
        int strSize = _value[i].size();
        Xdr::write <StreamIO> (os, strSize);
	Xdr::write <StreamIO> (os, &_value[i][0], strSize);
    }
}


template <>
void
StringVectorAttribute::readValueFrom (IStream &is, int size, int version)
{
    int read = 0;

    while (read < size)
    {   
       int strSize;
       Xdr::read <StreamIO> (is, strSize);
       read += Xdr::size<int>();       

       std::string str;
       str.resize (strSize);
  
       Xdr::read<StreamIO> (is, &str[0], strSize);
       read += strSize;

       _value.push_back (str);
    }
}


} // namespace Imf
