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



#ifndef INCLUDED_IMF_NAME_H
#define INCLUDED_IMF_NAME_H

//-----------------------------------------------------------------------------
//
//	class ImfName -- a zero-terminated string
//	with a fixed, small maximum length
//
//-----------------------------------------------------------------------------

#include <string.h>
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class Name
{
  public:

    //-------------
    // Constructors
    //-------------

    IMF_EXPORT
    Name ();
    IMF_EXPORT
    Name (const char text[]);


    //--------------------
    // Assignment operator
    //--------------------

    IMF_EXPORT
    Name &		operator = (const char text[]);


    //---------------------
    // Access to the string
    //---------------------

    IMF_EXPORT
    const char *	text () const		{return _text;}
    IMF_EXPORT
    const char *	operator * () const	{return _text;}

    //---------------
    // Maximum length
    //---------------

    static const int	SIZE = 256;
    static const int	MAX_LENGTH = SIZE - 1;

  private:

    char		_text[SIZE];
};


IMF_EXPORT
bool operator == (const Name &x, const Name &y);
IMF_EXPORT
bool operator != (const Name &x, const Name &y);
IMF_EXPORT
bool operator < (const Name &x, const Name &y);


//-----------------
// Inline functions
//-----------------

inline Name &
Name::operator = (const char text[])
{
    strncpy (_text, text, MAX_LENGTH);
    return *this;
}


inline
Name::Name ()
{
    _text[0] = 0;
}


inline
Name::Name (const char text[])
{
    *this = text;
    _text [MAX_LENGTH] = 0;
}


inline bool
operator == (const Name &x, const Name &y)
{
    return strcmp (*x, *y) == 0;
}


inline bool
operator != (const Name &x, const Name &y)
{
    return !(x == y);
}


inline bool
operator < (const Name &x, const Name &y)
{
    return strcmp (*x, *y) < 0;
}


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT




#endif
