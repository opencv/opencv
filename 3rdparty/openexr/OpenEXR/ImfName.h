//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_NAME_H
#define INCLUDED_IMF_NAME_H

//-----------------------------------------------------------------------------
//
//	class ImfName -- a zero-terminated string
//	with a fixed, small maximum length
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include <cstring>

#if defined(_MSC_VER)
#pragma warning( push, 0 )
#pragma warning (disable : 4996)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class IMF_EXPORT_TYPE Name
{
  public:

    //-------------
    // Constructors
    //-------------

    Name ();
    Name (const char text[]);
    Name (const Name &) = default;
    Name (Name &&) = default;
    ~Name () = default;


    //--------------------
    // Assignment operator
    //--------------------

    Name &operator = (const Name &) = default;
    Name &operator = (Name &&) = default;
    Name &operator = (const char text[]);


    //---------------------
    // Access to the string
    //---------------------

    inline
    const char *	text () const		{return _text;}
    inline
    const char *	operator * () const	{return _text;}

    //---------------
    // Maximum length
    //---------------

    static const int	SIZE = 256;
    static const int	MAX_LENGTH = SIZE - 1;

  private:

    char		_text[SIZE];
};

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
operator == (const Name &x, const char text[])
{
    return strcmp (*x, text) == 0;
}


inline bool
operator == (const char text[], const Name &y)
{
    return strcmp (text, *y) == 0;
}


inline bool
operator != (const Name &x, const Name &y)
{
    return !(x == y);
}


inline bool
operator != (const Name &x, const char text[])
{
    return !(x == text);
}


inline bool
operator != (const char text[], const Name &y)
{
    return !(text == y);
}


inline bool
operator < (const Name &x, const Name &y)
{
    return strcmp (*x, *y) < 0;
}


inline bool
operator < (const Name &x, const char text[])
{
    return strcmp (*x, text) < 0;
}


inline bool
operator < (const char text[], const Name &y)
{
    return strcmp (text, *y) < 0;
}


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#if defined(_MSC_VER)
#pragma warning (pop)
#endif

#endif
