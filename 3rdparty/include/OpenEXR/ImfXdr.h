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


#ifndef INCLUDED_IMF_XDR_H
#define INCLUDED_IMF_XDR_H

//----------------------------------------------------------------------------
//
//	Xdr -- routines to convert data between the machine's native
//	format and a machine-independent external data representation:
//
//	    write<R> (T &o, S v);	converts a value, v, of type S
//					into a machine-independent
//					representation and stores the
//					result in an output buffer, o.
//
//	    read<R> (T &i, S &v);	reads the machine-independent
//					representation of a value of type
//					S from input buffer i, converts
//					the value into the machine's native
//					representation, and stores the result
//					in v.
//
//	    size<S>();			returns the size, in bytes, of the
//					machine-independent representation
//					of an object of type S.
//					
//	The write() and read() routines are templates; data can be written
//	to and read from any output or input buffer type T for which a helper
//	class, R, exits.  Class R must define a method to store a char array
//	in a T, and a method to read a char array from a T:
//
//	    struct R
//	    {
//	        static void
//	        writeChars (T &o, const char c[/*n*/], int n)
//	        {
//	            ... // Write c[0], c[1] ... c[n-1] to output buffer o.
//	        }
//
//	        static void
//	        readChars (T &i, char c[/*n*/], int n)
//	        {
//	            ... // Read n characters from input buffer i
//		        // and copy them to c[0], c[1] ... c[n-1].
//	        }
//	    };
//
//	Example - writing to and reading from iostreams:
//
//	    struct CharStreamIO
//	    {
//	        static void
//	        writeChars (ostream &os, const char c[], int n)
//	        {
//	            os.write (c, n);
//	        }
//
//	        static void
//	        readChars (istream &is, char c[], int n)
//	        {
//	            is.read (c, n);
//	        }
//	    };
//
//          ...
//
//	    Xdr::write<CharStreamIO> (os, 3);
//	    Xdr::write<CharStreamIO> (os, 5.0);
//
//----------------------------------------------------------------------------

#include <ImfInt64.h>
#include "IexMathExc.h"
#include "half.h"
#include <limits.h>

namespace Imf {
namespace Xdr {


//-------------------------------
// Write data to an output stream
//-------------------------------

template <class S, class T>
void
write (T &out, bool v);

template <class S, class T>
void
write (T &out, char v);

template <class S, class T>
void
write (T &out, signed char v);

template <class S, class T>
void
write (T &out, unsigned char v);

template <class S, class T>
void
write (T &out, signed short v);

template <class S, class T>
void
write (T &out, unsigned short v);

template <class S, class T>
void
write (T &out, signed int v);

template <class S, class T>
void
write (T &out, unsigned int v);

template <class S, class T>
void
write (T &out, signed long v);

template <class S, class T>
void
write (T &out, unsigned long v);

#if ULONG_MAX != 18446744073709551615LU

    template <class S, class T>
    void
    write (T &out, Int64 v);

#endif

template <class S, class T>
void
write (T &out, float v);

template <class S, class T>
void
write (T &out, double v);

template <class S, class T>
void
write (T &out, half v);

template <class S, class T>
void
write (T &out, const char v[/*n*/], int n);	// fixed-size char array

template <class S, class T>
void
write (T &out, const char v[]);			// zero-terminated string


//-----------------------------------------
// Append padding bytes to an output stream
//-----------------------------------------

template <class S, class T>
void
pad (T &out, int n);				// write n padding bytes



//-------------------------------
// Read data from an input stream
//-------------------------------

template <class S, class T>
void
read (T &in, bool &v);

template <class S, class T>
void
read (T &in, char &v);

template <class S, class T>
void
read (T &in, signed char &v);

template <class S, class T>
void
read (T &in, unsigned char &v);

template <class S, class T>
void
read (T &in, signed short &v);

template <class S, class T>
void
read (T &in, unsigned short &v);

template <class S, class T>
void
read (T &in, signed int &v);

template <class S, class T>
void
read (T &in, unsigned int &v);

template <class S, class T>
void
read (T &in, signed long &v);

template <class S, class T>
void
read (T &in, unsigned long &v);

#if ULONG_MAX != 18446744073709551615LU

    template <class S, class T>
    void
    read (T &in, Int64 &v);

#endif

template <class S, class T>
void
read (T &in, float &v);

template <class S, class T>
void
read (T &in, double &v);

template <class S, class T>
void
read (T &in, half &v);

template <class S, class T>
void
read (T &in, char v[/*n*/], int n);		// fixed-size char array

template <class S, class T>
void
read (T &in, int n, char v[/*n*/]);		// zero-terminated string


//-------------------------------------------
// Skip over padding bytes in an input stream
//-------------------------------------------

template <class S, class T>
void
skip (T &in, int n);				// skip n padding bytes



//--------------------------------------
// Size of the machine-independent
// representation of an object of type S
//--------------------------------------

template <class S>
int
size ();


//---------------
// Implementation
//---------------

template <class S, class T>
inline void
writeSignedChars (T &out, const signed char c[], int n)
{
    S::writeChars (out, (const char *) c, n);
}


template <class S, class T>
inline void
writeUnsignedChars (T &out, const unsigned char c[], int n)
{
    S::writeChars (out, (const char *) c, n);
}


template <class S, class T>
inline void
readSignedChars (T &in, signed char c[], int n)
{
    S::readChars (in, (char *) c, n);
}


template <class S, class T>
inline void
readUnsignedChars (T &in, unsigned char c[], int n)
{
    S::readChars (in, (char *) c, n);
}


template <class S, class T>
inline void
write (T &out, bool v)
{
    char c = !!v;
    S::writeChars (out, &c, 1);
}


template <class S, class T>
inline void
write (T &out, char v)
{
    S::writeChars (out, &v, 1);
}


template <class S, class T>
inline void
write (T &out, signed char v)
{
    writeSignedChars<S> (out, &v, 1);
}


template <class S, class T>
inline void
write (T &out, unsigned char v)
{
    writeUnsignedChars<S> (out, &v, 1);
}


template <class S, class T>
void
write (T &out, signed short v)
{
    signed char b[2];

    b[0] =  (signed char) (v);
    b[1] =  (signed char) (v >> 8);

    writeSignedChars<S> (out, b, 2);
}


template <class S, class T>
void
write (T &out, unsigned short v)
{
    unsigned char b[2];

    b[0] =  (unsigned char) (v);
    b[1] =  (unsigned char) (v >> 8);

    writeUnsignedChars<S> (out, b, 2);
}


template <class S, class T>
void
write (T &out, signed int v)
{
    signed char b[4];

    b[0] =  (signed char) (v);
    b[1] =  (signed char) (v >> 8);
    b[2] =  (signed char) (v >> 16);
    b[3] =  (signed char) (v >> 24);

    writeSignedChars<S> (out, b, 4);
}


template <class S, class T>
void
write (T &out, unsigned int v)
{
    unsigned char b[4];

    b[0] =  (unsigned char) (v);
    b[1] =  (unsigned char) (v >> 8);
    b[2] =  (unsigned char) (v >> 16);
    b[3] =  (unsigned char) (v >> 24);

    writeUnsignedChars<S> (out, b, 4);
}


template <class S, class T>
void
write (T &out, signed long v)
{
    signed char b[8];

    b[0] = (signed char) (v);
    b[1] = (signed char) (v >> 8);
    b[2] = (signed char) (v >> 16);
    b[3] = (signed char) (v >> 24);

    #if LONG_MAX == 2147483647

	if (v >= 0)
	{
	    b[4] = 0;
	    b[5] = 0;
	    b[6] = 0;
	    b[7] = 0;
	}
	else
	{
	    b[4] = ~0;
	    b[5] = ~0;
	    b[6] = ~0;
	    b[7] = ~0;
	}

    #elif LONG_MAX == 9223372036854775807L

	b[4] = (signed char) (v >> 32);
	b[5] = (signed char) (v >> 40);
	b[6] = (signed char) (v >> 48);
	b[7] = (signed char) (v >> 56);

    #else
	
	#error write<T> (T &out, signed long v) not implemented

    #endif

    writeSignedChars<S> (out, b, 8);
}


template <class S, class T>
void
write (T &out, unsigned long v)
{
    unsigned char b[8];

    b[0] = (unsigned char) (v);
    b[1] = (unsigned char) (v >> 8);
    b[2] = (unsigned char) (v >> 16);
    b[3] = (unsigned char) (v >> 24);

    #if ULONG_MAX == 4294967295U

	b[4] = 0;
	b[5] = 0;
	b[6] = 0;
	b[7] = 0;

    #elif ULONG_MAX == 18446744073709551615LU

	b[4] = (unsigned char) (v >> 32);
	b[5] = (unsigned char) (v >> 40);
	b[6] = (unsigned char) (v >> 48);
	b[7] = (unsigned char) (v >> 56);

    #else
	
	#error write<T> (T &out, unsigned long v) not implemented

    #endif

    writeUnsignedChars<S> (out, b, 8);
}


#if ULONG_MAX != 18446744073709551615LU

    template <class S, class T>
    void
    write (T &out, Int64 v)
    {
        unsigned char b[8];

        b[0] = (unsigned char) (v);
        b[1] = (unsigned char) (v >> 8);
        b[2] = (unsigned char) (v >> 16);
        b[3] = (unsigned char) (v >> 24);
        b[4] = (unsigned char) (v >> 32);
        b[5] = (unsigned char) (v >> 40);
        b[6] = (unsigned char) (v >> 48);
        b[7] = (unsigned char) (v >> 56);

        writeUnsignedChars<S> (out, b, 8);
    }

#endif


template <class S, class T>
void
write (T &out, float v)
{
    union {unsigned int i; float f;} u;
    u.f = v;

    unsigned char b[4];

    b[0] = (unsigned char) (u.i);
    b[1] = (unsigned char) (u.i >> 8);
    b[2] = (unsigned char) (u.i >> 16);
    b[3] = (unsigned char) (u.i >> 24);

    writeUnsignedChars<S> (out, b, 4);
}


template <class S, class T>
void
write (T &out, double v)
{
    union {Int64 i; double d;} u;
    u.d = v;

    unsigned char b[8];

    b[0] = (unsigned char) (u.i);
    b[1] = (unsigned char) (u.i >> 8);
    b[2] = (unsigned char) (u.i >> 16);
    b[3] = (unsigned char) (u.i >> 24);
    b[4] = (unsigned char) (u.i >> 32);
    b[5] = (unsigned char) (u.i >> 40);
    b[6] = (unsigned char) (u.i >> 48);
    b[7] = (unsigned char) (u.i >> 56);

    writeUnsignedChars<S> (out, b, 8);
}


template <class S, class T>
inline void
write (T &out, half v)
{
    unsigned char b[2];

    b[0] =  (unsigned char) (v.bits());
    b[1] =  (unsigned char) (v.bits() >> 8);

    writeUnsignedChars<S> (out, b, 2);
}


template <class S, class T>
inline void
write (T &out, const char v[], int n)	// fixed-size char array
{
    S::writeChars (out, v, n);
}


template <class S, class T>
void
write (T &out, const char v[])		// zero-terminated string
{
    while (*v)
    {
	S::writeChars (out, v, 1);
	++v;
    }

    S::writeChars (out, v, 1);
}


template <class S, class T>
void
pad (T &out, int n)			// add n padding bytes
{
    for (int i = 0; i < n; i++)
    {
	const char c = 0;
	S::writeChars (out, &c, 1);
    }
}


template <class S, class T>
inline void
read (T &in, bool &v)
{
    char c;

    S::readChars (in, &c, 1);
    v = !!c;
}


template <class S, class T>
inline void
read (T &in, char &v)
{
    S::readChars (in, &v, 1);
}


template <class S, class T>
inline void
read (T &in, signed char &v)
{
    readSignedChars<S> (in, &v, 1);
}


template <class S, class T>
inline void
read (T &in, unsigned char &v)
{
    readUnsignedChars<S> (in, &v, 1);
}


template <class S, class T>
void
read (T &in, signed short &v)
{
    signed char b[2];

    readSignedChars<S> (in, b, 2);

    v = (b[0] & 0x00ff) |
	(b[1] << 8);
}


template <class S, class T>
void
read (T &in, unsigned short &v)
{
    unsigned char b[2];

    readUnsignedChars<S> (in, b, 2);

    v = (b[0] & 0x00ff) |
	(b[1] << 8);
}


template <class S, class T>
void
read (T &in, signed int &v)
{
    signed char b[4];

    readSignedChars<S> (in, b, 4);

    v =  (b[0]        & 0x000000ff) |
	((b[1] << 8)  & 0x0000ff00) |
	((b[2] << 16) & 0x00ff0000) |
	 (b[3] << 24);
}


template <class S, class T>
void
read (T &in, unsigned int &v)
{
    unsigned char b[4];

    readUnsignedChars<S> (in, b, 4);

    v =  (b[0]        & 0x000000ff) |
	((b[1] << 8)  & 0x0000ff00) |
	((b[2] << 16) & 0x00ff0000) |
	 (b[3] << 24);
}


template <class S, class T>
void
read (T &in, signed long &v)
{
    signed char b[8];

    readSignedChars<S> (in, b, 8);

    #if LONG_MAX == 2147483647

	v =  (b[0]        & 0x000000ff) |
	    ((b[1] << 8)  & 0x0000ff00) |
	    ((b[2] << 16) & 0x00ff0000) |
	     (b[3] << 24);

	if (( b[4] ||  b[5] ||  b[6] ||  b[7]) &&
	    (~b[4] || ~b[5] || ~b[6] || ~b[7]))
	{
	    throw Iex::OverflowExc ("Long int overflow - read a large "
				    "64-bit integer in a 32-bit process.");
	}

    #elif LONG_MAX == 9223372036854775807L

	v =  ((long) b[0]        & 0x00000000000000ff) |
	    (((long) b[1] << 8)  & 0x000000000000ff00) |
	    (((long) b[2] << 16) & 0x0000000000ff0000) |
	    (((long) b[3] << 24) & 0x00000000ff000000) |
	    (((long) b[4] << 32) & 0x000000ff00000000) |
	    (((long) b[5] << 40) & 0x0000ff0000000000) |
	    (((long) b[6] << 48) & 0x00ff000000000000) |
	     ((long) b[7] << 56);

    #else

	#error read<T> (T &in, signed long &v) not implemented

    #endif
}


template <class S, class T>
void
read (T &in, unsigned long &v)
{
    unsigned char b[8];

    readUnsignedChars<S> (in, b, 8);

    #if ULONG_MAX == 4294967295U

	v =  (b[0]        & 0x000000ff) |
	    ((b[1] << 8)  & 0x0000ff00) |
	    ((b[2] << 16) & 0x00ff0000) |
	     (b[3] << 24);

	if (b[4] || b[5] || b[6] || b[7])
	{
	    throw Iex::OverflowExc ("Long int overflow - read a large "
				    "64-bit integer in a 32-bit process.");
	}

    #elif ULONG_MAX == 18446744073709551615LU

	v =  ((unsigned long) b[0]        & 0x00000000000000ff) |
	    (((unsigned long) b[1] << 8)  & 0x000000000000ff00) |
	    (((unsigned long) b[2] << 16) & 0x0000000000ff0000) |
	    (((unsigned long) b[3] << 24) & 0x00000000ff000000) |
	    (((unsigned long) b[4] << 32) & 0x000000ff00000000) |
	    (((unsigned long) b[5] << 40) & 0x0000ff0000000000) |
	    (((unsigned long) b[6] << 48) & 0x00ff000000000000) |
	     ((unsigned long) b[7] << 56);

    #else

	#error read<T> (T &in, unsigned long &v) not implemented

    #endif
}


#if ULONG_MAX != 18446744073709551615LU

    template <class S, class T>
    void
    read (T &in, Int64 &v)
    {
        unsigned char b[8];

        readUnsignedChars<S> (in, b, 8);

        v =  ((Int64) b[0]        & 0x00000000000000ffLL) |
	    (((Int64) b[1] << 8)  & 0x000000000000ff00LL) |
	    (((Int64) b[2] << 16) & 0x0000000000ff0000LL) |
	    (((Int64) b[3] << 24) & 0x00000000ff000000LL) |
	    (((Int64) b[4] << 32) & 0x000000ff00000000LL) |
	    (((Int64) b[5] << 40) & 0x0000ff0000000000LL) |
	    (((Int64) b[6] << 48) & 0x00ff000000000000LL) |
	    ((Int64) b[7] << 56);
    }

#endif


template <class S, class T>
void
read (T &in, float &v)
{
    unsigned char b[4];

    readUnsignedChars<S> (in, b, 4);

    union {unsigned int i; float f;} u;

    u.i = (b[0]        & 0x000000ff) |
	 ((b[1] << 8)  & 0x0000ff00) |
	 ((b[2] << 16) & 0x00ff0000) |
	  (b[3] << 24);

    v = u.f;
}


template <class S, class T>
void
read (T &in, double &v)
{
    unsigned char b[8];

    readUnsignedChars<S> (in, b, 8);

    union {Int64 i; double d;} u;

    u.i = ((Int64) b[0]        & 0x00000000000000ffULL) |
	 (((Int64) b[1] << 8)  & 0x000000000000ff00ULL) |
	 (((Int64) b[2] << 16) & 0x0000000000ff0000ULL) |
	 (((Int64) b[3] << 24) & 0x00000000ff000000ULL) |
	 (((Int64) b[4] << 32) & 0x000000ff00000000ULL) |
	 (((Int64) b[5] << 40) & 0x0000ff0000000000ULL) |
	 (((Int64) b[6] << 48) & 0x00ff000000000000ULL) |
	  ((Int64) b[7] << 56);

    v = u.d;
}


template <class S, class T>
inline void
read (T &in, half &v)
{
    unsigned char b[2];

    readUnsignedChars<S> (in, b, 2);

    v.setBits ((b[0] & 0x00ff) | (b[1] << 8));
}


template <class S, class T>
inline void
read (T &in, char v[], int n)		// fixed-size char array
{
    S::readChars (in, v, n);
}


template <class S, class T>
void
read (T &in, int n, char v[])		// zero-terminated string
{
    while (n >= 0)
    {
	S::readChars (in, v, 1);

	if (*v == 0)
	    break;

	--n;
	++v;
    }
}


template <class S, class T>
void
skip (T &in, int n)			// skip n padding bytes
{
    char c[1024];

    while (n >= (int) sizeof (c))
    {
	if (!S::readChars (in, c, sizeof (c)))
	    return;

	n -= sizeof (c);
    }

    if (n >= 1)
	S::readChars (in, c, n);
}


template <> inline int size <bool> ()			{return 1;}
template <> inline int size <char> ()			{return 1;}
template <> inline int size <signed char> ()		{return 1;}
template <> inline int size <unsigned char> ()		{return 1;}
template <> inline int size <signed short> ()		{return 2;}
template <> inline int size <unsigned short> ()		{return 2;}
template <> inline int size <signed int> ()		{return 4;}
template <> inline int size <unsigned int> ()		{return 4;}
template <> inline int size <signed long> ()		{return 8;}
template <> inline int size <unsigned long> ()		{return 8;}
template <> inline int size <float> ()			{return 4;}
template <> inline int size <double> ()			{return 8;}
template <> inline int size <half> ()			{return 2;}


} // namespace Xdr
} // namespace Imf

#endif
