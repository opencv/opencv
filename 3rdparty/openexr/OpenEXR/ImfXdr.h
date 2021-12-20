//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

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

#include "ImfNamespace.h"

#include "IexMathExc.h"
#include <half.h>
#include <limits.h>
#include <cstdint>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

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
write (T &out, int64_t v);

template <class S, class T>
void
write (T &out, uint64_t v);


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
read (T &in, int64_t &v);

template <class S, class T>
void
read (T &in, uint64_t &v);

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
write (T &out, int64_t v)
{
    signed char b[8];

    b[0] = (signed char) (v);
    b[1] = (signed char) (v >> 8);
    b[2] = (signed char) (v >> 16);
    b[3] = (signed char) (v >> 24);
    b[4] = (signed char) (v >> 32);
    b[5] = (signed char) (v >> 40);
    b[6] = (signed char) (v >> 48);
    b[7] = (signed char) (v >> 56);

    writeSignedChars<S> (out, b, 8);
}

template <class S, class T>
void
write (T &out, uint64_t v)
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
    union {uint64_t i; double d;} u;
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

    v = (static_cast <unsigned char> (b[0]) & 0x00ff) |
	(static_cast <unsigned char> (b[1]) << 8);
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

    v =  (static_cast <unsigned char> (b[0])        & 0x000000ff) |
	((static_cast <unsigned char> (b[1]) << 8)  & 0x0000ff00) |
	((static_cast <unsigned char> (b[2]) << 16) & 0x00ff0000) |
         (static_cast <unsigned char> (b[3]) << 24);
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
read (T &in, int64_t &v)
{
    signed char b[8];

    readSignedChars<S> (in, b, 8);

	v =  (static_cast <int64_t> (b[0])        & 0x00000000000000ff) |
	    ((static_cast <int64_t> (b[1]) << 8)  & 0x000000000000ff00) |
	    ((static_cast <int64_t> (b[2]) << 16) & 0x0000000000ff0000) |
	    ((static_cast <int64_t> (b[3]) << 24) & 0x00000000ff000000) |
	    ((static_cast <int64_t> (b[4]) << 32) & 0x000000ff00000000) |
	    ((static_cast <int64_t> (b[5]) << 40) & 0x0000ff0000000000) |
	    ((static_cast <int64_t> (b[6]) << 48) & 0x00ff000000000000) |
             (static_cast <int64_t> (b[7]) << 56);

}


template <class S, class T>
void
read (T &in, uint64_t &v)
{
    unsigned char b[8];

    readUnsignedChars<S> (in, b, 8);

    v =  ((uint64_t) b[0]        & 0x00000000000000ffLL) |
        (((uint64_t) b[1] << 8)  & 0x000000000000ff00LL) |
        (((uint64_t) b[2] << 16) & 0x0000000000ff0000LL) |
        (((uint64_t) b[3] << 24) & 0x00000000ff000000LL) |
        (((uint64_t) b[4] << 32) & 0x000000ff00000000LL) |
        (((uint64_t) b[5] << 40) & 0x0000ff0000000000LL) |
        (((uint64_t) b[6] << 48) & 0x00ff000000000000LL) |
        ((uint64_t) b[7] << 56);
}


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

    union {uint64_t i; double d;} u;

    u.i = ((uint64_t) b[0]        & 0x00000000000000ffULL) |
	 (((uint64_t) b[1] << 8)  & 0x000000000000ff00ULL) |
	 (((uint64_t) b[2] << 16) & 0x0000000000ff0000ULL) |
	 (((uint64_t) b[3] << 24) & 0x00000000ff000000ULL) |
	 (((uint64_t) b[4] << 32) & 0x000000ff00000000ULL) |
	 (((uint64_t) b[5] << 40) & 0x0000ff0000000000ULL) |
	 (((uint64_t) b[6] << 48) & 0x00ff000000000000ULL) |
	  ((uint64_t) b[7] << 56);

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
template <> inline int size <unsigned long long> ()     {return 8;}
template <> inline int size <float> ()			{return 4;}
template <> inline int size <double> ()			{return 8;}
template <> inline int size <half> ()			{return 2;}


} // namespace Xdr
OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#if defined (OPENEXR_IMF_INTERNAL_NAMESPACE_AUTO_EXPOSE)
namespace Imf{using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;}
#endif


#endif
