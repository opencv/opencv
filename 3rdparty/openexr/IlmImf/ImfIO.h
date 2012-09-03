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


#ifndef INCLUDED_IMF_IO_H
#define INCLUDED_IMF_IO_H

//-----------------------------------------------------------------------------
//
//	Low-level file input and output for OpenEXR.
//
//-----------------------------------------------------------------------------

#include <ImfInt64.h>
#include <string>

namespace Imf {

//-----------------------------------------------------------
// class IStream -- an abstract base class for input streams.
//-----------------------------------------------------------

class IStream
{
  public:

    //-----------
    // Destructor
    //-----------

    virtual ~IStream ();
    
    
    //-------------------------------------------------
    // Does this input stream support memory-mapped IO?
    //
    // Memory-mapped streams can avoid an extra copy;
    // memory-mapped read operations return a pointer
    // to an internal buffer instead of copying data
    // into a buffer supplied by the caller.
    //-------------------------------------------------

    virtual bool        isMemoryMapped () const;


    //------------------------------------------------------
    // Read from the stream:
    //
    // read(c,n) reads n bytes from the stream, and stores
    // them in array c.  If the stream contains less than n
    // bytes, or if an I/O error occurs, read(c,n) throws
    // an exception.  If read(c,n) reads the last byte from
    // the file it returns false, otherwise it returns true.
    //------------------------------------------------------

    virtual bool	read (char c[/*n*/], int n) = 0;
    
    
    //---------------------------------------------------
    // Read from a memory-mapped stream:
    //
    // readMemoryMapped(n) reads n bytes from the stream
    // and returns a pointer to the first byte.  The
    // returned pointer remains valid until the stream
    // is closed.  If there are less than n byte left to
    // read in the stream or if the stream is not memory-
    // mapped, readMemoryMapped(n) throws an exception.  
    //---------------------------------------------------

    virtual char *	readMemoryMapped (int n);


    //--------------------------------------------------------
    // Get the current reading position, in bytes from the
    // beginning of the file.  If the next call to read() will
    // read the first byte in the file, tellg() returns 0.
    //--------------------------------------------------------

    virtual Int64	tellg () = 0;


    //-------------------------------------------
    // Set the current reading position.
    // After calling seekg(i), tellg() returns i.
    //-------------------------------------------

    virtual void	seekg (Int64 pos) = 0;


    //------------------------------------------------------
    // Clear error conditions after an operation has failed.
    //------------------------------------------------------

    virtual void	clear ();


    //------------------------------------------------------
    // Get the name of the file associated with this stream.
    //------------------------------------------------------

    const char *	fileName () const;

  protected:

    IStream (const char fileName[]);

  private:

    IStream (const IStream &);			// not implemented
    IStream & operator = (const IStream &);	// not implemented

    std::string		_fileName;
};


//-----------------------------------------------------------
// class OStream -- an abstract base class for output streams
//-----------------------------------------------------------

class OStream
{
  public:

    //-----------
    // Destructor
    //-----------

    virtual ~OStream ();
  

    //----------------------------------------------------------
    // Write to the stream:
    //
    // write(c,n) takes n bytes from array c, and stores them
    // in the stream.  If an I/O error occurs, write(c,n) throws
    // an exception.
    //----------------------------------------------------------

    virtual void	write (const char c[/*n*/], int n) = 0;


    //---------------------------------------------------------
    // Get the current writing position, in bytes from the
    // beginning of the file.  If the next call to write() will
    // start writing at the beginning of the file, tellp()
    // returns 0.
    //---------------------------------------------------------

    virtual Int64	tellp () = 0;


    //-------------------------------------------
    // Set the current writing position.
    // After calling seekp(i), tellp() returns i.
    //-------------------------------------------

    virtual void	seekp (Int64 pos) = 0;


    //------------------------------------------------------
    // Get the name of the file associated with this stream.
    //------------------------------------------------------

    const char *	fileName () const;

  protected:

    OStream (const char fileName[]);

  private:

    OStream (const OStream &);			// not implemented
    OStream & operator = (const OStream &);	// not implemented

    std::string		_fileName;
};


//-----------------------
// Helper classes for Xdr
//-----------------------

struct StreamIO
{
    static void
    writeChars (OStream &os, const char c[/*n*/], int n)
    {
	os.write (c, n);
    }

    static bool
    readChars (IStream &is, char c[/*n*/], int n)
    {
	return is.read (c, n);
    }
};


struct CharPtrIO
{
    static void
    writeChars (char *&op, const char c[/*n*/], int n)
    {
	while (n--)
	    *op++ = *c++;
    }

    static bool
    readChars (const char *&ip, char c[/*n*/], int n)
    {
	while (n--)
	    *c++ = *ip++;

	return true;
    }
};


} // namespace Imf

#endif
