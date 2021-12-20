//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_IO_H
#define INCLUDED_IMF_IO_H

//-----------------------------------------------------------------------------
//
//	Low-level file input and output for OpenEXR.
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"

#include <string>
#include <cstdint>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-----------------------------------------------------------
// class IStream -- an abstract base class for input streams.
//-----------------------------------------------------------

class IMF_EXPORT_TYPE IStream
{
  public:

    //-----------
    // Destructor
    //-----------

    IMF_EXPORT virtual ~IStream ();
    
    
    //-------------------------------------------------
    // Does this input stream support memory-mapped IO?
    //
    // Memory-mapped streams can avoid an extra copy;
    // memory-mapped read operations return a pointer
    // to an internal buffer instead of copying data
    // into a buffer supplied by the caller.
    //-------------------------------------------------

    IMF_EXPORT virtual bool        isMemoryMapped () const;


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

    IMF_EXPORT virtual char *	readMemoryMapped (int n);


    //--------------------------------------------------------
    // Get the current reading position, in bytes from the
    // beginning of the file.  If the next call to read() will
    // read the first byte in the file, tellg() returns 0.
    //--------------------------------------------------------

    virtual uint64_t	tellg () = 0;


    //-------------------------------------------
    // Set the current reading position.
    // After calling seekg(i), tellg() returns i.
    //-------------------------------------------

    virtual void	seekg (uint64_t pos) = 0;


    //------------------------------------------------------
    // Clear error conditions after an operation has failed.
    //------------------------------------------------------

    IMF_EXPORT virtual void	clear ();


    //------------------------------------------------------
    // Get the name of the file associated with this stream.
    //------------------------------------------------------

    IMF_EXPORT const char *	fileName () const;

  protected:

    IMF_EXPORT IStream (const char fileName[]);

  private:

    IStream (const IStream &) = delete;
    IStream & operator = (const IStream &) = delete;
    IStream (IStream &&) = delete;
    IStream & operator = (IStream &&) = delete;

    std::string		_fileName;
};


//-----------------------------------------------------------
// class OStream -- an abstract base class for output streams
//-----------------------------------------------------------

class IMF_EXPORT_TYPE OStream
{
  public:

    //-----------
    // Destructor
    //-----------

    IMF_EXPORT virtual ~OStream ();
  

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

    virtual uint64_t	tellp () = 0;


    //-------------------------------------------
    // Set the current writing position.
    // After calling seekp(i), tellp() returns i.
    //-------------------------------------------

    virtual void	seekp (uint64_t pos) = 0;


    //------------------------------------------------------
    // Get the name of the file associated with this stream.
    //------------------------------------------------------

    IMF_EXPORT const char *	fileName () const;

  protected:

    IMF_EXPORT OStream (const char fileName[]);

  private:

    OStream (const OStream &) = delete;
    OStream & operator = (const OStream &) = delete;
    OStream (OStream &&) = delete;
    OStream & operator = (OStream &&) = delete;

    std::string		_fileName;
};


//-----------------------
// Helper classes for Xdr
//-----------------------

struct StreamIO
{
    static inline void
    writeChars (OStream &os, const char c[/*n*/], int n)
    {
        os.write (c, n);
    }

    static inline bool
    readChars (IStream &is, char c[/*n*/], int n)
    {
        return is.read (c, n);
    }
};


struct CharPtrIO
{
    static inline void
    writeChars (char *&op, const char c[/*n*/], int n)
    {
        while (n--)
            *op++ = *c++;
    }

    static inline bool
    readChars (const char *&ip, char c[/*n*/], int n)
    {
        while (n--)
            *c++ = *ip++;

        return true;
    }
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
