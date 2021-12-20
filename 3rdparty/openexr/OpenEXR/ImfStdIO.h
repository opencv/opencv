//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_STD_IO_H
#define INCLUDED_IMF_STD_IO_H

//-----------------------------------------------------------------------------
//
//	Low-level file input and output for OpenEXR
//	based on C++ standard iostreams.
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfIO.h"

#include <fstream>
#include <sstream>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-------------------------------------------
// class StdIFStream -- an implementation of
// class OPENEXR_IMF_INTERNAL_NAMESPACE::IStream based on class std::ifstream
//-------------------------------------------

class IMF_EXPORT_TYPE StdIFStream: public OPENEXR_IMF_INTERNAL_NAMESPACE::IStream
{
  public:

    //-------------------------------------------------------
    // A constructor that opens the file with the given name.
    // The destructor will close the file.
    //-------------------------------------------------------

    IMF_EXPORT StdIFStream (const char fileName[]);

    
    //---------------------------------------------------------
    // A constructor that uses a std::ifstream that has already
    // been opened by the caller.  The StdIFStream's destructor
    // will not close the std::ifstream.
    //---------------------------------------------------------

    IMF_EXPORT StdIFStream (std::ifstream &is, const char fileName[]);


    IMF_EXPORT virtual ~StdIFStream ();
    StdIFStream (const StdIFStream &) = delete;
    StdIFStream (StdIFStream &&) = delete;
    StdIFStream &operator=(const StdIFStream &) = delete;
    StdIFStream &operator=(StdIFStream &&) = delete;

    IMF_EXPORT virtual bool     read (char c[/*n*/], int n);
    IMF_EXPORT virtual uint64_t tellg ();
    IMF_EXPORT virtual void     seekg (uint64_t pos);
    IMF_EXPORT virtual void     clear ();

  private:

    std::ifstream *	_is;
    bool		_deleteStream;
};


//------------------------------------------------
// class StdISStream -- an implementation of class
// OPENEXR_IMF_INTERNAL_NAMESPACE::IStream, based on class std::istringstream
//------------------------------------------------

class IMF_EXPORT_TYPE StdISStream: public OPENEXR_IMF_INTERNAL_NAMESPACE::IStream
{
  public:

    IMF_EXPORT StdISStream ();
    IMF_EXPORT ~StdISStream ();
    StdISStream (const StdISStream &) = delete;
    StdISStream (StdISStream &&) = delete;
    StdISStream &operator=(const StdISStream &) = delete;
    StdISStream &operator=(StdISStream &&) = delete;

    IMF_EXPORT virtual bool     read (char c[/*n*/], int n);
    IMF_EXPORT virtual uint64_t tellg ();
    IMF_EXPORT virtual void     seekg (uint64_t pos);
    IMF_EXPORT virtual void     clear ();

    IMF_EXPORT std::string   str () const; 

    IMF_EXPORT void          str (const std::string &s);

  private:

    std::istringstream 	_is;
};



//-------------------------------------------
// class StdOFStream -- an implementation of
// class OPENEXR_IMF_INTERNAL_NAMESPACE::OStream based on class std::ofstream
//-------------------------------------------

class IMF_EXPORT_TYPE StdOFStream: public OPENEXR_IMF_INTERNAL_NAMESPACE::OStream
{
  public:

    //-------------------------------------------------------
    // A constructor that opens the file with the given name.
    // The destructor will close the file.
    //-------------------------------------------------------

    IMF_EXPORT StdOFStream (const char fileName[]);


    //---------------------------------------------------------
    // A constructor that uses a std::ofstream that has already
    // been opened by the caller.  The StdOFStream's destructor
    // will not close the std::ofstream.
    //---------------------------------------------------------

    IMF_EXPORT StdOFStream (std::ofstream &os, const char fileName[]);


    IMF_EXPORT virtual ~StdOFStream ();
    StdOFStream (const StdOFStream &) = delete;
    StdOFStream (StdOFStream &&) = delete;
    StdOFStream &operator=(const StdOFStream &) = delete;
    StdOFStream &operator=(StdOFStream &&) = delete;

    IMF_EXPORT virtual void     write (const char c[/*n*/], int n);
    IMF_EXPORT virtual uint64_t tellp ();
    IMF_EXPORT virtual void     seekp (uint64_t pos);

  private:

    std::ofstream *	_os;
    bool		_deleteStream;
};


//------------------------------------------------
// class StdOSStream -- an implementation of class
// OPENEXR_IMF_INTERNAL_NAMESPACE::OStream, based on class std::ostringstream
//------------------------------------------------

class IMF_EXPORT_TYPE StdOSStream: public OPENEXR_IMF_INTERNAL_NAMESPACE::OStream
{
  public:

    IMF_EXPORT StdOSStream ();
    IMF_EXPORT ~StdOSStream ();
    StdOSStream (const StdOSStream &) = delete;
    StdOSStream (StdOSStream &&) = delete;
    StdOSStream &operator=(const StdOSStream &) = delete;
    StdOSStream &operator=(StdOSStream &&) = delete;

    IMF_EXPORT virtual void     write (const char c[/*n*/], int n);
    IMF_EXPORT virtual uint64_t tellp ();
    IMF_EXPORT virtual void     seekp (uint64_t pos);

    IMF_EXPORT std::string      str () const;

  private:

    std::ostringstream 	_os;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
