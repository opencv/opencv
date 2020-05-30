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


#ifndef INCLUDED_IMF_STD_IO_H
#define INCLUDED_IMF_STD_IO_H

//-----------------------------------------------------------------------------
//
//	Low-level file input and output for OpenEXR
//	based on C++ standard iostreams.
//
//-----------------------------------------------------------------------------

#include "ImfIO.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

#include <fstream>
#include <sstream>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-------------------------------------------
// class StdIFStream -- an implementation of
// class OPENEXR_IMF_INTERNAL_NAMESPACE::IStream based on class std::ifstream
//-------------------------------------------

class StdIFStream: public OPENEXR_IMF_INTERNAL_NAMESPACE::IStream
{
  public:

    //-------------------------------------------------------
    // A constructor that opens the file with the given name.
    // The destructor will close the file.
    //-------------------------------------------------------

    IMF_EXPORT
    StdIFStream (const char fileName[]);

    
    //---------------------------------------------------------
    // A constructor that uses a std::ifstream that has already
    // been opened by the caller.  The StdIFStream's destructor
    // will not close the std::ifstream.
    //---------------------------------------------------------

    IMF_EXPORT
    StdIFStream (std::ifstream &is, const char fileName[]);


    IMF_EXPORT
    virtual ~StdIFStream ();

    IMF_EXPORT
    virtual bool	read (char c[/*n*/], int n);
    IMF_EXPORT
    virtual Int64	tellg ();
    IMF_EXPORT
    virtual void	seekg (Int64 pos);
    IMF_EXPORT
    virtual void	clear ();

  private:

    std::ifstream *	_is;
    bool		_deleteStream;
};


//-------------------------------------------
// class StdOFStream -- an implementation of
// class OPENEXR_IMF_INTERNAL_NAMESPACE::OStream based on class std::ofstream
//-------------------------------------------

class StdOFStream: public OPENEXR_IMF_INTERNAL_NAMESPACE::OStream
{
  public:

    //-------------------------------------------------------
    // A constructor that opens the file with the given name.
    // The destructor will close the file.
    //-------------------------------------------------------

    IMF_EXPORT
    StdOFStream (const char fileName[]);
    

    //---------------------------------------------------------
    // A constructor that uses a std::ofstream that has already
    // been opened by the caller.  The StdOFStream's destructor
    // will not close the std::ofstream.
    //---------------------------------------------------------

    IMF_EXPORT
    StdOFStream (std::ofstream &os, const char fileName[]);


    IMF_EXPORT
    virtual ~StdOFStream ();

    IMF_EXPORT
    virtual void	write (const char c[/*n*/], int n);
    IMF_EXPORT
    virtual Int64	tellp ();
    IMF_EXPORT
    virtual void	seekp (Int64 pos);

  private:

    std::ofstream *	_os;
    bool		_deleteStream;
};


//------------------------------------------------
// class StdOSStream -- an implementation of class
// OPENEXR_IMF_INTERNAL_NAMESPACE::OStream, based on class std::ostringstream
//------------------------------------------------

class StdOSStream: public OPENEXR_IMF_INTERNAL_NAMESPACE::OStream
{
  public:

    IMF_EXPORT
    StdOSStream ();

    IMF_EXPORT
    virtual void	write (const char c[/*n*/], int n);
    IMF_EXPORT
    virtual Int64	tellp ();
    IMF_EXPORT
    virtual void	seekp (Int64 pos);

    IMF_EXPORT
    std::string		str () const {return _os.str();}

  private:

    std::ostringstream 	_os;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
