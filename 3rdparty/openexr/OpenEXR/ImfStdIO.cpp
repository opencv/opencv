//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Low-level file input and output for OpenEXR
//	based on C++ standard iostreams.
//
//-----------------------------------------------------------------------------

#include <ImfStdIO.h>
#include "Iex.h"
#include <errno.h>
#ifdef _WIN32
# define VC_EXTRALEAN
# include <windows.h>
# include <string.h>
# include <io.h>
# include <fcntl.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <share.h>
# include <string>
# include <iostream>
#endif

using namespace std;
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {

#ifdef _WIN32
wstring WidenFilename (const char *filename)
{
    wstring ret;
    int fnlen = static_cast<int>( strlen(filename) );
    int len = MultiByteToWideChar(CP_UTF8, 0, filename, fnlen, NULL, 0 );
    if (len > 0)
    {
        ret.resize(len);
        MultiByteToWideChar(CP_UTF8, 0, filename, fnlen, &ret[0], len);
    }
    return ret;
}

# if defined(__GLIBCXX__) && !(defined(_GLIBCXX_HAVE_WFOPEN) && defined(_GLIBCXX_USE_WCHAR_T))
#  define USE_CUSTOM_WIDE_OPEN 1
# endif

# ifdef USE_CUSTOM_WIDE_OPEN
template <typename CharT, typename TraitsT>
class InjectFilebuf : public basic_filebuf<CharT, TraitsT>
{
public:
    using base_filebuf = basic_filebuf<CharT, TraitsT>;
    inline base_filebuf* wide_open (int fd, ios_base::openmode m)
    {
        // sys_open will do an fdopen internally which will then clean up the fd upon close
        this->_M_file.sys_open (fd, m);
        if (this->is_open ())
        {
            // reset the internal state, these members are consistent between gcc versions 4.3 - 9
            // but at 9, the wfopen stuff should become available, such that this will no longer be
            // active
            this->_M_allocate_internal_buffer ();
            this->_M_mode    = m;
            this->_M_reading = false;
            this->_M_writing = false;
            this->_M_set_buffer (-1);
            this->_M_state_last = this->_M_state_cur = this->_M_state_beg;
            // we don't ever seek to end or anything, so should be done at this point...
            return this;
        }
        return nullptr;
    }
};
# endif // USE_CUSTOM_WIDE_OPEN

ifstream*
make_ifstream (const char *filename)
{
    wstring wfn = WidenFilename (filename);
# ifdef USE_CUSTOM_WIDE_OPEN
    int     fd;
    errno_t e = _wsopen_s (
        &fd, wfn.c_str (), _O_RDONLY|_O_BINARY, _SH_DENYNO, _S_IREAD | _S_IWRITE);
    if (e != 0)
    {
        char errbuf[4096];
        strerror_s (errbuf, 4096, e);
        errno = e;
        throw IEX_NAMESPACE::ErrnoExc (
            "Unable to open input filestream: " + std::string (errbuf));
    }
    ifstream* ret = new ifstream;
    using CharT   = ifstream::char_type;
    using TraitsT = ifstream::traits_type;
    if (static_cast<InjectFilebuf<CharT, TraitsT>*> (ret->rdbuf ())
        ->wide_open (fd, ios_base::in | ios_base::binary))
    {
        ret->clear();
        ret->setstate(ios_base::goodbit);
    }
#    else
    ifstream* ret = new ifstream(wfn.c_str (), ios_base::in | ios_base::binary);
#    endif
    return ret;
}

ofstream*
make_ofstream (const char* filename)
{
    wstring   wfn = WidenFilename (filename);
# ifdef USE_CUSTOM_WIDE_OPEN
    int     fd;
    errno_t e = _wsopen_s (
        &fd,
        wfn.c_str (),
        _O_WRONLY | _O_CREAT | _O_BINARY,
        _SH_DENYNO,
        _S_IREAD | _S_IWRITE);
    if (e != 0)
    {
        char errbuf[4096];
        strerror_s (errbuf, 4096, e);
        errno = e;
        throw IEX_NAMESPACE::ErrnoExc (
            "Unable to open output filestream: " + std::string(errbuf));
    }
    ofstream* ret = new ofstream;
    using CharT   = ifstream::char_type;
    using TraitsT = ifstream::traits_type;
    if (static_cast<InjectFilebuf<CharT, TraitsT>*> (ret->rdbuf ())
            ->wide_open (fd, ios_base::out | ios_base::binary))
    {
        ret->clear ();
        ret->setstate (ios_base::goodbit);
    }
#    else
    ofstream *ret = new ofstream (wfn.c_str (), ios_base::binary);
#    endif
    return ret;
}
#else
ifstream*
make_ifstream (const char* filename)
{
    return new ifstream (filename, ios_base::binary);
}

inline ofstream*
make_ofstream (const char* filename)
{
    return new ofstream (filename, ios_base::binary);
}
#endif

void
clearError ()
{
    errno = 0;
}


bool
checkError (istream &is, streamsize expected = 0)
{
    if (!is)
    {
	if (errno)
	    IEX_NAMESPACE::throwErrnoExc();

	if (is.gcount() < expected) 
	{
		THROW (IEX_NAMESPACE::InputExc, "Early end of file: read " << is.gcount() 
			<< " out of " << expected << " requested bytes.");
	}
	return false;
    }

    return true;
}


void
checkError (ostream &os)
{
    if (!os)
    {
	if (errno)
	    IEX_NAMESPACE::throwErrnoExc();

	throw IEX_NAMESPACE::ErrnoExc ("File output failed.");
    }
}

} // namespace


StdIFStream::StdIFStream (const char fileName[]):
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream (fileName),
    _is (make_ifstream (fileName)),
    _deleteStream (true)
{
    if (!*_is)
    {
	delete _is;
	IEX_NAMESPACE::throwErrnoExc();
    }
}

    
StdIFStream::StdIFStream (ifstream &is, const char fileName[]):
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream (fileName),
    _is (&is),
    _deleteStream (false)
{
    // empty
}


StdIFStream::~StdIFStream ()
{
    if (_deleteStream)
	delete _is;
}


bool
StdIFStream::read (char c[/*n*/], int n)
{
    if (!*_is)
        throw IEX_NAMESPACE::InputExc ("Unexpected end of file.");

    clearError();
    _is->read (c, n);
    return checkError (*_is, n);
}


uint64_t
StdIFStream::tellg ()
{
    return std::streamoff (_is->tellg());
}


void
StdIFStream::seekg (uint64_t pos)
{
    _is->seekg (pos);
    checkError (*_is);
}


void
StdIFStream::clear ()
{
    _is->clear();
}


StdISStream::StdISStream (): OPENEXR_IMF_INTERNAL_NAMESPACE::IStream ("(string)")
{
    // empty
}

StdISStream::~StdISStream ()
{
}

bool
StdISStream::read (char c[/*n*/], int n)
{
    if (!_is)
        throw IEX_NAMESPACE::InputExc ("Unexpected end of file.");

    clearError();
    _is.read (c, n);
    return checkError (_is, n);
}


uint64_t
StdISStream::tellg ()
{
    return std::streamoff (_is.tellg());
}


void
StdISStream::seekg (uint64_t pos)
{
    _is.seekg (pos);
    checkError (_is);
}


void
StdISStream::clear ()
{
    _is.clear();
}


std::string
StdISStream::str () const
{
    return _is.str ();
}


void
StdISStream::str (const std::string &s)
{
    _is.str(s);
}



StdOFStream::StdOFStream (const char fileName[])
    : OPENEXR_IMF_INTERNAL_NAMESPACE::OStream (fileName)
    , _os (make_ofstream (fileName))
    , _deleteStream (true)
{
    if (!*_os)
    {
	delete _os;
	IEX_NAMESPACE::throwErrnoExc();
    }
}


StdOFStream::StdOFStream (ofstream &os, const char fileName[]):
    OPENEXR_IMF_INTERNAL_NAMESPACE::OStream (fileName),
    _os (&os),
    _deleteStream (false)
{
    // empty
}


StdOFStream::~StdOFStream ()
{
    if (_deleteStream)
	delete _os;
}


void
StdOFStream::write (const char c[/*n*/], int n)
{
    clearError();
    _os->write (c, n);
    checkError (*_os);
}


uint64_t
StdOFStream::tellp ()
{
    return std::streamoff (_os->tellp());
}


void
StdOFStream::seekp (uint64_t pos)
{
    _os->seekp (pos);
    checkError (*_os);
}


StdOSStream::StdOSStream (): OPENEXR_IMF_INTERNAL_NAMESPACE::OStream ("(string)")
{
    // empty
}

StdOSStream::~StdOSStream ()
{
}

void
StdOSStream::write (const char c[/*n*/], int n)
{
    clearError();
    _os.write (c, n);
    checkError (_os);
}


uint64_t
StdOSStream::tellp ()
{
    return std::streamoff (_os.tellp());
}


void
StdOSStream::seekp (uint64_t pos)
{
    _os.seekp (pos);
    checkError (_os);
}

std::string
StdOSStream::str () const
{
    return _os.str ();
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
