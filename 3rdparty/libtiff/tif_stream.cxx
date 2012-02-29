/* $Id: tif_stream.cxx,v 1.11 2010-12-11 23:12:29 faxguy Exp $ */

/*
 * Copyright (c) 1988-1996 Sam Leffler
 * Copyright (c) 1991-1996 Silicon Graphics, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and 
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 * 
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY 
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  
 * 
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF 
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
 * OF THIS SOFTWARE.
 */

/*
 * TIFF Library UNIX-specific Routines.
 */
#include "tiffiop.h"
#include <iostream>

#ifndef __VMS
using namespace std;
#endif

/*
  ISO C++ uses a 'std::streamsize' type to define counts.  This makes
  it similar to, (but perhaps not the same as) size_t.

  The std::ios::pos_type is used to represent stream positions as used
  by tellg(), tellp(), seekg(), and seekp().  This makes it similar to
  (but perhaps not the same as) 'off_t'.  The std::ios::streampos type
  is used for character streams, but is documented to not be an
  integral type anymore, so it should *not* be assigned to an integral
  type.

  The std::ios::off_type is used to specify relative offsets needed by
  the variants of seekg() and seekp() which accept a relative offset
  argument.

  Useful prototype knowledge:

  Obtain read position
    ios::pos_type basic_istream::tellg()

  Set read position
    basic_istream& basic_istream::seekg(ios::pos_type)
    basic_istream& basic_istream::seekg(ios::off_type, ios_base::seekdir)

  Read data
    basic_istream& istream::read(char *str, streamsize count)

  Number of characters read in last unformatted read
    streamsize istream::gcount();

  Obtain write position
    ios::pos_type basic_ostream::tellp()

  Set write position
    basic_ostream& basic_ostream::seekp(ios::pos_type)
    basic_ostream& basic_ostream::seekp(ios::off_type, ios_base::seekdir)

  Write data
    basic_ostream& ostream::write(const char *str, streamsize count)
*/

struct tiffis_data;
struct tiffos_data;

extern "C" {

	static tmsize_t _tiffosReadProc(thandle_t, void*, tmsize_t);
	static tmsize_t _tiffisReadProc(thandle_t fd, void* buf, tmsize_t size);
	static tmsize_t _tiffosWriteProc(thandle_t fd, void* buf, tmsize_t size);
	static tmsize_t _tiffisWriteProc(thandle_t, void*, tmsize_t);
	static uint64   _tiffosSeekProc(thandle_t fd, uint64 off, int whence);
	static uint64   _tiffisSeekProc(thandle_t fd, uint64 off, int whence);
	static uint64   _tiffosSizeProc(thandle_t fd);
	static uint64   _tiffisSizeProc(thandle_t fd);
	static int      _tiffosCloseProc(thandle_t fd);
	static int      _tiffisCloseProc(thandle_t fd);
	static int 	_tiffDummyMapProc(thandle_t , void** base, toff_t* size );
	static void     _tiffDummyUnmapProc(thandle_t , void* base, toff_t size );
	static TIFF*    _tiffStreamOpen(const char* name, const char* mode, void *fd);

struct tiffis_data
{
	istream	*stream;
        ios::pos_type start_pos;
};

struct tiffos_data
{
	ostream	*stream;
	ios::pos_type start_pos;
};

static tmsize_t
_tiffosReadProc(thandle_t, void*, tmsize_t)
{
        return 0;
}

static tmsize_t
_tiffisReadProc(thandle_t fd, void* buf, tmsize_t size)
{
        tiffis_data	*data = reinterpret_cast<tiffis_data *>(fd);

        // Verify that type does not overflow.
        streamsize request_size = size;
        if (static_cast<tmsize_t>(request_size) != size)
          return static_cast<tmsize_t>(-1);

        data->stream->read((char *) buf, request_size);

        return static_cast<tmsize_t>(data->stream->gcount());
}

static tmsize_t
_tiffosWriteProc(thandle_t fd, void* buf, tmsize_t size)
{
	tiffos_data	*data = reinterpret_cast<tiffos_data *>(fd);
	ostream		*os = data->stream;
	ios::pos_type	pos = os->tellp();

        // Verify that type does not overflow.
        streamsize request_size = size;
        if (static_cast<tmsize_t>(request_size) != size)
          return static_cast<tmsize_t>(-1);

	os->write(reinterpret_cast<const char *>(buf), request_size);

	return static_cast<tmsize_t>(os->tellp() - pos);
}

static tmsize_t
_tiffisWriteProc(thandle_t, void*, tmsize_t)
{
	return 0;
}

static uint64
_tiffosSeekProc(thandle_t fd, uint64 off, int whence)
{
	tiffos_data	*data = reinterpret_cast<tiffos_data *>(fd);
	ostream		*os = data->stream;

	// if the stream has already failed, don't do anything
	if( os->fail() )
		return static_cast<uint64>(-1);

	switch(whence) {
	case SEEK_SET:
		{
			// Compute 64-bit offset
			uint64 new_offset = static_cast<uint64>(data->start_pos) + off;

			// Verify that value does not overflow
			ios::off_type offset = static_cast<ios::off_type>(new_offset);
			if (static_cast<uint64>(offset) != new_offset)
				return static_cast<uint64>(-1);
			
			os->seekp(offset, ios::beg);
		break;
		}
	case SEEK_CUR:
		{
			// Verify that value does not overflow
			ios::off_type offset = static_cast<ios::off_type>(off);
			if (static_cast<uint64>(offset) != off)
				return static_cast<uint64>(-1);

			os->seekp(offset, ios::cur);
			break;
		}
	case SEEK_END:
		{
			// Verify that value does not overflow
			ios::off_type offset = static_cast<ios::off_type>(off);
			if (static_cast<uint64>(offset) != off)
				return static_cast<uint64>(-1);

			os->seekp(offset, ios::end);
			break;
		}
	}

	// Attempt to workaround problems with seeking past the end of the
	// stream.  ofstream doesn't have a problem with this but
	// ostrstream/ostringstream does. In that situation, add intermediate
	// '\0' characters.
	if( os->fail() ) {
#ifdef __VMS
		int		old_state;
#else
		ios::iostate	old_state;
#endif
		ios::pos_type	origin;

		old_state = os->rdstate();
		// reset the fail bit or else tellp() won't work below
		os->clear(os->rdstate() & ~ios::failbit);
		switch( whence ) {
			case SEEK_SET:
                        default:
				origin = data->start_pos;
				break;
			case SEEK_CUR:
				origin = os->tellp();
				break;
			case SEEK_END:
				os->seekp(0, ios::end);
				origin = os->tellp();
				break;
		}
		// restore original stream state
		os->clear(old_state);	

		// only do something if desired seek position is valid
		if( (static_cast<uint64>(origin) + off) > static_cast<uint64>(data->start_pos) ) {
			uint64	num_fill;

			// clear the fail bit 
			os->clear(os->rdstate() & ~ios::failbit);

			// extend the stream to the expected size
			os->seekp(0, ios::end);
			num_fill = (static_cast<uint64>(origin)) + off - os->tellp();
			for( uint64 i = 0; i < num_fill; i++ )
				os->put('\0');

			// retry the seek
			os->seekp(static_cast<ios::off_type>(static_cast<uint64>(origin) + off), ios::beg);
		}
	}

	return static_cast<uint64>(os->tellp());
}

static uint64
_tiffisSeekProc(thandle_t fd, uint64 off, int whence)
{
	tiffis_data	*data = reinterpret_cast<tiffis_data *>(fd);

	switch(whence) {
	case SEEK_SET:
		{
			// Compute 64-bit offset
			uint64 new_offset = static_cast<uint64>(data->start_pos) + off;
			
			// Verify that value does not overflow
			ios::off_type offset = static_cast<ios::off_type>(new_offset);
			if (static_cast<uint64>(offset) != new_offset)
				return static_cast<uint64>(-1);

			data->stream->seekg(offset, ios::beg);
			break;
		}
	case SEEK_CUR:
		{
			// Verify that value does not overflow
			ios::off_type offset = static_cast<ios::off_type>(off);
			if (static_cast<uint64>(offset) != off)
				return static_cast<uint64>(-1);

			data->stream->seekg(offset, ios::cur);
			break;
		}
	case SEEK_END:
		{
			// Verify that value does not overflow
			ios::off_type offset = static_cast<ios::off_type>(off);
			if (static_cast<uint64>(offset) != off)
				return static_cast<uint64>(-1);

			data->stream->seekg(offset, ios::end);
			break;
		}
	}

	return (uint64) (data->stream->tellg() - data->start_pos);
}

static uint64
_tiffosSizeProc(thandle_t fd)
{
	tiffos_data	*data = reinterpret_cast<tiffos_data *>(fd);
	ostream		*os = data->stream;
	ios::pos_type	pos = os->tellp();
	ios::pos_type	len;

	os->seekp(0, ios::end);
	len = os->tellp();
	os->seekp(pos);

	return (uint64) len;
}

static uint64
_tiffisSizeProc(thandle_t fd)
{
	tiffis_data	*data = reinterpret_cast<tiffis_data *>(fd);
	ios::pos_type	pos = data->stream->tellg();
	ios::pos_type	len;

	data->stream->seekg(0, ios::end);
	len = data->stream->tellg();
	data->stream->seekg(pos);

	return (uint64) len;
}

static int
_tiffosCloseProc(thandle_t fd)
{
	// Our stream was not allocated by us, so it shouldn't be closed by us.
	delete reinterpret_cast<tiffos_data *>(fd);
	return 0;
}

static int
_tiffisCloseProc(thandle_t fd)
{
	// Our stream was not allocated by us, so it shouldn't be closed by us.
	delete reinterpret_cast<tiffis_data *>(fd);
	return 0;
}

static int
_tiffDummyMapProc(thandle_t , void** base, toff_t* size )
{
	return (0);
}

static void
_tiffDummyUnmapProc(thandle_t , void* base, toff_t size )
{
}

/*
 * Open a TIFF file descriptor for read/writing.
 */
static TIFF*
_tiffStreamOpen(const char* name, const char* mode, void *fd)
{
	TIFF*	tif;

	if( strchr(mode, 'w') ) {
		tiffos_data	*data = new tiffos_data;
		data->stream = reinterpret_cast<ostream *>(fd);
		data->start_pos = data->stream->tellp();

		// Open for writing.
		tif = TIFFClientOpen(name, mode,
				reinterpret_cast<thandle_t>(data),
				_tiffosReadProc,
                                _tiffosWriteProc,
				_tiffosSeekProc,
                                _tiffosCloseProc,
				_tiffosSizeProc,
				_tiffDummyMapProc,
                                _tiffDummyUnmapProc);
	} else {
		tiffis_data	*data = new tiffis_data;
		data->stream = reinterpret_cast<istream *>(fd);
		data->start_pos = data->stream->tellg();
		// Open for reading.
		tif = TIFFClientOpen(name, mode,
				reinterpret_cast<thandle_t>(data),
				_tiffisReadProc,
                                _tiffisWriteProc,
				_tiffisSeekProc,
                                _tiffisCloseProc,
				_tiffisSizeProc,
				_tiffDummyMapProc,
                                _tiffDummyUnmapProc);
	}

	return (tif);
}

} /* extern "C" */

TIFF*
TIFFStreamOpen(const char* name, ostream *os)
{
	// If os is either a ostrstream or ostringstream, and has no data
	// written to it yet, then tellp() will return -1 which will break us.
	// We workaround this by writing out a dummy character and
	// then seek back to the beginning.
	if( !os->fail() && static_cast<int>(os->tellp()) < 0 ) {
		*os << '\0';
		os->seekp(0);
	}

	// NB: We don't support mapped files with streams so add 'm'
	return _tiffStreamOpen(name, "wm", os);
}

TIFF*
TIFFStreamOpen(const char* name, istream *is)
{
	// NB: We don't support mapped files with streams so add 'm'
	return _tiffStreamOpen(name, "rm", is);
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
  Local Variables:
  mode: c
  indent-tabs-mode: true
  c-basic-offset: 8
  End:
*/
