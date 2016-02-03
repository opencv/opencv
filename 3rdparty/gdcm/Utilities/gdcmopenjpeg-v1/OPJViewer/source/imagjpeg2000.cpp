/*
 * Copyright (c) 2007, Digital Signal Processing Laboratory, Università degli studi di Perugia (UPG), Italy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/////////////////////////////////////////////////////////////////////////////
// Name:        imagjpeg2000.cpp
// Purpose:     wxImage JPEG 2000 family file format handler
// Author:      Giuseppe Baruffa - based on imagjpeg.cpp, Vaclav Slavik
// RCS-ID:      $Id: imagjpeg2000.cpp,v 0.00 2008/01/31 10:58:00 MW Exp $
// Copyright:   (c) Giuseppe Baruffa
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
    #pragma hdrstop
#endif

#if wxUSE_IMAGE && wxUSE_LIBOPENJPEG

#include "imagjpeg2000.h"

#ifndef WX_PRECOMP
    #include "wx/log.h"
    #include "wx/app.h"
    #include "wx/intl.h"
    #include "wx/bitmap.h"
    #include "wx/module.h"
#endif

#include "libopenjpeg/openjpeg.h"

#include "wx/filefn.h"
#include "wx/wfstream.h"

// ----------------------------------------------------------------------------
// types
// ----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// wxJPEG2000Handler
//-----------------------------------------------------------------------------

IMPLEMENT_DYNAMIC_CLASS(wxJPEG2000Handler,wxImageHandler)

#if wxUSE_STREAMS

//------------- JPEG 2000 Data Source Manager

#define J2K_CFMT 0
#define JP2_CFMT 1
#define JPT_CFMT 2
#define MJ2_CFMT 3
#define PXM_DFMT 0
#define PGX_DFMT 1
#define BMP_DFMT 2
#define YUV_DFMT 3

#define MAX_MESSAGE_LEN 200

/* check file type */
int
jpeg2000familytype(unsigned char *hdr, int hdr_len)
{
	// check length
    if (hdr_len < 24)
        return -1;

	// check format
	if (hdr[0] == 0x00 &&
			hdr[1] == 0x00 &&
			hdr[2] == 0x00 &&
			hdr[3] == 0x0C &&
			hdr[4] == 0x6A &&
			hdr[5] == 0x50 &&
			hdr[6] == 0x20 &&
			hdr[7] == 0x20 &&
			hdr[20] == 0x6A &&
			hdr[21] == 0x70 &&
			hdr[22] == 0x32)
		// JP2 file format
		return JP2_CFMT;
	else if (hdr[0] == 0x00 &&
			hdr[1] == 0x00 &&
			hdr[2] == 0x00 &&
			hdr[3] == 0x0C &&
			hdr[4] == 0x6A &&
			hdr[5] == 0x50 &&
			hdr[6] == 0x20 &&
			hdr[7] == 0x20 &&
			hdr[20] == 0x6D &&
			hdr[21] == 0x6A &&
			hdr[22] == 0x70 &&
			hdr[23] == 0x32)
		// MJ2 file format
		return MJ2_CFMT;
	else if (hdr[0] == 0xFF &&
			hdr[1] == 0x4F)
		// J2K file format
		return J2K_CFMT;
	else
		// unknown format
		return -1;

}

/* we have to use this to avoid GUI-noGUI threads crashing */
void printevent(const char *msg)
{
#ifndef __WXGTK__ 
	wxMutexGuiEnter();
#endif /* __WXGTK__ */
	wxLogMessage(wxT("%s"), msg);
#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */
}

/* sample error callback expecting a FILE* client object */
void jpeg2000_error_callback(const char *msg, void *client_data) {
	char mess[MAX_MESSAGE_LEN + 20];
	int message_len = strlen(msg);

	if (message_len > MAX_MESSAGE_LEN)
		message_len = MAX_MESSAGE_LEN;
	
	if (msg[message_len - 1] == '\n')
		message_len--;

	sprintf(mess, "[ERROR] %.*s", message_len, msg);
	printevent(mess);
}

/* sample warning callback expecting a FILE* client object */
void jpeg2000_warning_callback(const char *msg, void *client_data) {
	char mess[MAX_MESSAGE_LEN + 20];
	int message_len = strlen(msg);

	if (message_len > MAX_MESSAGE_LEN)
		message_len = MAX_MESSAGE_LEN;
	
	if (msg[message_len - 1] == '\n')
		message_len--;

	sprintf(mess, "[WARNING] %.*s", message_len, msg);
	printevent(mess);
}

/* sample debug callback expecting no client object */
void jpeg2000_info_callback(const char *msg, void *client_data) {
	char mess[MAX_MESSAGE_LEN + 20];
	int message_len = strlen(msg);

	if (message_len > MAX_MESSAGE_LEN)
		message_len = MAX_MESSAGE_LEN;
	
	if (msg[message_len - 1] == '\n')
		message_len--;

	sprintf(mess, "[INFO] %.*s", message_len, msg);
	printevent(mess);
}

/* macro functions */
/* From little endian to big endian, 2 and 4 bytes */
#define	BYTE_SWAP2(X)	((X & 0x00FF) << 8) | ((X & 0xFF00) >> 8)
#define	BYTE_SWAP4(X)	((X & 0x000000FF) << 24) | ((X & 0x0000FF00) << 8) | ((X & 0x00FF0000) >> 8) | ((X & 0xFF000000) >> 24)

#ifdef __WXGTK__
#define	BYTE_SWAP8(X)	((X & 0x00000000000000FFULL) << 56) | ((X & 0x000000000000FF00ULL) << 40) | \
                        ((X & 0x0000000000FF0000ULL) << 24) | ((X & 0x00000000FF000000ULL) << 8) | \
						((X & 0x000000FF00000000ULL) >> 8)  | ((X & 0x0000FF0000000000ULL) >> 24) | \
						((X & 0x00FF000000000000ULL) >> 40) | ((X & 0xFF00000000000000ULL) >> 56)
#else
#define	BYTE_SWAP8(X)	((X & 0x00000000000000FF) << 56) | ((X & 0x000000000000FF00) << 40) | \
                        ((X & 0x0000000000FF0000) << 24) | ((X & 0x00000000FF000000) << 8) | \
						((X & 0x000000FF00000000) >> 8)  | ((X & 0x0000FF0000000000) >> 24) | \
						((X & 0x00FF000000000000) >> 40) | ((X & 0xFF00000000000000) >> 56)
#endif

/* From codestream to int values */
#define STREAM_TO_UINT32(C, P)	(((unsigned long int) (C)[(P) + 0] << 24) + \
								((unsigned long int) (C)[(P) + 1] << 16) + \
								((unsigned long int) (C)[(P) + 2] << 8) + \
								((unsigned long int) (C)[(P) + 3] << 0))

#define STREAM_TO_UINT16(C, P)	(((unsigned long int) (C)[(P) + 0] << 8) + \
								((unsigned long int) (C)[(P) + 1] << 0))

/* defines */
#define SHORT_DESCR_LEN        32
#define LONG_DESCR_LEN         256

/* enumeration for file formats */
#define JPEG2000FILENUM              4
typedef enum {

        JP2_FILE,
        J2K_FILE,
		MJ2_FILE,
		UNK_FILE

} jpeg2000filetype;

/* enumeration for the box types */
#define JPEG2000BOXNUM                23
typedef enum {

			FILE_BOX,
			JP_BOX,
			FTYP_BOX,
			JP2H_BOX,
			IHDR_BOX,
			COLR_BOX,
			JP2C_BOX,
			JP2I_BOX,
			XML_BOX,
			UUID_BOX,
			UINF_BOX,
			MOOV_BOX,
			MVHD_BOX,
			TRAK_BOX,
			TKHD_BOX,
			MDIA_BOX,
			MINF_BOX,
			STBL_BOX,
			STSD_BOX,
			MJP2_BOX,
			MDAT_BOX,
			ANY_BOX,
			UNK_BOX

} jpeg2000boxtype;

/* jpeg2000 family box signatures */
#define FILE_SIGN           ""
#define JP_SIGN             "jP\040\040"
#define FTYP_SIGN           "ftyp"
#define JP2H_SIGN           "jp2h"
#define IHDR_SIGN           "ihdr"
#define COLR_SIGN           "colr"
#define JP2C_SIGN           "jp2c"
#define JP2I_SIGN           "jp2i"
#define XML_SIGN            "xml\040"
#define UUID_SIGN           "uuid"
#define UINF_SIGN           "uinf"
#define MOOV_SIGN           "moov"
#define MVHD_SIGN           "mvhd"
#define TRAK_SIGN           "trak"
#define TKHD_SIGN           "tkhd"
#define MDIA_SIGN           "mdia"
#define MINF_SIGN           "minf"
#define VMHD_SIGN           "vmhd"
#define STBL_SIGN           "stbl"
#define STSD_SIGN           "stsd"
#define MJP2_SIGN           "mjp2"
#define MDAT_SIGN           "mdat"
#define ANY_SIGN 			""
#define UNK_SIGN            ""

/* the box structure itself */
struct jpeg2000boxdef {

        char                  value[5];                 /* hexadecimal value/string*/
		char                  name[SHORT_DESCR_LEN];    /* short description       */
		char                  descr[LONG_DESCR_LEN];    /* long  description       */
		int                   sbox;                     /* is it a superbox?       */
		int                   req[JPEG2000FILENUM];     /* mandatory box           */
		jpeg2000boxtype       ins;                      /* contained in box...     */

};

/* the possible boxes */
struct jpeg2000boxdef jpeg2000box[] =
{
/* sign */	{FILE_SIGN,
/* short */	"placeholder for nothing",
/* long */	"Nothing to say",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	FILE_BOX},

/* sign */	{JP_SIGN,
/* short */	"JPEG 2000 Signature box",
/* long */	"This box uniquely identifies the file as being part of the JPEG 2000 family of files",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	FILE_BOX},

/* sign */	{FTYP_SIGN,
/* short */	"File Type box",
/* long */	"This box specifies file type, version and compatibility information, including specifying if this file "
			"is a conforming JP2 file or if it can be read by a conforming JP2 reader",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	FILE_BOX},

/* sign */	{JP2H_SIGN,
/* short */	"JP2 Header box",
/* long */	"This box contains a series of boxes that contain header-type information about the file",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	FILE_BOX},

/* sign */	{IHDR_SIGN,
/* short */	"Image Header box",
/* long */	"This box specifies the size of the image and other related fields",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	JP2H_BOX},

/* sign */	{COLR_SIGN,
/* short */	"Colour Specification box",
/* long */	"This box specifies the colourspace of the image",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	JP2H_BOX},

/* sign */	{JP2C_SIGN,
/* short */	"Contiguous Codestream box",
/* long */	"This box contains the codestream as defined by Annex A",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	FILE_BOX},

/* sign */	{JP2I_SIGN,
/* short */	"Intellectual Property box",
/* long */	"This box contains intellectual property information about the image",
/* sbox */	0,
/* req */	{0, 0, 0},
/* ins */	FILE_BOX},

/* sign */	{XML_SIGN,
/* short */	"XML box",
/* long */	"This box provides a tool by which vendors can add XML formatted information to a JP2 file",
/* sbox */	0,
/* req */	{0, 0, 0},
/* ins */	FILE_BOX},

/* sign */	{UUID_SIGN,
/* short */	"UUID box",
/* long */	"This box provides a tool by which vendors can add additional information to a file "
			"without risking conflict with other vendors",
/* sbox */	0,
/* req */	{0, 0, 0},
/* ins */	FILE_BOX},

/* sign */	{UINF_SIGN,
/* short */	"UUID Info box",
/* long */	"This box provides a tool by which a vendor may provide access to additional information associated with a UUID",
/* sbox */	0,
/* req */	{0, 0, 0},
/* ins */	FILE_BOX},

/* sign */	{MOOV_SIGN,
/* short */	"Movie box",
/* long */	"This box contains the media data. In video tracks, this box would contain JPEG2000 video frames",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	FILE_BOX},

/* sign */	{MVHD_SIGN,
/* short */	"Movie Header box",
/* long */	"This box defines overall information which is media-independent, and relevant to the entire presentation "
			"considered as a whole",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	MOOV_BOX},

/* sign */	{TRAK_SIGN,
/* short */	"Track box",
/* long */	"This is a container box for a single track of a presentation. A presentation may consist of one or more tracks",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	MOOV_BOX},

/* sign */	{TKHD_SIGN,
/* short */	"Track Header box",
/* long */	"This box specifies the characteristics of a single track. Exactly one Track Header Box is contained in a track",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	TRAK_BOX},

/* sign */	{MDIA_SIGN,
/* short */	"Media box",
/* long */	"The media declaration container contains all the objects which declare information about the media data "
			"within a track",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	TRAK_BOX},

/* sign */	{MINF_SIGN,
/* short */	"Media Information box",
/* long */	"This box contains all the objects which declare characteristic information of the media in the track",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	MDIA_BOX},

/* sign */	{STBL_SIGN,
/* short */	"Sample Table box",
/* long */	"The sample table contains all the time and data indexing of the media samples in a track",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	MINF_BOX},

/* sign */	{STSD_SIGN,
/* short */	"Sample Description box",
/* long */	"The sample description table gives detailed information about the coding type used, and any initialization "
			"information needed for that coding",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	MINF_BOX},

/* sign */	{MJP2_SIGN,
/* short */	"MJP2 Sample Description box",
/* long */	"The MJP2 sample description table gives detailed information about the coding type used, and any initialization "
			"information needed for that coding",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	MINF_BOX},

/* sign */	{MDAT_SIGN,
/* short */	"Media Data box",
/* long */	"The meta-data for a presentation is stored in the single Movie Box which occurs at the top-level of a file",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	FILE_BOX},

/* sign */	{ANY_SIGN,
/* short */	"Any box",
/* long */	"All the existing boxes",
/* sbox */	0,
/* req */	{0, 0, 0},
/* ins */	FILE_BOX},

/* sign */	{UNK_SIGN,
/* short */	"Unknown Type box",
/* long */	"The signature is not recognised to be that of an existing box",
/* sbox */	0,
/* req */	{0, 0, 0},
/* ins */	ANY_BOX}

};

/* declaration */
int
jpeg2000_box_handler_function(jpeg2000boxtype boxtype, wxInputStream& stream, unsigned long int filepoint,
							  unsigned long int filelimit, int level, char *scansign,
							  unsigned long int *scanpoint);

#ifdef __WXMSW__
typedef unsigned __int64 int8byte;
#endif // __WXMSW__

#ifdef __WXGTK__
typedef unsigned long long int8byte;
#endif // __WXGTK__

/* internal mini-search for a box signature */
int
jpeg2000_file_parse(wxInputStream& stream, unsigned long int filepoint, unsigned long int filelimit, int level,
					char *scansign, unsigned long int *scanpoint)
{
	unsigned long int       LBox = 0x00000000;
	char                    TBox[5] = "\0\0\0\0";
	int8byte                XLBox = 0x0000000000000000;
	unsigned long int       box_length = 0;
	int                     last_box = 0, box_num = 0;
	int                     box_type = ANY_BOX;
	unsigned char           fourbytes[4];
	int                     box_number = 0;

	/* cycle all over the file */
	box_num = 0;
	last_box = 0;
	while (!last_box) {

		/* do not exceed file limit */
		if (filepoint >= filelimit)
			return (0);

		/* seek on file */
		if (stream.SeekI(filepoint, wxFromStart) == wxInvalidOffset)
			return (-1);

		/* read the mandatory LBox, 4 bytes */
		if (!stream.Read(fourbytes, 4)) {
			wxLogError(wxT("Problem reading LBox from the file (file ended?)"));
			return -1;
		};
		LBox = STREAM_TO_UINT32(fourbytes, 0);

		/* read the mandatory TBox, 4 bytes */
		if (!stream.Read(TBox, 4)) {
			wxLogError(wxT("Problem reading TBox from the file (file ended?)"));
			return -1;
		};

		/* look if scansign is got */
		if ((scansign != NULL) && (memcmp(TBox, scansign, 4) == 0)) {
			/* hack/exploit */
			// stop as soon as you find the level-th codebox
			if (box_number == level) {
				memcpy(scansign, "    ", 4);
				*scanpoint = filepoint;
				return (0);
			} else
				box_number++;

		};

		/* determine the box type */
		for (box_type = JP_BOX; box_type < UNK_BOX; box_type++)
			if (memcmp(TBox, jpeg2000box[box_type].value, 4) == 0)
				break;	

		/* read the optional XLBox, 8 bytes */
		if (LBox == 1) {

			if (!stream.Read(&XLBox, 8)) {
				wxLogError(wxT("Problem reading XLBox from the file (file ended?)"));
				return -1;
			};
			box_length = (unsigned long int) BYTE_SWAP8(XLBox);

		} else if (LBox == 0x00000000) {

			/* last box in file */
			last_box = 1; 
			box_length = filelimit - filepoint;

		} else

			box_length = LBox;


		/* go deep in the box */
		jpeg2000_box_handler_function((jpeg2000boxtype) box_type,
			stream, (LBox == 1) ? (filepoint + 16) : (filepoint + 8),
			filepoint + box_length, level, scansign, scanpoint);

		/* if it's a superbox go inside it */
		if (jpeg2000box[box_type].sbox)
			jpeg2000_file_parse(stream, (LBox == 1) ? (filepoint + 16) : (filepoint + 8), filepoint + box_length,
				level, scansign, scanpoint);

		/* increment box number and filepoint*/
		box_num++;
		filepoint += box_length;

	};

	/* all good */
	return (0);
}

// search first contiguos codestream box in an mj2 file
unsigned long int
searchjpeg2000c(wxInputStream& stream, unsigned long int fsize, int number)
{
	char scansign[] = "jp2c";
	unsigned long int scanpoint = 0L;

	wxLogMessage(wxT("Searching jp2c box... "));

	/* do the parsing */
	if (jpeg2000_file_parse(stream, 0, fsize, number, scansign, &scanpoint) < 0)		
		wxLogMessage(wxT("Unrecoverable error during JPEG 2000 box parsing: stopping"));

	if (strcmp(scansign, "    "))
		wxLogMessage(wxT("Box not found"));
	else {

		wxLogMessage(wxString::Format(wxT("Box found at byte %d"), scanpoint));

	};

	return (scanpoint);
}

// search the jp2h box in the file
unsigned long int
searchjpeg2000headerbox(wxInputStream& stream, unsigned long int fsize)
{
	char scansign[] = "jp2h";
	unsigned long int scanpoint = 0L;

	wxLogMessage(wxT("Searching jp2h box... "));

	/* do the parsing */
	if (jpeg2000_file_parse(stream, 0, fsize, 0, scansign, &scanpoint) < 0)		
		wxLogMessage(wxT("Unrecoverable error during JPEG 2000 box parsing: stopping"));

	if (strcmp(scansign, "    "))
		wxLogMessage(wxT("Box not found"));
	else
		wxLogMessage(wxString::Format(wxT("Box found at byte %d"), scanpoint));

	return (scanpoint);
}

/* handling functions */
#define ITEM_PER_ROW	10

/* Box handler function */
int
jpeg2000_box_handler_function(jpeg2000boxtype boxtype, wxInputStream& stream, unsigned long int filepoint,
							  unsigned long int filelimit, int level,
							  char *scansign, unsigned long int *scanpoint)
{
	switch (boxtype) {

	/* Sample Description box */
	case (STSD_BOX):
		jpeg2000_file_parse(stream, filepoint + 8, filelimit, level, scansign, scanpoint);
		break;

	/* MJP2 Sample Description box */
	case (MJP2_BOX):
		jpeg2000_file_parse(stream, filepoint + 78, filelimit, level, scansign, scanpoint);
		break;
		
	/* not yet implemented */
	default:
		break;

	};

	return (0);
}

// the jP and ftyp parts of the header
#define jpeg2000headSIZE	32
unsigned char jpeg2000head[jpeg2000headSIZE] = {
		0x00, 0x00, 0x00, 0x0C,  'j',  'P',  ' ',  ' ',
		0x0D, 0x0A, 0x87, 0x0A, 0x00, 0x00, 0x00, 0x14,
		 'f',  't',  'y',  'p',  'j',  'p',  '2',  ' ',
		0x00, 0x00, 0x00, 0x00,  'j',  'p',  '2',  ' '			
};

/////////////////////////////////////////////////
/////////////////////////////////////////////////

// load the jpeg2000 file format
bool wxJPEG2000Handler::LoadFile(wxImage *image, wxInputStream& stream, bool verbose, int index)
{
	opj_dparameters_t parameters;	/* decompression parameters */
	opj_event_mgr_t event_mgr;		/* event manager */
	opj_image_t *opjimage = NULL;
	unsigned char *src = NULL;
    unsigned char *ptr;
	int file_length, jp2c_point, jp2h_point;
	unsigned long int jp2hboxlen, jp2cboxlen;
	opj_codestream_info_t cstr_info;  /* Codestream information structure */
    unsigned char hdr[24];
	int jpfamform;

	// destroy the image
    image->Destroy();

	/* read the beginning of the file to check the type */ 
    if (!stream.Read(hdr, WXSIZEOF(hdr)))
        return false;
	if ((jpfamform = jpeg2000familytype(hdr, WXSIZEOF(hdr))) < 0)
		return false;
	stream.SeekI(0, wxFromStart);

	/* handle to a decompressor */
	opj_dinfo_t* dinfo = NULL;	
	opj_cio_t *cio = NULL;

	/* configure the event callbacks */
	memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
	event_mgr.error_handler = jpeg2000_error_callback;
	event_mgr.warning_handler = jpeg2000_warning_callback;
	event_mgr.info_handler = jpeg2000_info_callback;

	/* set decoding parameters to default values */
	opj_set_default_decoder_parameters(&parameters);

	/* prepare parameters */
	strncpy(parameters.infile, "", sizeof(parameters.infile) - 1);
	strncpy(parameters.outfile, "", sizeof(parameters.outfile) - 1);
	parameters.decod_format = jpfamform;
	parameters.cod_format = BMP_DFMT;
	if (m_reducefactor)
		parameters.cp_reduce = m_reducefactor;
	if (m_qualitylayers)
		parameters.cp_layer = m_qualitylayers;
	/*if (n_components)
		parameters. = n_components;*/

	/* JPWL only */
#ifdef USE_JPWL
	parameters.jpwl_exp_comps = m_expcomps;
	parameters.jpwl_max_tiles = m_maxtiles;
	parameters.jpwl_correct = m_enablejpwl;
#endif /* USE_JPWL */

	/* get a decoder handle */
	if (jpfamform == JP2_CFMT || jpfamform == MJ2_CFMT)
		dinfo = opj_create_decompress(CODEC_JP2);
	else if (jpfamform == J2K_CFMT)
		dinfo = opj_create_decompress(CODEC_J2K);
	else
		return false;

	/* find length of the stream */
	stream.SeekI(0, wxFromEnd);
	file_length = (int) stream.TellI();

	/* it's a movie */
	if (jpfamform == MJ2_CFMT) {
		/* search for the first codestream box and the movie header box  */
		jp2c_point = searchjpeg2000c(stream, file_length, m_framenum);
		jp2h_point = searchjpeg2000headerbox(stream, file_length);

		// read the jp2h box and store it
		stream.SeekI(jp2h_point, wxFromStart);
		stream.Read(&jp2hboxlen, sizeof(unsigned long int));
		jp2hboxlen = BYTE_SWAP4(jp2hboxlen);

		// read the jp2c box and store it
		stream.SeekI(jp2c_point, wxFromStart);
		stream.Read(&jp2cboxlen, sizeof(unsigned long int));
		jp2cboxlen = BYTE_SWAP4(jp2cboxlen);

		// malloc memory source
		src = (unsigned char *) malloc(jpeg2000headSIZE + jp2hboxlen + jp2cboxlen);

		// copy the jP and ftyp
		memcpy(src, jpeg2000head, jpeg2000headSIZE);

		// copy the jp2h
		stream.SeekI(jp2h_point, wxFromStart);
		stream.Read(&src[jpeg2000headSIZE], jp2hboxlen);

		// copy the jp2c
		stream.SeekI(jp2c_point, wxFromStart);
		stream.Read(&src[jpeg2000headSIZE + jp2hboxlen], jp2cboxlen);
	} else 	if (jpfamform == JP2_CFMT || jpfamform == J2K_CFMT) {
		/* It's a plain image */
		/* get data */
		stream.SeekI(0, wxFromStart);
		src = (unsigned char *) malloc(file_length);
		stream.Read(src, file_length);
	} else
		return false;

	/* catch events using our callbacks and give a local context */
	opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);

	/* setup the decoder decoding parameters using user parameters */
	opj_setup_decoder(dinfo, &parameters);

	/* open a byte stream */
	if (jpfamform == MJ2_CFMT)
		cio = opj_cio_open((opj_common_ptr)dinfo, src, jpeg2000headSIZE + jp2hboxlen + jp2cboxlen);
	else if (jpfamform == JP2_CFMT || jpfamform == J2K_CFMT)
		cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);
	else {
		free(src);
		return false;
	}

	/* decode the stream and fill the image structure */
	opjimage = opj_decode_with_info(dinfo, cio, &cstr_info);
	if (!opjimage) {
		wxMutexGuiEnter();
		wxLogError(wxT("JPEG 2000 failed to decode image!"));
		wxMutexGuiLeave();
		opj_destroy_decompress(dinfo);
		opj_cio_close(cio);
		free(src);
		return false;
	}

	/* close the byte stream */
	opj_cio_close(cio);

	/*

	- At this point, we have the structure "opjimage" that is filled with decompressed
	  data, as processed by the OpenJPEG decompression engine

	- We need to fill the class "image" with the proper pixel sample values

	*/
	{
		int shiftbpp;
		int c, tempcomps;

		// check components number
		if (m_components > opjimage->numcomps)
			m_components = opjimage->numcomps;

		// check image depth (only on the first one, for now)
		if (m_components)
			shiftbpp = opjimage->comps[m_components - 1].prec - 8;
		else
			shiftbpp = opjimage->comps[0].prec - 8;

		// prepare image size
		if (m_components)
			image->Create(opjimage->comps[m_components - 1].w, opjimage->comps[m_components - 1].h, true);
		else
			image->Create(opjimage->comps[0].w, opjimage->comps[0].h, true);

		// access image raw data
		image->SetMask(false);
		ptr = image->GetData();

		// workaround for components different from 1 or 3
		if ((opjimage->numcomps != 1) && (opjimage->numcomps != 3)) {
#ifndef __WXGTK__ 
			wxMutexGuiEnter();
#endif /* __WXGTK__ */
			wxLogMessage(wxT("JPEG2000: weird number of components"));
#ifndef __WXGTK__ 
			wxMutexGuiLeave();
#endif /* __WXGTK__ */
			tempcomps = 1;
		} else
			tempcomps = opjimage->numcomps;

		// workaround for subsampled components
		for (c = 1; c < tempcomps; c++) {
			if ((opjimage->comps[c].w != opjimage->comps[c - 1].w) || (opjimage->comps[c].h != opjimage->comps[c - 1].h)) {
				tempcomps = 1;
				break;
			}
		}

		// workaround for different precision components
		for (c = 1; c < tempcomps; c++) {
			if (opjimage->comps[c].bpp != opjimage->comps[c - 1].bpp) {
				tempcomps = 1;
				break;
			}
		}

		// only one component selected
		if (m_components)
			tempcomps = 1;

		// RGB color picture
		if (tempcomps == 3) {
			int row, col;
			int *r = opjimage->comps[0].data;
			int *g = opjimage->comps[1].data;
			int *b = opjimage->comps[2].data;
			if (shiftbpp > 0) {
				for (row = 0; row < opjimage->comps[0].h; row++) {
					for (col = 0; col < opjimage->comps[0].w; col++) {
						
						*(ptr++) = (*(r++)) >> shiftbpp;
						*(ptr++) = (*(g++)) >> shiftbpp;
						*(ptr++) = (*(b++)) >> shiftbpp;

					}
				}

			} else if (shiftbpp < 0) {
				for (row = 0; row < opjimage->comps[0].h; row++) {
					for (col = 0; col < opjimage->comps[0].w; col++) {
						
						*(ptr++) = (*(r++)) << -shiftbpp;
						*(ptr++) = (*(g++)) << -shiftbpp;
						*(ptr++) = (*(b++)) << -shiftbpp;

					}
				}
				
			} else {
				for (row = 0; row < opjimage->comps[0].h; row++) {
					for (col = 0; col < opjimage->comps[0].w; col++) {

						*(ptr++) = *(r++);
						*(ptr++) = *(g++);
						*(ptr++) = *(b++);
					
					}
				}
			}
		}

		// B/W picture
		if (tempcomps == 1) {
			int row, col;
			int selcomp;

			if (m_components)
				selcomp = m_components - 1;
			else
				selcomp = 0;

			int *y = opjimage->comps[selcomp].data;
			if (shiftbpp > 0) {
				for (row = 0; row < opjimage->comps[selcomp].h; row++) {
					for (col = 0; col < opjimage->comps[selcomp].w; col++) {
						
						*(ptr++) = (*(y)) >> shiftbpp;
						*(ptr++) = (*(y)) >> shiftbpp;
						*(ptr++) = (*(y++)) >> shiftbpp;

					}
				}
			} else if (shiftbpp < 0) {
				for (row = 0; row < opjimage->comps[selcomp].h; row++) {
					for (col = 0; col < opjimage->comps[selcomp].w; col++) {
						
						*(ptr++) = (*(y)) << -shiftbpp;
						*(ptr++) = (*(y)) << -shiftbpp;
						*(ptr++) = (*(y++)) << -shiftbpp;

					}
				}
			} else {
				for (row = 0; row < opjimage->comps[selcomp].h; row++) {
					for (col = 0; col < opjimage->comps[selcomp].w; col++) {
						
						*(ptr++) = *(y);
						*(ptr++) = *(y);
						*(ptr++) = *(y++);

					}
				}
			}
		}


	}

    wxMutexGuiEnter();
    wxLogMessage(wxT("JPEG 2000 image loaded."));
    wxMutexGuiLeave();

	/* close openjpeg structs */
	opj_destroy_decompress(dinfo);
	opj_image_destroy(opjimage);
	free(src);

	if (!image->Ok())
		return false;
	else
		return true;

}

#define CINEMA_24_CS 1302083	/* Codestream length for 24fps */
#define CINEMA_48_CS 651041		/* Codestream length for 48fps */
#define COMP_24_CS 1041666		/* Maximum size per color component for 2K & 4K @ 24fps */
#define COMP_48_CS 520833		/* Maximum size per color component for 2K @ 48fps */

// save the j2k codestream
bool wxJPEG2000Handler::SaveFile( wxImage *wimage, wxOutputStream& stream, bool verbose )
{
        opj_cparameters_t parameters;	/* compression parameters */
        opj_event_mgr_t event_mgr;		/* event manager */
        opj_image_t *oimage = NULL;
        opj_image_cmptparm_t *cmptparm;	
        opj_cio_t *cio = NULL;
        opj_codestream_info_t cstr_info;
        int codestream_length;
        bool bSuccess;
        int i;
        char indexfilename[OPJ_PATH_LEN] = "";	/* index file name */

        /*
        configure the event callbacks (not required)
        setting of each callback is optionnal
        */
        memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
        event_mgr.error_handler = jpeg2000_error_callback;
        event_mgr.warning_handler = jpeg2000_warning_callback;
        event_mgr.info_handler = jpeg2000_info_callback;

        /* set encoding parameters to default values */
        opj_set_default_encoder_parameters(&parameters);

        /* load parameters */
        parameters.cp_cinema = OFF;

        /* subsampling */
        if (sscanf(m_subsampling.ToAscii(), "%d,%d", &(parameters.subsampling_dx), &(parameters.subsampling_dy)) != 2) {
                wxLogError(wxT("Wrong sub-sampling encoder setting: dx,dy"));
                return false;
        }

        /* compression rates */
        if ((m_rates != wxT("")) && (!m_enablequality)) {
                const char *s1 = m_rates.ToAscii();
                wxLogMessage(wxT("rates %s"), s1);
                while (sscanf(s1, "%f", &(parameters.tcp_rates[parameters.tcp_numlayers])) == 1) {
                        parameters.tcp_numlayers++;
                        while (*s1 && *s1 != ',') {
                                s1++;
                        }
                        if (!*s1)
                                break;
                        s1++;
                }
                wxLogMessage(wxT("%d layers"), parameters.tcp_numlayers);
                parameters.cp_disto_alloc = 1;
        }

        /* image quality, dB */
        if ((m_quality != wxT("")) && (m_enablequality)) {
                const char *s2 = m_quality.ToAscii();
                wxLogMessage(wxT("qualities %s"), s2);
                while (sscanf(s2, "%f", &parameters.tcp_distoratio[parameters.tcp_numlayers]) == 1) {
                        parameters.tcp_numlayers++;
                        while (*s2 && *s2 != ',') {
                                s2++;
                        }
                        if (!*s2)
                                break;
                        s2++;
                }
                wxLogMessage(wxT("%d layers"), parameters.tcp_numlayers);
                parameters.cp_fixed_quality = 1;
        }

        /* image origin */
        if (sscanf(m_origin.ToAscii(), "%d,%d", &parameters.image_offset_x0, &parameters.image_offset_y0) != 2) {
                wxLogError(wxT("bad coordinate of the image origin: x0,y0"));
                return false;
        }
                                
        /* Create comment for codestream */
        if(m_enablecomm) {
                parameters.cp_comment = (char *) malloc(strlen(m_comment.ToAscii()) + 1);
                if(parameters.cp_comment) {
                        strcpy(parameters.cp_comment, m_comment.ToAscii());
                }
        } else {
                parameters.cp_comment = NULL;
        }

        /* indexing file */
        if (m_enableidx) {
                strncpy(indexfilename, m_index.ToAscii(), OPJ_PATH_LEN);
                wxLogMessage(wxT("index file is %s"), indexfilename);
        }

        /* if no rate entered, lossless by default */
        if (parameters.tcp_numlayers == 0) {
                parameters.tcp_rates[0] = 0;	/* MOD antonin : losslessbug */
                parameters.tcp_numlayers++;
                parameters.cp_disto_alloc = 1;
        }

        /* irreversible transform */
        parameters.irreversible = (m_irreversible == true) ? 1 : 0;

        /* resolutions */
        parameters.numresolution = m_resolutions;

        /* codeblocks size */
        if (m_cbsize != wxT("")) {
                int cblockw_init = 0, cblockh_init = 0;
                sscanf(m_cbsize.ToAscii(), "%d,%d", &cblockw_init, &cblockh_init);
                if (cblockw_init * cblockh_init > 4096 || cblockw_init > 1024 || cblockw_init < 4 || cblockh_init > 1024 || cblockh_init < 4) {
                        wxLogError(wxT("!! Size of code_block error !! Restrictions:\n  width*height<=4096\n  4<=width,height<= 1024"));
                        return false;
                }
                parameters.cblockw_init = cblockw_init;
                parameters.cblockh_init = cblockh_init;
        }

        /* precincts size */
        if (m_prsize != wxT("")) {
                char sep;
                int res_spec = 0;
                char *s = (char *) m_prsize.c_str();
                do {
                        sep = 0;
                        sscanf(s, "[%d,%d]%c", &parameters.prcw_init[res_spec], &parameters.prch_init[res_spec], &sep);
                        parameters.csty |= 0x01;
                        res_spec++;
                        s = strpbrk(s, "]") + 2;
                } while (sep == ',');
                parameters.res_spec = res_spec;
        }

        /* tiles */
        if (m_tsize != wxT("")) {
                sscanf(m_tsize.ToAscii(), "%d,%d", &parameters.cp_tdx, &parameters.cp_tdy);
                parameters.tile_size_on = true;
        }

        /* tile origin */
        if (sscanf(m_torigin.ToAscii(), "%d,%d", &parameters.cp_tx0, &parameters.cp_ty0) != 2) {
                wxLogError(wxT("tile offset setting error: X0,Y0"));
                return false;
        }

        /* use SOP */
        if (m_enablesop)
                parameters.csty |= 0x02;

        /* use EPH */
        if (m_enableeph)
                parameters.csty |= 0x04;

        /* multiple component transform */
        if (m_multicomp)
                parameters.tcp_mct = 1;
        else
                parameters.tcp_mct = 0;

        /* mode switch */
        parameters.mode = (m_enablebypass ? 1 : 0) + (m_enablereset ? 2 : 0)
                + (m_enablerestart ? 4 : 0) + (m_enablevsc ? 8 : 0)
                + (m_enableerterm ? 16 : 0) + (m_enablesegmark ? 32 : 0);

        /* progression order */
        switch (m_progression) {

                /* LRCP */
        case 0:
                parameters.prog_order = LRCP;
                break;

                /* RLCP */
        case 1:
                parameters.prog_order = RLCP;
                break;

                /* RPCL */
        case 2:
                parameters.prog_order = RPCL;
                break;

                /* PCRL */
        case 3:
                parameters.prog_order = PCRL;
                break;

                /* CPRL */
        case 4:
                parameters.prog_order = CPRL;
                break;

                /* DCI2K24 */
        case 5:
                parameters.cp_cinema = CINEMA2K_24;
                parameters.cp_rsiz = CINEMA2K;
                break;

                /* DCI2K48 */
        case 6:
                parameters.cp_cinema = CINEMA2K_48;
                parameters.cp_rsiz = CINEMA2K;
                break;

                /* DCI4K */
        case 7:
                parameters.cp_cinema = CINEMA4K_24;
                parameters.cp_rsiz = CINEMA4K;
                break;

        default:
                break;
        }

        /* check cinema */
        if (parameters.cp_cinema) {

                /* set up */
                parameters.tile_size_on = false;
                parameters.cp_tdx=1;
                parameters.cp_tdy=1;
                
                /*Tile part*/
                parameters.tp_flag = 'C';
                parameters.tp_on = 1;

                /*Tile and Image shall be at (0,0)*/
                parameters.cp_tx0 = 0;
                parameters.cp_ty0 = 0;
                parameters.image_offset_x0 = 0;
                parameters.image_offset_y0 = 0;

                /*Codeblock size= 32*32*/
                parameters.cblockw_init = 32;	
                parameters.cblockh_init = 32;
                parameters.csty |= 0x01;

                /*The progression order shall be CPRL*/
                parameters.prog_order = CPRL;

                /* No ROI */
                parameters.roi_compno = -1;

                parameters.subsampling_dx = 1;
                parameters.subsampling_dy = 1;

                /* 9-7 transform */
                parameters.irreversible = 1;

        }				

        /* convert wx image into opj image */
        cmptparm = (opj_image_cmptparm_t*) malloc(3 * sizeof(opj_image_cmptparm_t));

        /* initialize opj image components */	
        memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
        for(i = 0; i < 3; i++) {		
                cmptparm[i].prec = 8;
                cmptparm[i].bpp = 8;
                cmptparm[i].sgnd = false;
                cmptparm[i].dx = parameters.subsampling_dx;
                cmptparm[i].dy = parameters.subsampling_dy;
                cmptparm[i].w = wimage->GetWidth();
                cmptparm[i].h = wimage->GetHeight();
        }

        /* create the image */
        oimage = opj_image_create(3, &cmptparm[0], CLRSPC_SRGB);
        if(!oimage) {
                if (cmptparm)
                        free(cmptparm);
                return false;
        }

        /* set image offset and reference grid */
        oimage->x0 = parameters.image_offset_x0;
        oimage->y0 = parameters.image_offset_y0;
        oimage->x1 = parameters.image_offset_x0 + (wimage->GetWidth() - 1) * 1 + 1;
        oimage->y1 = parameters.image_offset_y0 + (wimage->GetHeight() - 1) * 1 + 1;

        /* load image data */
        unsigned char *value = wimage->GetData(); 
        int area = wimage->GetWidth() * wimage->GetHeight();
        for (i = 0; i < area; i++) {
                        oimage->comps[0].data[i] = *(value++);
                        oimage->comps[1].data[i] = *(value++);
                        oimage->comps[2].data[i] = *(value++);
        }

        /* check cinema again */
        if (parameters.cp_cinema) {
                int i;
                float temp_rate;
                opj_poc_t *POC = NULL;

                switch (parameters.cp_cinema) {

                case CINEMA2K_24:
                case CINEMA2K_48:
                        if (parameters.numresolution > 6) {
                                parameters.numresolution = 6;
                        }
                        if (!((oimage->comps[0].w == 2048) | (oimage->comps[0].h == 1080))) {
                                wxLogWarning(wxT("Image coordinates %d x %d is not 2K compliant. JPEG Digital Cinema Profile-3 "
                                        "(2K profile) compliance requires that at least one of coordinates match 2048 x 1080"),
                                        oimage->comps[0].w, oimage->comps[0].h);
                                parameters.cp_rsiz = STD_RSIZ;
                        }
                break;
                
                case CINEMA4K_24:
                        if (parameters.numresolution < 1) {
                                        parameters.numresolution = 1;
                        } else if (parameters.numresolution > 7) {
                                        parameters.numresolution = 7;
                        }
                        if (!((oimage->comps[0].w == 4096) | (oimage->comps[0].h == 2160))) {
                                wxLogWarning(wxT("Image coordinates %d x %d is not 4K compliant. JPEG Digital Cinema Profile-4" 
                                        "(4K profile) compliance requires that at least one of coordinates match 4096 x 2160"),
                                        oimage->comps[0].w, oimage->comps[0].h);
                                parameters.cp_rsiz = STD_RSIZ;
                        }
                        parameters.POC[0].tile  = 1; 
                        parameters.POC[0].resno0  = 0; 
                        parameters.POC[0].compno0 = 0;
                        parameters.POC[0].layno1  = 1;
                        parameters.POC[0].resno1  = parameters.numresolution - 1;
                        parameters.POC[0].compno1 = 3;
                        parameters.POC[0].prg1 = CPRL;
                        parameters.POC[1].tile  = 1;
                        parameters.POC[1].resno0  = parameters.numresolution - 1; 
                        parameters.POC[1].compno0 = 0;
                        parameters.POC[1].layno1  = 1;
                        parameters.POC[1].resno1  = parameters.numresolution;
                        parameters.POC[1].compno1 = 3;
                        parameters.POC[1].prg1 = CPRL;
                        parameters.numpocs = 2;
                        break;
                }

                switch (parameters.cp_cinema) {
                case CINEMA2K_24:
                case CINEMA4K_24:
                        for (i = 0 ; i < parameters.tcp_numlayers; i++) {
                                temp_rate = 0;
                                if (parameters.tcp_rates[i] == 0) {
                                        parameters.tcp_rates[0] = ((float) (oimage->numcomps * oimage->comps[0].w * oimage->comps[0].h * oimage->comps[0].prec)) / 
                                        (CINEMA_24_CS * 8 * oimage->comps[0].dx * oimage->comps[0].dy);
                                }else{
                                        temp_rate = ((float) (oimage->numcomps * oimage->comps[0].w * oimage->comps[0].h * oimage->comps[0].prec)) / 
                                                (parameters.tcp_rates[i] * 8 * oimage->comps[0].dx * oimage->comps[0].dy);
                                        if (temp_rate > CINEMA_24_CS ) {
                                                parameters.tcp_rates[i]= ((float) (oimage->numcomps * oimage->comps[0].w * oimage->comps[0].h * oimage->comps[0].prec)) / 
                                                (CINEMA_24_CS * 8 * oimage->comps[0].dx * oimage->comps[0].dy);
                                        } else {
                                                /* do nothing */
                                        }
                                }
                        }
                        parameters.max_comp_size = COMP_24_CS;
                        break;
                        
                case CINEMA2K_48:
                        for (i = 0; i < parameters.tcp_numlayers; i++) {
                                temp_rate = 0 ;
                                if (parameters.tcp_rates[i] == 0) {
                                        parameters.tcp_rates[0] = ((float) (oimage->numcomps * oimage->comps[0].w * oimage->comps[0].h * oimage->comps[0].prec)) / 
                                        (CINEMA_48_CS * 8 * oimage->comps[0].dx * oimage->comps[0].dy);
                                }else{
                                        temp_rate =((float) (oimage->numcomps * oimage->comps[0].w * oimage->comps[0].h * oimage->comps[0].prec)) / 
                                                (parameters.tcp_rates[i] * 8 * oimage->comps[0].dx * oimage->comps[0].dy);
                                        if (temp_rate > CINEMA_48_CS ){
                                                parameters.tcp_rates[0]= ((float) (oimage->numcomps * oimage->comps[0].w * oimage->comps[0].h * oimage->comps[0].prec)) / 
                                                (CINEMA_48_CS * 8 * oimage->comps[0].dx * oimage->comps[0].dy);
                                        }else{
                                                /* do nothing */
                                        }
                                }
                        }
                        parameters.max_comp_size = COMP_48_CS;
                        break;
                }

                parameters.cp_disto_alloc = 1;
        }
        
        /* get a J2K compressor handle */
        opj_cinfo_t* cinfo = opj_create_compress(CODEC_J2K);

        /* catch events using our callbacks and give a local context */
        opj_set_event_mgr((opj_common_ptr)cinfo, &event_mgr, stderr);

        /* setup the encoder parameters using the current image and user parameters */
        opj_setup_encoder(cinfo, &parameters, oimage);

        /* open a byte stream for writing */
        /* allocate memory for all tiles */
        cio = opj_cio_open((opj_common_ptr)cinfo, NULL, 0);

        /* encode the image */
        bSuccess = opj_encode_with_info(cinfo, cio, oimage, &cstr_info);
        if (!bSuccess) {

                opj_cio_close(cio);
                opj_destroy_compress(cinfo);
                opj_image_destroy(oimage);
                if (cmptparm)
                        free(cmptparm);
                if(parameters.cp_comment)
                        free(parameters.cp_comment);
                if(parameters.cp_matrice)
                        free(parameters.cp_matrice);

#ifndef __WXGTK__ 
    wxMutexGuiEnter();
#endif /* __WXGTK__ */

                wxLogError(wxT("failed to encode image"));

#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */

                return false;
        }
        codestream_length = cio_tell(cio);
        wxLogMessage(wxT("Codestream: %d bytes"), codestream_length);

        /* write the buffer to stream */
        stream.Write(cio->buffer, codestream_length);

        /* close and free the byte stream */
        opj_cio_close(cio);

        /* Write the index to disk */
        if (*indexfilename) {
                if (write_index_file(&cstr_info, indexfilename)) {
                        wxLogError(wxT("Failed to output index file"));
                }
        }

        /* free remaining compression structures */
        opj_destroy_compress(cinfo);

        /* free image data */
        opj_image_destroy(oimage);

        if (cmptparm)
                free(cmptparm);
        if(parameters.cp_comment)
                free(parameters.cp_comment);
        if(parameters.cp_matrice)
                free(parameters.cp_matrice);

#ifndef __WXGTK__ 
    wxMutexGuiEnter();
#endif /* __WXGTK__ */

    wxLogMessage(wxT("J2K: Image encoded!"));

#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */

    return true;
}

#ifdef __VISUALC__
    #pragma warning(default:4611)
#endif /* VC++ */

// recognize the JPEG 2000 family starting box or the 0xFF4F JPEG 2000 SOC marker
bool wxJPEG2000Handler::DoCanRead(wxInputStream& stream)
{
    unsigned char hdr[24];
	int jpfamform;

    if ( !stream.Read(hdr, WXSIZEOF(hdr)) )
        return false;

	jpfamform = jpeg2000familytype(hdr, WXSIZEOF(hdr));

	return ((jpfamform == JP2_CFMT) || (jpfamform == MJ2_CFMT) || (jpfamform == J2K_CFMT));
}

#endif   // wxUSE_STREAMS

#endif   // wxUSE_LIBOPENJPEG
