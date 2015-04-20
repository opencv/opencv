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
#include "OPJViewer.h"

/* defines */
#define SHORT_DESCR_LEN        32
#define LONG_DESCR_LEN         256

/* enumeration for file formats */
#define J2FILENUM              4
typedef enum {

        JP2_FILE,
        J2K_FILE,
		MJ2_FILE,
		UNK_FILE

} j2filetype;

/* enumeration for the box types */
#define j22boxNUM                23
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
			MDHD_BOX,
			HDLR_BOX,
			MINF_BOX,
			VMHD_BOX,
			STBL_BOX,
			STSD_BOX,
			STSZ_BOX,
			MJP2_BOX,
			MDAT_BOX,
			ANY_BOX,
			UNK_BOX

} j22boxtype;

/* the box structure itself */
struct boxdef {

        char                  value[5];                 /* hexadecimal value/string*/
		char                  name[SHORT_DESCR_LEN];    /* short description       */
		char                  descr[LONG_DESCR_LEN];    /* long  description       */
		int                   sbox;                     /* is it a superbox?       */
		int                   req[J2FILENUM];           /* mandatory box           */
		j22boxtype             ins;                      /* contained in box...     */

};


/* jp2 family box signatures */
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
#define MDHD_SIGN           "mdhd"
#define HDLR_SIGN           "hdlr"
#define MINF_SIGN           "minf"
#define VMHD_SIGN           "vmhd"
#define STBL_SIGN           "stbl"
#define STSD_SIGN           "stsd"
#define STSZ_SIGN           "stsz"
#define MJP2_SIGN           "mjp2"
#define MDAT_SIGN           "mdat"
#define ANY_SIGN 			""
#define UNK_SIGN            ""

/* the possible boxes */
struct boxdef j22box[] =
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

/* sign */	{MDHD_SIGN,
/* short */	"Media Header box",
/* long */	"The media header declares overall information which is media-independent, and relevant to characteristics "
			"of the media in a track",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	MDIA_BOX},

/* sign */	{HDLR_SIGN,
/* short */	"Handler Reference box",
/* long */	"This box within a Media Box declares the process by which the media-data in the track may be presented, "
			"and thus, the nature of the media in a track",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	MDIA_BOX},

/* sign */	{MINF_SIGN,
/* short */	"Media Information box",
/* long */	"This box contains all the objects which declare characteristic information of the media in the track",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	MDIA_BOX},

/* sign */	{VMHD_SIGN,
/* short */	"Video Media Header box",
/* long */	"The video media header contains general presentation information, independent of the coding, for video media",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	MINF_BOX},

/* sign */	{STBL_SIGN,
/* short */	"Sample Table box",
/* long */	"The sample table contains all the time and data indexing of the media samples in a track",
/* sbox */	1,
/* req */	{1, 1, 1},
/* ins */	MINF_BOX},

/* sign */	{STSD_SIGN,
/* short */	"STSD Sample Description box",
/* long */	"The sample description table gives detailed information about the coding type used, and any initialization "
			"information needed for that coding",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	MINF_BOX},

/* sign */	{STSZ_SIGN,
/* short */	"Sample Size box",
/* long */	"This box contains the sample count and a table giving the size of each sample",
/* sbox */	0,
/* req */	{1, 1, 1},
/* ins */	STBL_BOX},

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

#define OPJREAD_LONG(F,L,N) { \
							if (F->Read(fourbytes, 4) < 4) { \
								wxLogMessage(wxT("Problem reading " N " from the file (file ended?)")); \
								return -1; \
							}; \
							L = STREAM_TO_UINT32(fourbytes, 0); \
							}

/* handling functions */
#define ITEM_PER_ROW	10

//#define indprint	if (0) printf("%.*s", 2 * level + 9, indent), printf
char    indent[] =  "                                                                   "
					"                                                                   "
					"                                                                   "
					"                                                                   ";

void indprint(wxString printout, int level)
{
	wxLogMessage(/*wxString::Format(wxT("%.*s"), 2 * level + 9, indent) + */printout);
}

/* Box handler function */
int OPJParseThread::box_handler_function(int boxtype, wxFile *fileid, wxFileOffset filepoint, wxFileOffset filelimit,
						 wxTreeItemId parentid, int level, char *scansign, unsigned long int *scanpoint)
{
	switch ((j22boxtype) boxtype) {


	/* JPEG 2000 Signature box */
	case (JP_BOX): {

			unsigned long int checkdata = 0;
			fileid->Read(&checkdata, sizeof(unsigned long int));
			checkdata = BYTE_SWAP4(checkdata);

			// add info
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Check data: %X -> %s"), checkdata, (checkdata == 0x0D0A870A) ? wxT("OK") : wxT("KO")),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

		};
		break;


	/* JPEG 2000 codestream box */
	case (JP2C_BOX): {

			// add info
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString(wxT("Codestream")),
				m_tree->TreeCtrlIcon_Folder, m_tree->TreeCtrlIcon_Folder + 1,
				new OPJMarkerData(wxT("INFO-CSTREAM"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
				);

			m_tree->SetItemHasChildren(currid);

			// parse the file
			//ParseJ2KFile(fileid, filepoint, filelimit, currid);

		};
		break;





	/* File Type box */
	case (FTYP_BOX): {

			char BR[4], CL[4];
			unsigned long int MinV, numCL, i;
			fileid->Read(BR, sizeof(char) * 4);
			fileid->Read(&MinV, sizeof(unsigned long int));
			MinV = BYTE_SWAP4(MinV);
			numCL = (filelimit - fileid->Tell()) / 4;				

			// add info
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxT("Brand/Minor version: ") +
				wxString::FromAscii(BR).Truncate(4) +
				wxString::Format(wxT("/%d"), MinV),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Compatibility list")),
				m_tree->TreeCtrlIcon_Folder, m_tree->TreeCtrlIcon_Folder + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			for (i = 0; i < numCL; i++) {
				fileid->Read(CL, sizeof(char) * 4);
				m_tree->AppendItem(currid,
					wxString::FromAscii(CL).Truncate(4),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
			};
			
		};
		break;



	/* JP2 Header box */
	case (IHDR_BOX): {

			unsigned long int height, width;
			unsigned short int nc;
			unsigned char bpc, C, UnkC, IPR;
			fileid->Read(&height, sizeof(unsigned long int));
			height = BYTE_SWAP4(height);
			fileid->Read(&width, sizeof(unsigned long int));
			width = BYTE_SWAP4(width);
			fileid->Read(&nc, sizeof(unsigned short int));
			nc = BYTE_SWAP2(nc);
			fileid->Read(&bpc, sizeof(unsigned char));
			fileid->Read(&C, sizeof(unsigned char));
			fileid->Read(&UnkC, sizeof(unsigned char));
			fileid->Read(&IPR, sizeof(unsigned char));
			
			// add info
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Dimensions: %d x %d x %d @ %d bpc"), width, height, nc, bpc + 1),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Compression type: %d"), C),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Colourspace unknown: %d"), UnkC),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Intellectual Property Rights: %d"), IPR),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
			
		};
		break;



	/* Colour Specification box */
	case (COLR_BOX): {

			unsigned char METH, PREC, APPROX;
			char methdescr[80], enumcsdescr[80];
			unsigned long int EnumCS;
			fileid->Read(&METH, sizeof(unsigned char));
			switch (METH) {
			case 1:
				strcpy(methdescr, "Enumerated Colourspace");
				break;
			case 2:
				strcpy(methdescr, "Restricted ICC profile");
				break;
			default:
				strcpy(methdescr, "Unknown");
				break;
			};
			fileid->Read(&PREC, sizeof(unsigned char));
			fileid->Read(&APPROX, sizeof(unsigned char));
			if (METH != 2) {
				fileid->Read(&EnumCS, sizeof(unsigned long int));
				EnumCS = BYTE_SWAP4(EnumCS);
				switch (EnumCS) {
				case 16:
					strcpy(enumcsdescr, "sRGB");
					break;
				case 17:
					strcpy(enumcsdescr, "greyscale");
					break;
				case 18:
					strcpy(enumcsdescr, "sYCC");
					break;
				default:
					strcpy(enumcsdescr, "Unknown");
					break;
				};
			};

			// add info
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Specification method: %d ("), METH) +
				wxString::FromAscii(methdescr) +
				wxT(")"),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Precedence: %d"), PREC),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Colourspace approximation: %d"), APPROX),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);

			if (METH != 2)
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Enumerated colourspace: %d ("), EnumCS) +
					wxString::FromAscii(enumcsdescr) +
					wxT(")"),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);

			if (METH != 1)
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("ICC profile: there is one")),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);


		};
		break;




		

	/* Movie Header Box */
	case (MVHD_BOX): {

			unsigned long int version, rate, matrix[9], next_track_ID;
			unsigned short int volume;
			fileid->Read(&version, sizeof(unsigned long int));
			version = BYTE_SWAP4(version);
			if (version == 0) {
				unsigned long int creation_time, modification_time, timescale, duration;
				fileid->Read(&creation_time, sizeof(unsigned long int));
				creation_time = BYTE_SWAP4(creation_time);
				fileid->Read(&modification_time, sizeof(unsigned long int));
				modification_time = BYTE_SWAP4(modification_time);
				fileid->Read(&timescale, sizeof(unsigned long int));
				timescale = BYTE_SWAP4(timescale);
				fileid->Read(&duration, sizeof(unsigned long int));
				duration = BYTE_SWAP4(duration);
				const long unix_time = creation_time - 2082844800L;
				wxTreeItemId currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Creation time: %u (%.24s)"), creation_time, ctime(&unix_time)),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				const long unix_time1 = modification_time - 2082844800L;
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Modification time: %u (%.24s)"), modification_time, ctime(&unix_time1)),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Timescale: %u (%.6fs)"), timescale, 1.0 / (float) timescale),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Duration: %u (%.3fs)"), duration, (float) duration / (float) timescale),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
			} else {
				int8byte creation_time, modification_time, duration;
				unsigned long int timescale;
				fileid->Read(&creation_time, sizeof(int8byte));
				creation_time = BYTE_SWAP8(creation_time);
				fileid->Read(&modification_time, sizeof(int8byte));
				modification_time = BYTE_SWAP8(modification_time);
				fileid->Read(&timescale, sizeof(unsigned long int));
				timescale = BYTE_SWAP4(timescale);
				fileid->Read(&duration, sizeof(int8byte));
				duration = BYTE_SWAP8(duration);
				wxTreeItemId currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Creation time: %u"), creation_time),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Modification time: %u"), modification_time),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Timescale: %u"), timescale),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Duration: %u"), duration),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
			};
			fileid->Read(&rate, sizeof(unsigned long int));
			rate = BYTE_SWAP4(rate);
			fileid->Read(&volume, sizeof(unsigned short int));
			volume = BYTE_SWAP2(volume);
			fileid->Seek(6, wxFromCurrent);
			fileid->Read(&matrix, sizeof(unsigned char) * 9);
			fileid->Seek(4, wxFromCurrent);
			fileid->Read(&next_track_ID, sizeof(unsigned long int));
			next_track_ID = BYTE_SWAP4(next_track_ID);
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Rate: %d (%d.%d)"), rate, rate >> 16, rate & 0x0000FFFF),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Volume: %d (%d.%d)"), volume, volume >> 8, volume & 0x00FF),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Next track ID: %d"), next_track_ID),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
		};
		break;


			/* Sample Description box */
	case (STSD_BOX): {

			unsigned long int version, entry_count;
			fileid->Read(&version, sizeof(unsigned long int));
			version = BYTE_SWAP4(version);
			fileid->Read(&entry_count, sizeof(unsigned long int));
			entry_count = BYTE_SWAP4(entry_count);
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Entry count: %d"), entry_count),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
				);
			jpeg2000parse(fileid, filepoint + 8, filelimit, parentid, level + 1, scansign, scanpoint);
		};
		break;


			/* Sample Size box */
	case (STSZ_BOX): {

			unsigned long int version, sample_size, sample_count, entry_size;
			
			fileid->Read(&version, sizeof(unsigned long int));
			version = BYTE_SWAP4(version);
			
			fileid->Read(&sample_size, sizeof(unsigned long int));
			sample_size = BYTE_SWAP4(sample_size);

			if (sample_size == 0) {
				fileid->Read(&sample_count, sizeof(unsigned long int));
				sample_count = BYTE_SWAP4(sample_count);

				wxTreeItemId currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Sample count: %d"), sample_count),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
					);

				currid = m_tree->AppendItem(parentid,
					wxT("Entries size (bytes)"),
					m_tree->TreeCtrlIcon_Folder, m_tree->TreeCtrlIcon_Folder + 1,
					new OPJMarkerData(wxT("INFO"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
					);

				wxString text;
				for (unsigned int s = 0; s < sample_count; s++) {
					fileid->Read(&entry_size, sizeof(unsigned long int));
					entry_size = BYTE_SWAP4(entry_size);
					
					text << wxString::Format(wxT("%d, "), entry_size);

					if (((s % 10) == (ITEM_PER_ROW - 1)) || (s == (sample_count - 1))) {
						m_tree->AppendItem(currid,
							text,
							m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
							new OPJMarkerData(wxT("INFO"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
							);
						text = wxT("");
					}

				}
				
			}

		};
		break;


			/* Video Media Header box */
	case (VMHD_BOX): {

			unsigned long int version;
			unsigned short int graphicsmode, opcolor[3];
			char graphicsdescr[100];

			fileid->Read(&version, sizeof(unsigned long int));
			version = BYTE_SWAP4(version);

			fileid->Read(&graphicsmode, sizeof(unsigned short int));
			graphicsmode = BYTE_SWAP2(graphicsmode);
			switch (graphicsmode) {
			case (0x00):
					strcpy(graphicsdescr, "copy");
					break;
			case (0x24):
					strcpy(graphicsdescr, "transparent");
					break;
			case (0x0100):
					strcpy(graphicsdescr, "alpha");
					break;
			case (0x0101):
					strcpy(graphicsdescr, "whitealpha");
					break;
			case (0x0102):
					strcpy(graphicsdescr, "blackalpha");
					break;
			default:
					strcpy(graphicsdescr, "unknown");
					break;
			};

			fileid->Read(opcolor, 3 * sizeof(unsigned short int));
			opcolor[0] = BYTE_SWAP2(opcolor[0]);
			opcolor[1] = BYTE_SWAP2(opcolor[1]);
			opcolor[2] = BYTE_SWAP2(opcolor[2]);

			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Composition mode: %d (")) + 
				wxString::FromAscii(graphicsdescr) +
				wxT(")"),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
				);

			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("OP color: %d %d %d"), opcolor[0], opcolor[1], opcolor[2]),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
				);
		};
		break;



			/* MJP2 Sample Description box */
	case (MJP2_BOX): {

			unsigned short int height, width, depth;
			unsigned long int horizresolution, vertresolution;
			char compressor_name[32];
			fileid->Seek(24, wxFromCurrent);
			fileid->Read(&width, sizeof(unsigned short int));
			width = BYTE_SWAP2(width);
			fileid->Read(&height, sizeof(unsigned short int));
			height = BYTE_SWAP2(height);
			fileid->Read(&horizresolution, sizeof(unsigned long int));
			horizresolution = BYTE_SWAP4(horizresolution);
			fileid->Read(&vertresolution, sizeof(unsigned long int));
			vertresolution = BYTE_SWAP4(vertresolution);
			fileid->Seek(6, wxFromCurrent);
			fileid->Read(compressor_name, sizeof(char) * 32);
			fileid->Read(&depth, sizeof(unsigned short int));
			depth = BYTE_SWAP2(depth);
			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Dimensions: %d x %d @ %d bpp"), width, height, depth),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"), m_tree->m_fname.GetFullPath(), filepoint, filelimit)
				);
			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Resolution: %d.%d x %d.%d"), horizresolution >> 16, horizresolution & 0x0000FFFF,
				vertresolution >> 16, vertresolution & 0x0000FFFF),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Compressor: %.32s"), compressor_name),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
			jpeg2000parse(fileid, filepoint + 78, filelimit, parentid, level + 1, scansign, scanpoint);

		};
		break;

		/* Media Header box */
	case (MDHD_BOX): {
			unsigned long int version;
			unsigned short int language;
			fileid->Read(&version, sizeof(unsigned long int));
			version = BYTE_SWAP4(version);
			if (version == 0) {
				unsigned long int creation_time, modification_time, timescale, duration;
				fileid->Read(&creation_time, sizeof(unsigned long int));
				creation_time = BYTE_SWAP4(creation_time);
				fileid->Read(&modification_time, sizeof(unsigned long int));
				modification_time = BYTE_SWAP4(modification_time);
				fileid->Read(&timescale, sizeof(unsigned long int));
				timescale = BYTE_SWAP4(timescale);
				fileid->Read(&duration, sizeof(unsigned long int));
				duration = BYTE_SWAP4(duration);
				const long unix_time = creation_time - 2082844800L;
				wxTreeItemId currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Creation time: %u (%.24s)"), creation_time, ctime(&unix_time)),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				const long unix_time1 = modification_time - 2082844800L;
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Modification time: %u (%.24s)"), modification_time, ctime(&unix_time1)),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Timescale: %u (%.6fs)"), timescale, 1.0 / (float) timescale),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Duration: %u (%.3fs)"), duration, (float) duration / (float) timescale),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
			} else {
				int8byte creation_time, modification_time, duration;
				unsigned long int timescale;
				fileid->Read(&creation_time, sizeof(int8byte));
				creation_time = BYTE_SWAP8(creation_time);
				fileid->Read(&modification_time, sizeof(int8byte));
				modification_time = BYTE_SWAP8(modification_time);
				fileid->Read(&timescale, sizeof(unsigned long int));
				timescale = BYTE_SWAP4(timescale);
				fileid->Read(&duration, sizeof(int8byte));
				duration = BYTE_SWAP8(duration);
				wxTreeItemId currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Creation time: %u"), creation_time),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Modification time: %u"), modification_time),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Timescale: %u"), timescale),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
				currid = m_tree->AppendItem(parentid,
					wxString::Format(wxT("Duration: %u"), duration),
					m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
					new OPJMarkerData(wxT("INFO"))
					);
			}
			fileid->Read(&language, sizeof(unsigned short int));

			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Language: %d (%c%c%c)"), language & 0xEFFF,
				0x60 + (char) ((language >> 10) & 0x001F), 0x60 + (char) ((language >> 5) & 0x001F), 0x60 + (char) ((language >> 0) & 0x001F)),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
		};
		break;
		
		/* Media Handler box */
	case (HDLR_BOX): {
			unsigned long int version, predefined, temp[3];
			char handler[4], name[256];
			int namelen = wxMin(256, (filelimit - filepoint - 24));
			fileid->Read(&version, sizeof(unsigned long int));
			version = BYTE_SWAP4(version);
			fileid->Read(&predefined, sizeof(unsigned long int));
			fileid->Read(handler, 4 * sizeof(char));
			fileid->Read(&temp, 3 * sizeof(unsigned long int));
			fileid->Read(name, namelen * sizeof(char));

			wxTreeItemId currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Handler: %.4s"), handler),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
					 
			currid = m_tree->AppendItem(parentid,
				wxString::Format(wxT("Name: %.255s"), name),
				m_tree->TreeCtrlIcon_File, m_tree->TreeCtrlIcon_File + 1,
				new OPJMarkerData(wxT("INFO"))
				);
					 				 
		}
		break;

	/* not yet implemented */
	default:
		break;

	};

	return (0);
}


void OPJParseThread::ParseJP2File(wxFile *fileid, wxFileOffset filepoint, wxFileOffset filelimit, wxTreeItemId parentid)
{
	unsigned long int scanpoint;

	jpeg2000parse(fileid, filepoint, filelimit, parentid, 0, NULL, &scanpoint);
}

/* the parsing function itself */
/*
  fileid    = fid of the file to scan (you should open it by yourself)
  filepoint = first byte where to start to scan from (usually 0)
  filelimit = first byte where to stop to scan from (usually the file size)
  level     = set this to 0
  scansign  = signature to scan for (NULL avoids search, returns "    " if successful)
  scanpoint = point where the scan signature lies
*/
int OPJParseThread::jpeg2000parse(wxFile *fileid, wxFileOffset filepoint, wxFileOffset filelimit,
								  wxTreeItemId parentid, int level, char *scansign, unsigned long int *scanpoint)
{
	unsigned long int       LBox = 0x00000000;
	//int                     LBox_read;
	char                    TBox[5] = "\0\0\0\0";
	//int                     TBox_read;
	int8byte				XLBox = 0x0000000000000000;
	//int                     XLBox_read;
	unsigned long int       box_length = 0;
	int                     last_box = 0, box_num = 0;
	int                     box_type = ANY_BOX;
	unsigned char           /*onebyte[1], twobytes[2],*/ fourbytes[4];

	/* cycle all over the file */
	box_num = 0;
	last_box = 0;
	while (!last_box) {

		/* do not exceed file limit */
		if (filepoint >= filelimit)
			return (0);

		/* seek on file */
		if (fileid->Seek(filepoint, wxFromStart) == wxInvalidOffset)
			return (-1);

		/* read the mandatory LBox, 4 bytes */
		if (fileid->Read(fourbytes, 4) < 4) {
			WriteText(wxT("Problem reading LBox from the file (file ended?)"));
			return -1;
		};
		LBox = STREAM_TO_UINT32(fourbytes, 0);

		/* read the mandatory TBox, 4 bytes */
		if (fileid->Read(TBox, 4) < 4) {
			WriteText(wxT("Problem reading TBox from the file (file ended?)"));
			return -1;
		};

		/* look if scansign is got */
		if ((scansign != NULL) && (memcmp(TBox, scansign, 4) == 0)) {
			memcpy(scansign, "    ", 4);
			*scanpoint = filepoint;

			/* hack/exploit */
			// stop as soon as you find the codebox
			return (0);

		};

		/* determine the box type */
		for (box_type = JP_BOX; box_type < UNK_BOX; box_type++)
			if (memcmp(TBox, j22box[box_type].value, 4) == 0)
				break;	

		/* read the optional XLBox, 8 bytes */
		if (LBox == 1) {

			if (fileid->Read(&XLBox, 8) < 8) {
				WriteText(wxT("Problem reading XLBox from the file (file ended?)"));
				return -1;
			};
			box_length = (unsigned long int) BYTE_SWAP8(XLBox);

		} else if (LBox == 0x00000000) {

			/* last box in file */
			last_box = 1; 
			box_length = filelimit - filepoint;

		} else

			box_length = LBox;

		/* show box info */

		// append the marker
		int image, imageSel;
		image = m_tree->TreeCtrlIcon_Folder;
		imageSel = image + 1;
		wxTreeItemId currid = m_tree->AppendItem(parentid,
			wxString::Format(wxT("%03d: "), box_num) +
			wxString::FromAscii(TBox) +
			wxString::Format(wxT(" (0x%04X)"),
				((unsigned long int) TBox[3]) + ((unsigned long int) TBox[2] << 8) +
				((unsigned long int) TBox[1] << 16) + ((unsigned long int) TBox[0] << 24)
			),
			image, imageSel,
			new OPJMarkerData(wxT("BOX"), m_tree->m_fname.GetFullPath(), filepoint, filepoint + box_length)
			);

		// append some info
		image = m_tree->TreeCtrlIcon_File;
		imageSel = image + 1;

		// box name
		wxTreeItemId subcurrid1 = m_tree->AppendItem(currid,
			wxT("*** ") + wxString::FromAscii(j22box[box_type].name) + wxT(" ***"),
			image, imageSel,
			new OPJMarkerData(wxT("INFO"))
			);
		m_tree->SetItemFont(subcurrid1, *wxITALIC_FONT);

		// position and length
		wxTreeItemId subcurrid2 = m_tree->AppendItem(currid,
			wxLongLong(filepoint).ToString() + wxT(" > ") + wxLongLong(filepoint + box_length - 1).ToString() + 
			wxT(", ") + wxString::Format(wxT("%d + 8 (%d)"), box_length, box_length + 8),
			image, imageSel,
			new OPJMarkerData(wxT("INFO"))
			);

		/* go deep in the box */
		box_handler_function((int) box_type, fileid, (LBox == 1) ? (filepoint + 16) : (filepoint + 8), filepoint + box_length,
			currid, level, scansign, scanpoint);

		/* if it's a superbox go inside it */
		if (j22box[box_type].sbox)
			jpeg2000parse(fileid, (LBox == 1) ? (filepoint + 16) : (filepoint + 8), filepoint + box_length,
				currid, level + 1, scansign, scanpoint);

		/* increment box number and filepoint*/
		box_num++;
		filepoint += box_length;

	};

	/* all good */
	return (0);
}

