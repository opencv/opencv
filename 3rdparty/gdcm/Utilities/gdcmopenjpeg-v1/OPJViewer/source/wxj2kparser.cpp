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

/* From little endian to big endian, 2 bytes */
#define	BYTE_SWAP2(X)	((X & 0x00FF) << 8) | ((X & 0xFF00) >> 8)
#define	BYTE_SWAP4(X)	((X & 0x000000FF) << 24) | ((X & 0x0000FF00) << 8) | ((X & 0x00FF0000) >> 8) | ((X & 0xFF000000) >> 24)

/* From codestream to int values */
#define STREAM_TO_UINT32(C, P)	(((unsigned long int) (C)[(P) + 0] << 24) + \
								((unsigned long int) (C)[(P) + 1] << 16) + \
								((unsigned long int) (C)[(P) + 2] << 8) + \
								((unsigned long int) (C)[(P) + 3] << 0))

#define STREAM_TO_UINT16(C, P)	(((unsigned long int) (C)[(P) + 0] << 8) + \
								((unsigned long int) (C)[(P) + 1] << 0))


/* Markers values */
#define J2KMARK_NUM 24
enum {
	SOC_VAL = 0xFF4F,
	SOT_VAL	= 0xFF90,
	SOD_VAL = 0xFF93,
	EOC_VAL	= 0xFFD9,
	SIZ_VAL	= 0xFF51,
	COD_VAL	= 0xFF52,
	COC_VAL = 0xFF53,
	RGN_VAL = 0xFF5E,
	QCD_VAL	= 0xFF5C,
	QCC_VAL	= 0xFF5D,
	POD_VAL	= 0xFF5F,
	TLM_VAL	= 0xFF55,
	PLM_VAL	= 0xFF57,
	PLT_VAL	= 0xFF58,
	PPM_VAL	= 0xFF60,
	PPT_VAL	= 0xFF61,
	SOP_VAL	= 0xFF91,
	EPH_VAL	= 0xFF92,
	COM_VAL	= 0xFF64
#ifdef USE_JPWL
	, EPB_VAL	= 0xFF66,
	ESD_VAL	= 0xFF67,
	EPC_VAL	= 0xFF68,
	RED_VAL	= 0xFF69
	/*, EPB_VAL = 0xFF96,
	ESD_VAL	= 0xFF98,
	EPC_VAL	= 0xFF97,
	RED_VAL	= 0xFF99*/
#endif // USE_JPWL
#ifdef USE_JPSEC
	, SEC_VAL = 0xFF65
#endif // USE_JPSEC
};

// All the markers in one vector
unsigned short int marker_val[] = {
	SOC_VAL, SOT_VAL, SOD_VAL, EOC_VAL,
	SIZ_VAL,
	COD_VAL, COC_VAL, RGN_VAL, QCD_VAL, QCC_VAL, POD_VAL,
	TLM_VAL, PLM_VAL, PLT_VAL, PPM_VAL, PPT_VAL,
	SOP_VAL, EPH_VAL,
	COM_VAL
#ifdef USE_JPWL
	, EPB_VAL, ESD_VAL, EPC_VAL, RED_VAL
#endif // USE_JPWL
#ifdef USE_JPSEC
	, SEC_VAL
#endif // USE_JPSEC
};

// Marker names
char *marker_name[] = {
	"SOC", "SOT", "SOD", "EOC",
	"SIZ",
	"COD", "COC", "RGN", "QCD", "QCC", "POD",
	"TLM", "PLM", "PLT", "PPM", "PPT",
	"SOP", "EPH",
	"COM"
#ifdef USE_JPWL
	, "EPB", "ESD", "EPC", "RED"
#endif // USE_JPWL
#ifdef USE_JPSEC
	, "SEC"
#endif // USE_JPSEC
};

// Marker descriptions
char *marker_descr[] = {
	"Start of codestream", "Start of tile-part", "Start of data", "End of codestream",
	"Image and tile size",
	"Coding style default", "Coding style component", "Region-of-interest", "Quantization default",
	"Quantization component", "Progression order change, default",
	"Tile-part lengths, main header", "Packet length, main header", "Packets length, tile-part header",
	"Packed packet headers, main header", "Packed packet headers, tile-part header",
	"Start of packet", "End of packet header",
	"Comment and extension"
#ifdef USE_JPWL
	, "Error Protection Block", "Error Sensitivity Descriptor", "Error Protection Capability",
	"Residual Errors Descriptor"
#endif // USE_JPWL
#ifdef USE_JPSEC
	, "Main security marker"
#endif // USE_JPSEC
};

void OPJParseThread::ParseJ2KFile(wxFile *m_file, wxFileOffset offset, wxFileOffset length, wxTreeItemId parentid)
{
	unsigned short int csiz = 0;

	// check if the file is opened
	if (m_file->IsOpened())
		WriteText(wxT("File OK"));
	else
		return;

	// position at the beginning
	m_file->Seek(offset, wxFromStart);

	// navigate the file
	int m, inside_sod = 0, inside_sop = 0;
	int nmarks = 0, maxmarks = 10000;
	unsigned char onebyte[1];
	unsigned char twobytes[2], firstbyte, secondbyte;
	unsigned char fourbytes[4];
	unsigned short int currmark;
	unsigned short int currlen;
	int lastPsot = 0, lastsotpos = 0;

	WriteText(wxT("Start search..."));

// advancing macro
#define OPJ_ADVANCE(A) {offset += A; if (offset < length) m_file->Seek(offset, wxFromStart); else return;}

	// begin search
	while ((offset < length) && (!m_file->Eof())) {

		// read one byte
		if (m_file->Read(&firstbyte, 1) != 1)
			break;

		// look for 0xFF
		if (firstbyte == 0xFF) {

			// it is a possible marker
			if (m_file->Read(&secondbyte, 1) != 1)
				break;
			else
				currmark = (((unsigned short int) firstbyte) << 8) + (unsigned short int) secondbyte;

		} else {

			// nope, advance by one and search again
			OPJ_ADVANCE(1);
			continue;
		}
		
		// search the marker
		for (m = 0; m < J2KMARK_NUM; m++) {
			if (currmark == marker_val[m])
				break;
		}

		// marker not found
		if (m == J2KMARK_NUM) {
			// nope, advance by one and search again
			OPJ_ADVANCE(1);
			continue;
		}

		// if we are inside SOD, only some markers are allowed
		if (inside_sod) {

			// we are inside SOP
			if (inside_sop) {

			}

			// randomly marker coincident data
			if ((currmark != SOT_VAL) &&
				(currmark != EOC_VAL) &&
				(currmark != SOP_VAL) &&
				(currmark != EPH_VAL)) {
				OPJ_ADVANCE(1);
				continue;
			}

			// possible SOT?
			if ((currmark == SOT_VAL)) {
				// too early SOT
				if (offset < (lastsotpos + lastPsot)) {
					OPJ_ADVANCE(1);
					continue;
				}
				// we were not in the last tile
				/*if (lastPsot != 0) {
					OPJ_ADVANCE(1);
					break;
				}*/
			}
		}

		// beyond this point, the marker MUST BE real

		// length of current marker segment
		if ((currmark == SOD_VAL) ||
			(currmark == SOC_VAL) ||
			(currmark == EOC_VAL) ||
			(currmark == EPH_VAL))

			// zero length markers
			currlen = 0;

		else {

			// read length field
			if (m_file->Read(twobytes, 2) != 2)
				break;

			currlen = (((unsigned short int) twobytes[0]) << 8) + (unsigned short int) twobytes[1];
		}

		// here we pass to AppendItem() normal and selected item images (we
		// suppose that selected image follows the normal one in the enum)
		int image, imageSel;
		image = m_tree->TreeCtrlIcon_Folder;
		imageSel = image + 1;

		// append the marker
		wxTreeItemId currid = m_tree->AppendItem(parentid,
			wxString::Format(wxT("%03d: "), nmarks) +
			wxString::FromAscii(marker_name[m]) + 
			wxString::Format(wxT(" (0x%04X)"), marker_val[m]),
			image, imageSel,
			new OPJMarkerData(wxT("MARK") + wxString::Format(wxT(" (%d)"), marker_val[m]),
				m_tree->m_fname.GetFullPath(), offset, offset + currlen + 1)
			);

		// append some info
		image = m_tree->TreeCtrlIcon_File;
		imageSel = image + 1;

		// marker name
		wxTreeItemId subcurrid1 = m_tree->AppendItem(currid,
			wxT("*** ") + wxString::FromAscii(marker_descr[m]) + wxT(" ***"),
			image, imageSel,
			new OPJMarkerData(wxT("INFO"))
			);
		m_tree->SetItemFont(subcurrid1, *wxITALIC_FONT);

		// position and length
		wxTreeItemId subcurrid2 = m_tree->AppendItem(currid,
			wxLongLong(offset).ToString() + wxT(" > ") + wxLongLong(offset + currlen + 1).ToString() + 
			wxT(", ") + wxString::Format(wxT("%d + 2 (%d)"), currlen, currlen + 2),
			image, imageSel,
			new OPJMarkerData(wxT("INFO"))
			);

		// give additional info on markers
		switch (currmark) {

		/////////
		// SOP //
		/////////
		case SOP_VAL:
			{
			// read packet number
			if (m_file->Read(twobytes, 2) != 2)
				break;
			int packnum = STREAM_TO_UINT16(twobytes, 0);

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				wxString::Format(wxT("Pack. no. %d"), packnum),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);
			inside_sop = 1;
			};
			break;

#ifdef USE_JPWL
		/////////
		// RED //
		/////////
		case RED_VAL:
			{
			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char pred = onebyte[0];

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			wxString address[] = {
				wxT("Packet addressing"),
				wxT("Byte-range addressing"),
				wxT("Packet-range addressing"),
				wxT("Reserved")
			};

			wxTreeItemId subcurrid = m_tree->AppendItem(currid,
				address[(pred & 0xC0) >> 6],
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("%d bytes range"), (((pred & 0x02) >> 1) + 1) * 2),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				pred & 0x01 ? wxT("Errors/erasures in codestream") : wxT("Error free codestream"),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("Residual corruption level: %d"), (pred & 0x38) >> 3),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			}
			break;

		/////////
		// ESD //
		/////////
		case ESD_VAL:
			{
			unsigned short int cesd;
			if (csiz < 257) {
				if (m_file->Read(onebyte, 1) != 1)
					break;
				cesd = onebyte[0];
			} else {
				if (m_file->Read(twobytes, 2) != 2)
					break;
				cesd = STREAM_TO_UINT16(twobytes, 0);
			}

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char pesd = onebyte[0];

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			wxTreeItemId subcurrid = m_tree->AppendItem(currid,
				pesd & 0x01 ? wxT("Comp. average") : wxString::Format(wxT("Comp. no. %d"), cesd),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			wxString meth[] = {
				wxT("Relative error sensitivity"),
				wxT("MSE"),
				wxT("MSE reduction"),
				wxT("PSNR"),
				wxT("PSNR increase"),
				wxT("MAXERR (absolute peak error)"),
				wxT("TSE (total squared error)"),
				wxT("Reserved")
			};

			subcurrid = m_tree->AppendItem(currid,
				meth[(pesd & 0x38) >> 3],
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			wxString address[] = {
				wxT("Packet addressing"),
				wxT("Byte-range addressing"),
				wxT("Packet-range addressing"),
				wxT("Reserved")
			};

			subcurrid = m_tree->AppendItem(currid,
				address[(pesd & 0xC0) >> 6],
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("%d bytes/value, %d bytes range"), ((pesd & 0x04) >> 2) + 1, (((pesd & 0x02) >> 1) + 1) * 2),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			}
			break;

		/////////
		// EPC //
		/////////
		case EPC_VAL:
			{
			if (m_file->Read(twobytes, 2) != 2)
				break;
			unsigned short int pcrc = STREAM_TO_UINT16(twobytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int dl = STREAM_TO_UINT32(fourbytes, 0);

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char pepc = onebyte[0];

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			wxTreeItemId subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("CRC-16 = 0x%x"), pcrc),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("Tot. length = %d"), dl),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("%s%s%s%s"),
					pepc & 0x10 ? wxT("ESD, ") : wxT(""),
					pepc & 0x20 ? wxT("RED, ") : wxT(""),
					pepc & 0x40 ? wxT("EPB, ") : wxT(""),
					pepc & 0x80 ? wxT("Info") : wxT("")
					),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			}
			break;

		/////////
		// EPB //
		/////////
		case EPB_VAL:
			{
			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char depb = onebyte[0];

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int ldpepb = STREAM_TO_UINT32(fourbytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int pepb = STREAM_TO_UINT32(fourbytes, 0);

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			wxTreeItemId subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("No. %d, %slatest, %spacked"),
					depb & 0x3F,
					depb & 0x40 ? wxT("") : wxT("not "),
					depb & 0x80 ? wxT("") : wxT("un")),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("%d bytes protected"), ldpepb),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (pepb == 0x00000000)

				subcurrid = m_tree->AppendItem(currid,
					wxT("Predefined codes"),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			else if ((pepb >= 0x10000000) && (pepb <= 0x1FFFFFFF)) {

				wxString text = wxT("CRC code");
				if (pepb == 0x10000000)
					text << wxT(", CCITT (X25) 16 bits");
				else if (pepb == 0x10000001)
					text << wxT(", Ethernet 32 bits");
				else
					text << wxT(", JPWL RA");
				subcurrid = m_tree->AppendItem(currid,
					text,
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			} else if ((pepb >= 0x20000000) && (pepb <= 0x2FFFFFFF)) {

				wxString text;
				subcurrid = m_tree->AppendItem(currid,
					wxString::Format(wxT("RS code, RS(%d, %d)"),
						(pepb & 0x0000FF00) >> 8,
						(pepb & 0x000000FF)),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			} else if ((pepb >= 0x30000000) && (pepb <= 0x3FFFFFFE))

				subcurrid = m_tree->AppendItem(currid,
					wxT("JPWL RA"),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			else if (pepb == 0xFFFFFFFF)

				subcurrid = m_tree->AppendItem(currid,
					wxT("No method"),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			}
			break;
#endif // USE_JPWL

#ifdef USE_JPSEC
		case SEC_VAL:
			{

			}
			break;
#endif // USE_JPSEC

		/////////
		// SIZ //
		/////////
		case SIZ_VAL:
			{
			int c;
			
			if (m_file->Read(twobytes, 2) != 2)
				break;
			unsigned short int rsiz = STREAM_TO_UINT16(twobytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int xsiz = STREAM_TO_UINT32(fourbytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int ysiz = STREAM_TO_UINT32(fourbytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int xosiz = STREAM_TO_UINT32(fourbytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int yosiz = STREAM_TO_UINT32(fourbytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int xtsiz = STREAM_TO_UINT32(fourbytes, 0);
			this->m_tree->m_childframe->m_twidth = xtsiz;

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int ytsiz = STREAM_TO_UINT32(fourbytes, 0);
			this->m_tree->m_childframe->m_theight = ytsiz;

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int xtosiz = STREAM_TO_UINT32(fourbytes, 0);
			this->m_tree->m_childframe->m_tx = xtosiz;

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int ytosiz = STREAM_TO_UINT32(fourbytes, 0);
			this->m_tree->m_childframe->m_ty = ytosiz;

			if (m_file->Read(twobytes, 2) != 2)
				break;
			csiz = STREAM_TO_UINT16(twobytes, 0);

			bool equaldepth = true, equalsize = true;
			unsigned char *ssiz  = new unsigned char(csiz);
			unsigned char *xrsiz = new unsigned char(csiz);
			unsigned char *yrsiz = new unsigned char(csiz);

			for (c = 0; c < csiz; c++) {

				if (m_file->Read(&ssiz[c], 1) != 1)
					break;

				if (c > 0)
					equaldepth = equaldepth && (ssiz[c] == ssiz[c - 1]);

				if (m_file->Read(&xrsiz[c], 1) != 1)
					break;

				if (m_file->Read(&yrsiz[c], 1) != 1)
					break;

				if (c > 0)
					equalsize = equalsize && (xrsiz[c] == xrsiz[c - 1]) && (yrsiz[c] == yrsiz[c - 1]) ;

			}

			if (equaldepth && equalsize)
				wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
					wxString::Format(wxT("I: %dx%d (%d, %d), %d c., %d%s bpp"),
					xsiz, ysiz,
					xosiz, yosiz,
					csiz, ((ssiz[0] & 0x7F) + 1),
					(ssiz[0] & 0x80) ? wxT("s") : wxT("u")),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);
			else
				wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
					wxString::Format(wxT("I: %dx%d (%d, %d), %d c."),
					xsiz, ysiz,
					xosiz, yosiz,
					csiz),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				wxString::Format(wxT("T: %dx%d (%d, %d)"),
				xtsiz, ytsiz,
				xtosiz, ytosiz),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			image = m_tree->TreeCtrlIcon_Folder;
			imageSel = image + 1;

			wxTreeItemId subcurrid4 = m_tree->AppendItem(currid,
				wxT("Components"),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			for (c = 0; c < csiz; c++) {

				wxTreeItemId subcurrid5 = m_tree->AppendItem(subcurrid4,
					wxString::Format(wxT("#%d: %dx%d, %d%s bpp"),
					c,
					xsiz/xrsiz[c], ysiz/yrsiz[c],
					((ssiz[c] & 0x7F) + 1),
					(ssiz[c] & 0x80) ? wxT("s") : wxT("u")),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			}

			};
			break;

		/////////
		// SOT //
		/////////
		case SOT_VAL:
			{
			if (m_file->Read(twobytes, 2) != 2)
				break;
			unsigned short int isot = STREAM_TO_UINT16(twobytes, 0);

			if (m_file->Read(fourbytes, 4) != 4)
				break;
			unsigned long int psot = STREAM_TO_UINT32(fourbytes, 0);

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char tpsot = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char tnsot = onebyte[0];

			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				wxString::Format(wxT("tile %d, psot = %d, part %d of %d"), isot, psot, tpsot, tnsot),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			lastPsot = psot;
			lastsotpos = offset;
			inside_sod = 0;
			};
			break;

		/////////
		// COC //
		/////////
		case COC_VAL:
			{
			unsigned short int ccoc;
			if (csiz < 257) {
				if (m_file->Read(onebyte, 1) != 1)
					break;
				ccoc = onebyte[0];
			} else {
				if (m_file->Read(twobytes, 2) != 2)
					break;
				ccoc = STREAM_TO_UINT16(twobytes, 0);
			}

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char scoc = onebyte[0];

			wxTreeItemId subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("Comp. no. %d"), ccoc),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);
			
			wxString text;
			if (scoc & 0x01)
				text << wxT("Partitioned entropy coder");
			else
				text << wxT("Unpartitioned entropy coder");

			subcurrid = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char decomplevs = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char cbswidth = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char cbsheight = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char cbstyle = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char transform = onebyte[0];

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("%d levels (%d resolutions)"), decomplevs, decomplevs + 1),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (transform & 0x01)
				text = wxT("5-3 reversible wavelet");
			else
				text = wxT("9-7 irreversible wavelet");
			subcurrid = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("Code-blocks: %dx%d"), 1 << ((cbswidth & 0x0F) + 2), 1 << ((cbsheight & 0x0F) + 2)),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			image = m_tree->TreeCtrlIcon_Folder;
			imageSel = image + 1;

			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				wxT("Coding styles"),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			if (cbstyle & 0x01)
				text = wxT("Selective arithmetic coding bypass");
			else
				text = wxT("No selective arithmetic coding bypass");
			wxTreeItemId subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x02)
				text = wxT("Reset context probabilities on coding pass boundaries");
			else
				text = wxT("No reset of context probabilities on coding pass boundaries");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x04)
				text = wxT("Termination on each coding passs");
			else
				text = wxT("No termination on each coding pass");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x08)
				text = wxT("Vertically stripe causal context");
			else
				text = wxT("No vertically stripe causal context");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x10)
				text = wxT("Predictable termination");
			else
				text = wxT("No predictable termination");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x20)
				text = wxT("Segmentation symbols are used");
			else
				text = wxT("No segmentation symbols are used");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			}
			break;

		/////////
		// COD //
		/////////
		case COD_VAL:
			{
			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char scod = onebyte[0];

			wxString text;

			if (scod & 0x01)
				text << wxT("Partitioned entropy coder");
			else
				text << wxT("Unpartitioned entropy coder");

			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			text = wxT("");
			if (scod & 0x02)
				text << wxT("Possible SOPs");
			else
				text << wxT("No SOPs");

			if (scod & 0x04)
				text << wxT(", possible EPHs");
			else
				text << wxT(", no EPHs");

			subcurrid3 = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char progord = onebyte[0];

			if (m_file->Read(twobytes, 2) != 2)
				break;
			unsigned short int numlayers = STREAM_TO_UINT16(twobytes, 0);

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char mctransform = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char decomplevs = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char cbswidth = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char cbsheight = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char cbstyle = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char transform = onebyte[0];

			subcurrid3 = m_tree->AppendItem(currid,
				wxString::Format(wxT("%d levels (%d resolutions)"), decomplevs, decomplevs + 1),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			text = wxT("");
			switch (progord) {
			case (0):
				text << wxT("LRCP");
				break;
			case (1):
				text << wxT("RLCP");
				break;
			case (2):
				text << wxT("LRCP");
				break;
			case (3):
				text << wxT("RPCL");
				break;
			case (4):
				text << wxT("CPRL");
				break;
			default:
				text << wxT("unknown progression");
				break;
			}
			text << wxString::Format(wxT(", %d layers"), numlayers);
			if (transform & 0x01)
				text << wxT(", 5-3 rev.");
			else
				text << wxT(", 9-7 irr.");
			subcurrid3 = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid3 = m_tree->AppendItem(currid,
				wxString::Format(wxT("Code-blocks: %dx%d"), 1 << ((cbswidth & 0x0F) + 2), 1 << ((cbsheight & 0x0F) + 2)),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			switch (mctransform) {
			case (0):
				{
				text = wxT("No MCT");
				}
				break;
			case (1):
				{
				text = wxT("Reversible MCT on 0, 1, 2");
				}
				break;
			case (2):
				{
				text = wxT("Irreversible MCT on 0, 1, 2");
				}
				break;
			default:
				{
				text = wxT("Unknown");
				}
				break;
			};
			subcurrid3 = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);


			image = m_tree->TreeCtrlIcon_Folder;
			imageSel = image + 1;

			subcurrid3 = m_tree->AppendItem(currid,
				wxT("Coding styles"),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			if (cbstyle & 0x01)
				text = wxT("Selective arithmetic coding bypass");
			else
				text = wxT("No selective arithmetic coding bypass");
			wxTreeItemId subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x02)
				text = wxT("Reset context probabilities on coding pass boundaries");
			else
				text = wxT("No reset of context probabilities on coding pass boundaries");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x04)
				text = wxT("Termination on each coding passs");
			else
				text = wxT("No termination on each coding pass");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x08)
				text = wxT("Vertically stripe causal context");
			else
				text = wxT("No vertically stripe causal context");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x10)
				text = wxT("Predictable termination");
			else
				text = wxT("No predictable termination");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (cbstyle & 0x20)
				text = wxT("Segmentation symbols are used");
			else
				text = wxT("No segmentation symbols are used");
			subcurrid4 = m_tree->AppendItem(subcurrid3,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			};
			break;

		/////////
		// QCC //
		/////////
		case QCC_VAL:
			{
			unsigned short int cqcc;
			if (csiz < 257) {
				if (m_file->Read(onebyte, 1) != 1)
					break;
				cqcc = onebyte[0];
			} else {
				if (m_file->Read(twobytes, 2) != 2)
					break;
				cqcc = STREAM_TO_UINT16(twobytes, 0);
			}

			wxTreeItemId subcurrid = m_tree->AppendItem(currid,
				wxString::Format(wxT("Comp. no. %d"), cqcc),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);
			
			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char sqcc = onebyte[0];

			wxString text;
			switch (sqcc & 0x1F) {
			case (0):
				text = wxT("No quantization");
				break;
			case (1):
				text = wxT("Scalar implicit");
				break;
			case (2):
				text = wxT("Scalar explicit");
				break;
			default:
				text = wxT("Unknown");
				break;
			}
			text << wxString::Format(wxT(", %d guard bits"), (sqcc & 0xE0) >> 5);
			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			}
			break;

		/////////
		// QCD //
		/////////
		case QCD_VAL:
			{
			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char sqcd = onebyte[0];

			wxString text;
			switch (sqcd & 0x1F) {
			case (0):
				text = wxT("No quantization");
				break;
			case (1):
				text = wxT("Scalar implicit");
				break;
			case (2):
				text = wxT("Scalar explicit");
				break;
			default:
				text = wxT("Unknown");
				break;
			}
			text << wxString::Format(wxT(", %d guard bits"), (sqcd & 0xE0) >> 5);
			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			};
			break;

		/////////
		// COM //
		/////////
		case COM_VAL:
			{
			#define showlen 25
			char comment[showlen];
			wxString comments;

			if (m_file->Read(twobytes, 2) != 2)
				break;
			unsigned short int rcom = STREAM_TO_UINT16(twobytes, 0);

			wxString text;
			if (rcom == 0)
				text = wxT("Binary values");
			else if (rcom == 1)
				text = wxT("ISO 8859-1 (latin-1) values");
			else if (rcom < 65535)
				text = wxT("Reserved for registration");
			else
				text = wxT("Reserved for extension");
			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				text,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			if (m_file->Read(comment, showlen) != showlen)
				break;
			comments = wxString::FromAscii(comment).Truncate(wxMin(showlen, currlen - 4));
			if ((currlen - 4) > showlen)
				comments << wxT("...");
			subcurrid3 = m_tree->AppendItem(currid,
				comments,
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);
			};
			break;

		/////////
		// TLM //
		/////////
		case TLM_VAL:
			{
			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char ztlm = onebyte[0];

			if (m_file->Read(onebyte, 1) != 1)
				break;
			unsigned char stlm = onebyte[0];

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
				wxString::Format(wxT("TLM #%d"), ztlm),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			subcurrid3 = m_tree->AppendItem(currid,
				wxString::Format(wxT("%d bits/index, %d bits/length"),
				8 * ((stlm & 0x30) >> 4), 16 + 16 * ((stlm & 0x40) >> 6)),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			int n, numparts;

			numparts = (currlen - 2) / ( ((stlm & 0x30) >> 4) + 2 + 2 * ((stlm & 0x40) >> 6));

			image = m_tree->TreeCtrlIcon_Folder;
			imageSel = image + 1;

			subcurrid3 = m_tree->AppendItem(currid,
				wxT("Tile parts"),
				image, imageSel,
				new OPJMarkerData(wxT("INFO"))
				);

			image = m_tree->TreeCtrlIcon_File;
			imageSel = image + 1;

			for (n = 0; n < numparts; n++) {

				unsigned short int ttlm;
				unsigned long int ptlm;

				switch (((stlm & 0x30) >> 4)) {

				case 0:
					ttlm = 0;
					break;

				case 1:
					if (m_file->Read(onebyte, 1) != 1)
						break;
					ttlm = onebyte[0];
					break;

				case 2:
					if (m_file->Read(twobytes, 2) != 2)
						break;
					ttlm = STREAM_TO_UINT16(twobytes, 0);
					break;

				}

				switch (((stlm & 0x40) >> 6)) {

				case 0:
					if (m_file->Read(twobytes, 2) != 2)
						break;
					ptlm = STREAM_TO_UINT16(twobytes, 0);
					break;

				case 1:
					if (m_file->Read(fourbytes, 4) != 4)
						break;
					ptlm = STREAM_TO_UINT32(fourbytes, 0);
					break;

				}

				wxTreeItemId subcurrid4 = m_tree->AppendItem(subcurrid3,
					wxString::Format(wxT("Tile %d: %d bytes"), ttlm, ptlm),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

			}

			}
			break;

		/////////
		// POD //
		/////////
		case POD_VAL:
			{
			int n, numchanges;

			if (csiz < 257)
				numchanges = (currlen - 2) / 7;
			else
				numchanges = (currlen - 2) / 9;

			for (n = 0; n < numchanges; n++) {

				image = m_tree->TreeCtrlIcon_Folder;
				imageSel = image + 1;

				wxTreeItemId subcurrid3 = m_tree->AppendItem(currid,
					wxString::Format(wxT("Change #%d"), n),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

				if (m_file->Read(onebyte, 1) != 1)
					break;
				unsigned char rspod = onebyte[0];

				unsigned short int cspod;
				if (csiz < 257) {
					if (m_file->Read(onebyte, 1) != 1)
						break;
					cspod = onebyte[0];
				} else {
					if (m_file->Read(twobytes, 2) != 2)
						break;
					cspod = STREAM_TO_UINT16(twobytes, 0);
				}

				if (m_file->Read(twobytes, 2) != 2)
					break;
				unsigned short int lyepod = STREAM_TO_UINT16(twobytes, 0);

				if (m_file->Read(onebyte, 1) != 1)
					break;
				unsigned char repod = onebyte[0];

				unsigned short int cepod;
				if (csiz < 257) {
					if (m_file->Read(onebyte, 1) != 1)
						break;
					cepod = onebyte[0];
				} else {
					if (m_file->Read(twobytes, 2) != 2)
						break;
					cepod = STREAM_TO_UINT16(twobytes, 0);
				}

				if (m_file->Read(onebyte, 1) != 1)
					break;
				unsigned char ppod = onebyte[0];

				image = m_tree->TreeCtrlIcon_File;
				imageSel = image + 1;

				wxTreeItemId subcurrid4 = m_tree->AppendItem(subcurrid3,
					wxString::Format(wxT("%d <= Resolution < %d"), rspod, repod),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

				subcurrid4 = m_tree->AppendItem(subcurrid3,
					wxString::Format(wxT("%d <= Component < %d"), cspod, cepod),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

				subcurrid4 = m_tree->AppendItem(subcurrid3,
					wxString::Format(wxT("0 <= Layer < %d"), lyepod),
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);

				wxString text = wxT("");
				switch (ppod) {
				case (0):
					text << wxT("LRCP");
					break;
				case (1):
					text << wxT("RLCP");
					break;
				case (2):
					text << wxT("LRCP");
					break;
				case (3):
					text << wxT("RPCL");
					break;
				case (4):
					text << wxT("CPRL");
					break;
				default:
					text << wxT("unknown progression");
					break;
				}
				subcurrid4 = m_tree->AppendItem(subcurrid3,
					text,
					image, imageSel,
					new OPJMarkerData(wxT("INFO"))
					);
			}

			}
			break;

		/////////
		// SOD //
		/////////
		case SOD_VAL:
			{
			inside_sod = 1;
			};
			break;

		default:
			break;
			
		}
								
		// increment number of markers
		if (nmarks++ >= maxmarks) {
			WriteText(wxT("Maximum amount of markers exceeded"));
			break;
		}

		// advance position
		OPJ_ADVANCE(currlen + 2);
	}	

	WriteText(wxT("Search finished"));
}
