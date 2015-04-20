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
// Name:        imagmxf.cpp
// Purpose:     wxImage MXF (Material eXchange Format) JPEG 2000 file format handler
// Author:      Giuseppe Baruffa - based on imagjpeg.cpp, Vaclav Slavik
// RCS-ID:      $Id: imagmxf.cpp,v 0.00 2007/11/19 17:00:00 MW Exp $
// Copyright:   (c) Giuseppe Baruffa
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifdef USE_MXF

#include "mxflib/mxflib.h"
using namespace mxflib;

namespace
{
	//! Structure holding information about the essence in each body stream
	struct EssenceInfo
	{
		UMIDPtr PackageID;
		PackagePtr Package;
		MDObjectPtr Descriptor;
	};
	//! Map of EssenceInfo structures indexed by BodySID
	typedef std::map<UInt32, EssenceInfo> EssenceInfoMap;

	//! The map of essence info for this file
	EssenceInfoMap EssenceLookup;
};

//! Build an EssenceInfoMap for the essence in a given file
/*! \return True if al OK, else false
 */
bool BuildEssenceInfo(MXFFilePtr &File, EssenceInfoMap &EssenceLookup);

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
    #pragma hdrstop
#endif

#if wxUSE_IMAGE && wxUSE_LIBOPENJPEG

#include "imagmxf.h"

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
// wxMXFHandler
//-----------------------------------------------------------------------------

IMPLEMENT_DYNAMIC_CLASS(wxMXFHandler,wxImageHandler)

#if wxUSE_STREAMS

#include <stdarg.h>
#define MAX_MESSAGE_LEN 200

//------------- MXF Manager

// Debug and error messages

//! Display a warning message
void mxflib::warning(const char *Fmt, ...)
{
	char msg[MAX_MESSAGE_LEN];
	va_list args;

	va_start(args, Fmt);
	_vsnprintf(msg, MAX_MESSAGE_LEN, Fmt, args);
	va_end(args);

	int message_len = strlen(msg) - 1;
	if (msg[message_len] != '\n')
		message_len = MAX_MESSAGE_LEN;
#ifndef __WXGTK__ 
		wxMutexGuiEnter();
#endif /* __WXGTK__ */
	wxLogMessage(wxT("[WARNING_MXF] %.*s"), message_len, msg);
#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */
}

//! Display an error message
void mxflib::error(const char *Fmt, ...)
{
	char msg[MAX_MESSAGE_LEN];
	va_list args;

	va_start(args, Fmt);
	_vsnprintf(msg, MAX_MESSAGE_LEN, Fmt, args);
	va_end(args);

	int message_len = strlen(msg) - 1;
	if (msg[message_len] != '\n')
		message_len = MAX_MESSAGE_LEN;
#ifndef __WXGTK__ 
		wxMutexGuiEnter();
#endif /* __WXGTK__ */
	wxLogMessage(wxT("[ERROR_MXF] %.*s"), message_len, msg);
#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */
}

//! Display an error message
void mxflib::debug(const char *Fmt, ...)
{
	char msg[MAX_MESSAGE_LEN];
	va_list args;

	va_start(args, Fmt);
	_vsnprintf(msg, MAX_MESSAGE_LEN, Fmt, args);
	va_end(args);

	int message_len = strlen(msg) - 1;
	if (msg[message_len] != '\n')
		message_len = MAX_MESSAGE_LEN;
#ifndef __WXGTK__ 
		wxMutexGuiEnter();
#endif /* __WXGTK__ */
	wxLogMessage(wxT("[DEBUG_MXF] %.*s"), message_len, msg);
#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */
}



//------------- JPEG 2000 Data Source Manager

#define J2K_CFMT 0
#define JP2_CFMT 1
#define JPT_CFMT 2
#define MJ2_CFMT 3
#define PXM_DFMT 0
#define PGX_DFMT 1
#define BMP_DFMT 2
#define YUV_DFMT 3

/* sample error callback expecting a FILE* client object */
void mxf_error_callback(const char *msg, void *client_data) {
	int message_len = strlen(msg) - 1;
	if (msg[message_len] != '\n')
		message_len = MAX_MESSAGE_LEN;
#ifndef __WXGTK__ 
		wxMutexGuiEnter();
#endif /* __WXGTK__ */
	wxLogMessage(wxT("[ERROR] %.*s"), message_len, msg);
#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */
}

/* sample warning callback expecting a FILE* client object */
void mxf_warning_callback(const char *msg, void *client_data) {
	int message_len = strlen(msg) - 1;
	if (msg[message_len] != '\n')
		message_len = MAX_MESSAGE_LEN;
#ifndef __WXGTK__ 
		wxMutexGuiEnter();
#endif /* __WXGTK__ */
	wxLogMessage(wxT("[WARNING] %.*s"), message_len, msg);
#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */
}

/* sample debug callback expecting no client object */
void mxf_info_callback(const char *msg, void *client_data) {
	int message_len = strlen(msg) - 1;
	if (msg[message_len] != '\n')
		message_len = MAX_MESSAGE_LEN;
#ifndef __WXGTK__ 
		wxMutexGuiEnter();
#endif /* __WXGTK__ */
	wxLogMessage(wxT("[INFO] %.*s"), message_len, msg);
#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif /* __WXGTK__ */
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////

// load the mxf file format
bool wxMXFHandler::LoadFile(wxImage *image, wxInputStream& stream, bool verbose, int index)
{
	opj_dparameters_t parameters;	/* decompression parameters */
	opj_event_mgr_t event_mgr;		/* event manager */
	opj_image_t *opjimage = NULL;
	unsigned char *src = NULL;
    unsigned char *ptr;
	int file_length, j2k_point, j2k_len;
	opj_codestream_info_t cstr_info;  /* Codestream information structure */
	
	// simply display the version of the library
	wxLogMessage(wxT("Version of MXF: %s   "), wxString::FromAscii(LibraryVersion().c_str()));
	//wxLogMessage(wxT("MXF file name: %s"), m_filename.GetFullPath());

	// open MXF file
	MXFFilePtr TestFile = new MXFFile;
	if (! TestFile->Open(m_filename.GetFullPath().c_str(), true))
	{
		wxLogError(wxT("Could not find %s"), m_filename.GetFullPath().c_str());
		return false;
	} else
		wxLogMessage(wxT("Found %s"), m_filename.GetFullPath().c_str());

	// Get the size
	TestFile->SeekEnd();
	wxLogMessage(wxT("Size is %d bytes"), TestFile->Tell());
	TestFile->Seek(0);

	// essence information
	//BuildEssenceInfo(TestFile, EssenceLookup);

	// close MXF file
	TestFile->Close();

	return false;
	
	// destroy the image
    image->Destroy();

	/* handle to a decompressor */
	opj_dinfo_t* dinfo = NULL;	
	opj_cio_t *cio = NULL;

	/* configure the event callbacks (not required) */
	memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
	event_mgr.error_handler = mxf_error_callback;
	event_mgr.warning_handler = mxf_warning_callback;
	event_mgr.info_handler = mxf_info_callback;

	/* set decoding parameters to default values */
	opj_set_default_decoder_parameters(&parameters);

	/* prepare parameters */
	strncpy(parameters.infile, "", sizeof(parameters.infile)-1);
	strncpy(parameters.outfile, "", sizeof(parameters.outfile)-1);
	parameters.decod_format = J2K_CFMT;
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
	dinfo = opj_create_decompress(CODEC_J2K);

	/* find length of the stream */
	stream.SeekI(0, wxFromEnd);
	file_length = (int) stream.TellI();

	/* search for the m_framenum codestream position and length  */
	//jp2c_point = searchjp2c(stream, file_length, m_framenum);
	//jp2c_len = searchjp2c(stream, file_length, m_framenum);
	j2k_point = 0;
	j2k_len = 10;

	// malloc memory source
    src = (unsigned char *) malloc(j2k_len);

	// copy the jp2c
	stream.SeekI(j2k_point, wxFromStart);
	stream.Read(src, j2k_len);

	/* catch events using our callbacks and give a local context */
	opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);

	/* setup the decoder decoding parameters using user parameters */
	opj_setup_decoder(dinfo, &parameters);

	/* open a byte stream */
	cio = opj_cio_open((opj_common_ptr)dinfo, src, j2k_len);

	/* decode the stream and fill the image structure */
	opjimage = opj_decode_with_info(dinfo, cio, &cstr_info);
	if (!opjimage) {
		wxMutexGuiEnter();
		wxLogError(wxT("MXF: failed to decode image!"));
		wxMutexGuiLeave();
		opj_destroy_decompress(dinfo);
		opj_cio_close(cio);
		free(src);
		return false;
	}

	/* close the byte stream */
	opj_cio_close(cio);

	/* common rendering method */
#include "imagjpeg2000.cpp"

    wxMutexGuiEnter();
    wxLogMessage(wxT("MXF: image loaded."));
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

// save the mxf file format
bool wxMXFHandler::SaveFile( wxImage *image, wxOutputStream& stream, bool verbose )
{
    wxLogError(wxT("MXF: Couldn't save movie -> not implemented."));
    return false;
}

#ifdef __VISUALC__
    #pragma warning(default:4611)
#endif /* VC++ */

// recognize the MXF JPEG 2000 starting box
bool wxMXFHandler::DoCanRead( wxInputStream& stream )
{
    unsigned char hdr[4];

    if ( !stream.Read(hdr, WXSIZEOF(hdr)) )
        return false;

    return (hdr[0] == 0x06 &&
			hdr[1] == 0x0E &&
			hdr[2] == 0x2B &&
			hdr[3] == 0x34);
}

//! Build an EssenceInfoMap for the essence in a given file
/*! \return True if al OK, else false
 */
bool BuildEssenceInfo(MXFFilePtr &File, EssenceInfoMap &EssenceLookup)
{
	// Empty any old data
	EssenceLookup.clear();

	// Get the master metadata set (or the header if we must)
	PartitionPtr MasterPartition = File->ReadMasterPartition();
	if(!MasterPartition)
	{
		File->Seek(0);
		MasterPartition = File->ReadPartition();
		warning("File %s does not contain a cloased copy of header metadata - using the open copy in the file header\n", File->Name.c_str());
	}

	if(!MasterPartition) 
	{
		error("Could not read header metadata from file %s\n", File->Name.c_str());
		return false;
	}

	// Read and parse the metadata
	MasterPartition->ReadMetadata();
	MetadataPtr HMeta = MasterPartition->ParseMetadata();
	
	if(!HMeta) 
	{
		error("Could not read header metadata from file %s\n", File->Name.c_str());
		return false;
	}

	/* Scan the Essence container data sets to get PackageID to BodySID mapping */
	MDObjectPtr ECDSet = HMeta[ContentStorage_UL];
	if(ECDSet) ECDSet = ECDSet->GetLink();
	if(ECDSet) ECDSet = ECDSet[EssenceContainerDataBatch_UL];
	if(!ECDSet)
	{
		error("Header metadata in file %s does not contain an EssenceContainerData set\n", File->Name.c_str());
		return false;
	}

	MDObject::iterator it = ECDSet->begin();
	while(it != ECDSet->end())
	{
		MDObjectPtr ThisECDSet = (*it).second->GetLink();
		MDObjectPtr PackageID;
		if(ThisECDSet) PackageID = ThisECDSet->Child(LinkedPackageUID_UL);
		if(PackageID)
		{
			EssenceInfo NewEI;
			NewEI.PackageID = new UMID(PackageID->PutData()->Data);

			// Inset the basic essence info - but not if this is external essence (BodySID == 0)
			UInt32 BodySID = ThisECDSet->GetUInt(BodySID_UL);
			if(BodySID) EssenceLookup[BodySID] = NewEI;
		}
		it++;
	}

	/* Now find the other items for the essence lookup map */
	if(EssenceLookup.size())
	{
		PackageList::iterator it = HMeta->Packages.begin();
		while(it != HMeta->Packages.end())
		{
			// Only Source Packages are of interest
			if((*it)->IsA(SourcePackage_UL))
			{
				MDObjectPtr Descriptor = (*it)->Child(Descriptor_UL);
				if(Descriptor) Descriptor = Descriptor->GetLink();

				if(Descriptor)
				{
					MDObjectPtr PackageID = (*it)->Child(PackageUID_UL);
					if(PackageID)
					{
						UMIDPtr TheID = new UMID(PackageID->PutData()->Data);
						
						/* Now do a lookup in the essence lookup map (it will need to be done the long way here */
						EssenceInfoMap::iterator EL_it = EssenceLookup.begin();
						while(EL_it != EssenceLookup.end())
						{
							if((*((*EL_it).second.PackageID)) == (*TheID))
							{
								// If found, set the missing items and stop searching
								(*EL_it).second.Package = (*it);
								(*EL_it).second.Descriptor = Descriptor;
								break;
							}
							EL_it++;
						}
					}
				}
			}

			it++;
		}
	}

	return true;
}


#endif   // wxUSE_STREAMS

#endif   // wxUSE_LIBOPENJPEG

#endif // USE_MXF

