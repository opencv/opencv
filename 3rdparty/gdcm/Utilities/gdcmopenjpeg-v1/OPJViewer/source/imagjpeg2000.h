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
// Name:        imagalljpeg2000.h
// Purpose:     wxImage JPEG 2000 family file format handler
// Author:      G. Baruffa - based on imagjpeg.h, Vaclav Slavik
// RCS-ID:      $Id: imagalljpeg2000.h,v 0.0 2008/01/31 11:22:00 VZ Exp $
// Copyright:   (c) Giuseppe Baruffa
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_IMAGJPEG2000_H_
#define _WX_IMAGJPEG2000_H_

#include "wx/defs.h"

//-----------------------------------------------------------------------------
// wxJPEG2000Handler
//-----------------------------------------------------------------------------

#if wxUSE_LIBOPENJPEG

#include "wx/image.h"
#include "libopenjpeg/openjpeg.h"
#include "codec/index.h"

#define wxBITMAP_TYPE_JPEG2000	50

class WXDLLEXPORT wxJPEG2000Handler: public wxImageHandler
{
public:
    inline wxJPEG2000Handler()
    {
        m_name = wxT("JPEG 2000 family file format");
        m_extension = wxT("mj2");
        m_type = wxBITMAP_TYPE_JPEG2000;
        m_mime = wxT("image/mj2");

		/* decoding */
		m_reducefactor = 0;
		m_qualitylayers = 0;
		m_components = 0;
#ifdef USE_JPWL
		m_enablejpwl = true;
		m_expcomps = JPWL_EXPECTED_COMPONENTS;
		m_maxtiles = JPWL_MAXIMUM_TILES;
#endif // USE_JPWL

		/* encoding */
		m_subsampling = wxT("1,1");
		m_origin = wxT("0,0");
		m_rates = wxT("20,10,5");
		m_quality = wxT("30,35,40");
		m_enablequality = false;
		m_multicomp = false;
		m_irreversible = false;
		m_resolutions = 6;
		m_progression = 0;
		m_cbsize = wxT("32,32");
		m_prsize = wxT("[128,128],[128,128]");
		m_tsize = wxT("");
		m_torigin = wxT("0,0");
		/*m_progression
		m_resilience*/
		m_enablesop = false;
		m_enableeph = false;
		m_enablereset = false;
		m_enablesegmark = false;
		m_enablevsc = false;
		m_enablerestart = false;
		m_enableerterm = false;
		m_enablebypass = false;
		/*m_roicompo
		m_roiup
		m_indexfname*/
		m_enableidx = false;
		m_index = wxT("index.txt");
		m_enablepoc = false;
		m_poc = wxT("T1=0,0,1,5,3,CPRL/T1=5,0,1,6,3,CPRL");
		m_enablecomm = true;

#if defined __WXMSW__
		m_comment = wxT("Created by OPJViewer Win32 - OpenJPEG  version ");
#elif defined __WXGTK__
		m_comment = wxT("Created by OPJViewer Lin32 - OpenJPEG version ");
#else
		m_comment = wxT("Created by OPJViewer - OpenJPEG version ");
#endif

#ifdef USE_JPWL
		m_comment += wxString::Format(wxT("%s with JPWL"), (char *) opj_version());
#else
		m_comment += wxString::Format(wxT("%s"), (char *) opj_version());
#endif

    }

	// decoding engine parameters
	int m_reducefactor, m_qualitylayers, m_components, m_framenum;
#ifdef USE_JPWL
	bool m_enablejpwl;
	int m_expcomps, m_maxtiles;
#endif // USE_JPWL

	// encoding engine parameters
	wxString m_subsampling;
	wxString m_origin;
	wxString m_rates;
	wxString m_quality;
	bool m_enablequality;
	bool m_multicomp;
	bool m_irreversible;
	int m_resolutions;
	int m_progression;
	wxString m_cbsize;
	wxString m_prsize;
	wxString m_tsize;
	wxString m_torigin;
	/*m_progression
	m_resilience*/
	bool m_enablesop;
	bool m_enableeph;
	bool m_enablebypass;
	bool m_enableerterm;
	bool m_enablerestart;
	bool m_enablereset;
	bool m_enablesegmark;
	bool m_enablevsc;
	/*m_roicompo
	m_roiup
	m_indexfname*/
	bool m_enableidx;
	wxString m_index;
	bool m_enablecomm;
	wxString m_comment;
	bool m_enablepoc;
	wxString m_poc;

#if wxUSE_STREAMS
    virtual bool LoadFile(wxImage *image, wxInputStream& stream, bool verbose=true, int index=-1);
    virtual bool SaveFile(wxImage *image, wxOutputStream& stream, bool verbose=true);
protected:
    virtual bool DoCanRead(wxInputStream& stream);
#endif

private:
	OPJ_PROG_ORDER give_progression(char progression[4]);
    DECLARE_DYNAMIC_CLASS(wxJPEG2000Handler)
};

#endif // wxUSE_LIBOPENJPEG

#endif // _WX_IMAGJPEG2000_H_

