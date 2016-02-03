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
// Name:        imagmxf.h
// Purpose:     wxImage MXF (Material eXchange Format) JPEG 2000 file format handler
// Author:      G. Baruffa - based on imagjpeg.h, Vaclav Slavik
// RCS-ID:      $Id: imagmj2.h,v 0.0 2007/11/19 17:00:00 VZ Exp $
// Copyright:   (c) Giuseppe Baruffa
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_IMAGMXF_H_
#define _WX_IMAGMXF_H_

#ifdef USE_MXF

#include "wx/defs.h"
#include "wx/filename.h"

//-----------------------------------------------------------------------------
// wxMXFHandler
//-----------------------------------------------------------------------------

#if wxUSE_LIBOPENJPEG

#include "wx/image.h"
#include "libopenjpeg/openjpeg.h"

#define wxBITMAP_TYPE_MXF	51

class WXDLLEXPORT wxMXFHandler: public wxImageHandler
{
public:
    inline wxMXFHandler()
    {
        m_name = wxT("MXF JPEG 2000 file format");
        m_extension = wxT("mxf");
        m_type = wxBITMAP_TYPE_MXF;
        m_mime = wxT("image/mxf");

		m_reducefactor = 0;
		m_qualitylayers = 0;
		m_components = 0;
		m_filename = wxT("");
#ifdef USE_JPWL
		m_enablejpwl = true;
		m_expcomps = JPWL_EXPECTED_COMPONENTS;
		m_maxtiles = JPWL_MAXIMUM_TILES;
#endif // USE_JPWL
    }

		// decoding engine parameters
		int m_reducefactor, m_qualitylayers, m_components, m_framenum;
		wxFileName m_filename;
#ifdef USE_JPWL
		bool m_enablejpwl;
		int m_expcomps, m_maxtiles;
#endif // USE_JPWL

#if wxUSE_STREAMS
    virtual bool LoadFile( wxImage *image, wxInputStream& stream, bool verbose=true, int index=-1 );
    virtual bool SaveFile( wxImage *image, wxOutputStream& stream, bool verbose=true );
protected:
    virtual bool DoCanRead( wxInputStream& stream );
#endif

private:
    DECLARE_DYNAMIC_CLASS(wxMXFHandler)
};

#endif // wxUSE_LIBOPENJPEG

#endif // USE_MXF

#endif // _WX_IMAGMXF_H_

