/*
 * Copyright (c) 2007, Digital Signal Processing Laboratory, Universita'  degli studi di Perugia (UPG), Italy
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
#ifdef USE_MXF
#include "mxflib/mxflib.h"
#endif // USE_MXF

#include "OPJViewer.h"

// about window for the frame
void OPJFrame::OnAbout(wxCommandEvent& WXUNUSED(event))
{
#ifdef OPJ_HTMLABOUT
#include "about_htm.h"
#include "opj_logo.xpm"

    wxBoxSizer *topsizer;
    wxHtmlWindow *html;
    wxDialog dlg(this, wxID_ANY, wxString(_("About")));

    wxMemoryFSHandler::AddFile(wxT("opj_logo.xpm"), wxBitmap(opj_logo), wxBITMAP_TYPE_XPM);

    topsizer = new wxBoxSizer(wxVERTICAL);

    html = new wxHtmlWindow(&dlg, wxID_ANY, wxDefaultPosition, wxSize(320, 250), wxHW_SCROLLBAR_NEVER);
    html->SetBorders(0);
    //html->LoadPage(wxT("about/about.htm"));
	//html->SetPage("<html><body>Hello, world!</body></html>");
	html->SetPage(htmlaboutpage);
    html->SetSize(html->GetInternalRepresentation()->GetWidth(),
                    html->GetInternalRepresentation()->GetHeight());

    topsizer->Add(html, 1, wxALL, 10);

    topsizer->Add(new wxStaticLine(&dlg, wxID_ANY), 0, wxEXPAND | wxLEFT | wxRIGHT, 10);

    wxButton *bu1 = new wxButton(&dlg, wxID_OK, wxT("OK"));
    bu1->SetDefault();

    topsizer->Add(bu1, 0, wxALL | wxALIGN_RIGHT, 15);

    dlg.SetSizer(topsizer);
    topsizer->Fit(&dlg);

    dlg.ShowModal();

#else

	wxMessageBox(wxString::Format(OPJ_APPLICATION_TITLEBAR
								  wxT("\n\n")
								  wxT("Built with %s and OpenJPEG ")
								  wxT(OPENJPEG_VERSION)
								  wxT("\non ") wxT(__DATE__) wxT(", ") wxT(__TIME__)
								  wxT("\nRunning under %s\n\n")
								  OPJ_APPLICATION_COPYRIGHT,
								  wxVERSION_STRING,
								  wxGetOsDescription().c_str()),
				 wxT("About ") OPJ_APPLICATION_NAME,
				 wxOK | wxICON_INFORMATION,
				 this
				 );

#endif

}
