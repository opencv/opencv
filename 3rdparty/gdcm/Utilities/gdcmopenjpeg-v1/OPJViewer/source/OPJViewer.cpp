/*
 * Copyright (c) 2007, Digital Signal Processing Laboratory, Universita' degli studi di Perugia (UPG), Italy
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
// Name:        sashtest.cpp
// Purpose:     Layout/sash sample
// Author:      Julian Smart
// Modified by:
// Created:     04/01/98
// RCS-ID:      $Id: sashtest.cpp,v 1.18 2005/08/23 15:54:35 ABX Exp $
// Copyright:   (c) Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        treetest.cpp
// Purpose:     wxTreeCtrl sample
// Author:      Julian Smart
// Modified by:
// Created:     04/01/98
// RCS-ID:      $Id: treetest.cpp,v 1.110 2006/11/04 11:26:51 VZ Exp $
// Copyright:   (c) Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        dialogs.cpp
// Purpose:     Common dialogs demo
// Author:      Julian Smart
// Modified by: ABX (2004) - adjustements for conditional building + new menu
// Created:     04/01/98
// RCS-ID:      $Id: dialogs.cpp,v 1.163 2006/11/04 10:57:24 VZ Exp $
// Copyright:   (c) Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        thread.cpp
// Purpose:     wxWidgets thread sample
// Author:      Guilhem Lavaux, Vadim Zeitlin
// Modified by:
// Created:     06/16/98
// RCS-ID:      $Id: thread.cpp,v 1.26 2006/10/02 05:36:28 PC Exp $
// Copyright:   (c) 1998-2002 wxWidgets team
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Name:        samples/image/image.cpp
// Purpose:     sample showing operations with wxImage
// Author:      Robert Roebling
// Modified by:
// Created:     1998
// RCS-ID:      $Id: image.cpp,v 1.120 2006/12/06 17:13:11 VZ Exp $
// Copyright:   (c) 1998-2005 Robert Roebling
// License:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        samples/console/console.cpp
// Purpose:     A sample console (as opposed to GUI) program using wxWidgets
// Author:      Vadim Zeitlin
// Modified by:
// Created:     04.10.99
// RCS-ID:      $Id: console.cpp,v 1.206 2006/11/12 19:55:19 VZ Exp $
// Copyright:   (c) 1999 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        samples/notebook/notebook.cpp
// Purpose:     a sample demonstrating notebook usage
// Author:      Julian Smart
// Modified by: Dimitri Schoolwerth
// Created:     26/10/98
// RCS-ID:      $Id: notebook.cpp,v 1.49 2006/11/04 18:24:07 RR Exp $
// Copyright:   (c) 1998-2002 wxWidgets team
// License:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        dialogs.cpp
// Purpose:     Common dialogs demo
// Author:      Julian Smart
// Modified by: ABX (2004) - adjustements for conditional building + new menu
// Created:     04/01/98
// RCS-ID:      $Id: dialogs.cpp,v 1.163 2006/11/04 10:57:24 VZ Exp $
// Copyright:   (c) Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        dnd.cpp
// Purpose:     Drag and drop sample
// Author:      Vadim Zeitlin
// Modified by:
// Created:     04/01/98
// RCS-ID:      $Id: dnd.cpp,v 1.107 2006/10/30 20:23:41 VZ Exp $
// Copyright:
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        test.cpp
// Purpose:     wxHtml testing example
/////////////////////////////////////////////////////////////////////////////


#include "OPJViewer.h"

IMPLEMENT_APP(OPJViewerApp)

// For drawing lines in a canvas
long xpos = -1;
long ypos = -1;

int winNumber = 1;

// Initialise this in OnInit, not statically
bool OPJViewerApp::OnInit(void)
{
	int n;
#if wxUSE_UNICODE

    wxChar **wxArgv = new wxChar *[argc + 1];

    for (n = 0; n < argc; n++ ) {
        wxMB2WXbuf warg = wxConvertMB2WX((char *) argv[n]);
        wxArgv[n] = wxStrdup(warg);
    }

    wxArgv[n] = NULL;

#else // !wxUSE_UNICODE

    #define wxArgv argv

#endif // wxUSE_UNICODE/!wxUSE_UNICODE

#if wxUSE_CMDLINE_PARSER

    static const wxCmdLineEntryDesc cmdLineDesc[] =
    {
        { wxCMD_LINE_SWITCH, _T("h"), _T("help"), _T("show this help message"),
            wxCMD_LINE_VAL_NONE, wxCMD_LINE_OPTION_HELP },

        { wxCMD_LINE_PARAM,  NULL, NULL, _T("input file"),
            wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },

        { wxCMD_LINE_NONE }
    };

    wxCmdLineParser parser(cmdLineDesc, argc, wxArgv);

    switch (parser.Parse()) {
    case -1:
        wxLogMessage(wxT("Help was given, terminating."));
        break;

    case 0:
        ShowCmdLine(parser);
        break;

    default:
        wxLogMessage(wxT("Syntax error detected."));
        break;
    }

#endif // wxUSE_CMDLINE_PARSER

    //wxInitAllImageHandlers();
#if wxUSE_LIBJPEG
  wxImage::AddHandler( new wxJPEGHandler );
#endif
#if wxUSE_LIBOPENJPEG
  wxImage::AddHandler( new wxJPEG2000Handler );
#endif
#if USE_MXF
  wxImage::AddHandler( new wxMXFHandler );
#endif // USE_MXF
#if OPJ_MANYFORMATS
  wxImage::AddHandler( new wxBMPHandler );
  wxImage::AddHandler( new wxPNGHandler );
  wxImage::AddHandler( new wxGIFHandler );
  wxImage::AddHandler( new wxPNMHandler );
  wxImage::AddHandler( new wxTIFFHandler );
#endif
    // we use a XPM image in our HTML page
    wxImage::AddHandler(new wxXPMHandler);

	// memory file system
    wxFileSystem::AddHandler(new wxMemoryFSHandler);

#ifdef OPJ_INICONFIG
	//load decoding engine parameters
	OPJconfig = new wxConfig(OPJ_APPLICATION, OPJ_APPLICATION_VENDOR);

	OPJconfig->Read(wxT("decode/enabledeco"), &m_enabledeco, (bool) true);
	OPJconfig->Read(wxT("decode/enableparse"), &m_enableparse, (bool) true);
	OPJconfig->Read(wxT("decode/resizemethod"), &m_resizemethod, (long) 0);
	OPJconfig->Read(wxT("decode/xxxreducefactor"), &m_reducefactor, (long) 0);
	OPJconfig->Read(wxT("decode/xxxqualitylayers"), &m_qualitylayers, (long) 0);
	OPJconfig->Read(wxT("decode/xxxcomponents"), &m_components, (long) 0);
	OPJconfig->Read(wxT("decode/xxxframenum"), &m_framenum, (long) 0);
#ifdef USE_JPWL
	OPJconfig->Read(wxT("decode/enablejpwl"), &m_enablejpwl, (bool) true);
	OPJconfig->Read(wxT("decode/expcomps"), &m_expcomps, (long) JPWL_EXPECTED_COMPONENTS);
	OPJconfig->Read(wxT("decode/maxtiles"), &m_maxtiles, (long) JPWL_MAXIMUM_TILES);
#endif // USE_JPWL

	OPJconfig->Write(wxT("teststring"), wxT("This is a test value"));
	OPJconfig->Write(wxT("testbool"), (bool) true);
	OPJconfig->Write(wxT("testlong"), (long) 245);

	OPJconfig->Read(wxT("showtoolbar"), &m_showtoolbar, (bool) true);
	OPJconfig->Read(wxT("showbrowser"), &m_showbrowser, (bool) true);
	OPJconfig->Read(wxT("showpeeker"), &m_showpeeker, (bool) true);
	OPJconfig->Read(wxT("browserwidth"), &m_browserwidth, (long) OPJ_BROWSER_WIDTH);
	OPJconfig->Read(wxT("peekerheight"), &m_peekerheight, (long) OPJ_PEEKER_HEIGHT);
	OPJconfig->Read(wxT("framewidth"), &m_framewidth, (long) OPJ_FRAME_WIDTH);
	OPJconfig->Read(wxT("frameheight"), &m_frameheight, (long) OPJ_FRAME_HEIGHT);

	// load encoding engine parameters
	OPJconfig->Read(wxT("encode/subsampling"), &m_subsampling, (wxString) wxT("1,1"));
	OPJconfig->Read(wxT("encode/origin"), &m_origin, (wxString) wxT("0,0"));
	OPJconfig->Read(wxT("encode/rates"), &m_rates, (wxString) wxT("20,10,5"));
	OPJconfig->Read(wxT("encode/quality"), &m_quality, (wxString) wxT("30,35,40"));
	OPJconfig->Read(wxT("encode/enablequality"), &m_enablequality, (bool) false);
	OPJconfig->Read(wxT("encode/multicomp"), &m_multicomp, (bool) false);	
	OPJconfig->Read(wxT("encode/irreversible"), &m_irreversible, (bool) false);	
	OPJconfig->Read(wxT("encode/resolutions"), &m_resolutions, (int) 6);	
	OPJconfig->Read(wxT("encode/progression"), &m_progression, (int) 0);	
	OPJconfig->Read(wxT("encode/cbsize"), &m_cbsize, (wxString) wxT("32,32"));
	OPJconfig->Read(wxT("encode/prsize"), &m_prsize, (wxString) wxT("[128,128],[128,128]"));
	OPJconfig->Read(wxT("encode/tsize"), &m_tsize, (wxString) wxT(""));
	OPJconfig->Read(wxT("encode/torigin"), &m_torigin, (wxString) wxT("0,0"));
	OPJconfig->Read(wxT("encode/enablesop"), &m_enablesop, (bool) false);	
	OPJconfig->Read(wxT("encode/enableeph"), &m_enableeph, (bool) false);	
	OPJconfig->Read(wxT("encode/enablebypass"), &m_enablebypass, (bool) false);	
	OPJconfig->Read(wxT("encode/enablereset"), &m_enablereset, (bool) false);	
	OPJconfig->Read(wxT("encode/enablerestart"), &m_enablerestart, (bool) false);	
	OPJconfig->Read(wxT("encode/enablevsc"), &m_enablevsc, (bool) false);	
	OPJconfig->Read(wxT("encode/enableerterm"), &m_enableerterm, (bool) false);	
	OPJconfig->Read(wxT("encode/enablesegmark"), &m_enablesegmark, (bool) false);	
	OPJconfig->Read(wxT("encode/enablecomm"), &m_enablecomm, (bool) true);	
	OPJconfig->Read(wxT("encode/enablepoc"), &m_enablepoc, (bool) false);	
	OPJconfig->Read(wxT("encode/comment"), &m_comment, (wxString) wxT(""));
	OPJconfig->Read(wxT("encode/poc"), &m_poc, (wxString) wxT("T1=0,0,1,5,3,CPRL/T1=5,0,1,6,3,CPRL"));
	OPJconfig->Read(wxT("encode/enableidx"), &m_enableidx, (bool) false);	
	OPJconfig->Read(wxT("encode/index"), &m_index, (wxString) wxT("index.txt"));
#ifdef USE_JPWL
	OPJconfig->Read(wxT("encode/enablejpwl"), &m_enablejpwle, (bool) true);
	for (n = 0; n < MYJPWL_MAX_NO_TILESPECS; n++) {
		OPJconfig->Read(wxT("encode/jpwl/hprotsel") + wxString::Format(wxT("%02d"), n), &m_hprotsel[n], 0);
		OPJconfig->Read(wxT("encode/jpwl/htileval") + wxString::Format(wxT("%02d"), n), &m_htileval[n], 0);
		OPJconfig->Read(wxT("encode/jpwl/pprotsel") + wxString::Format(wxT("%02d"), n), &m_pprotsel[n], 0);
		OPJconfig->Read(wxT("encode/jpwl/ptileval") + wxString::Format(wxT("%02d"), n), &m_ptileval[n], 0);
		OPJconfig->Read(wxT("encode/jpwl/ppackval") + wxString::Format(wxT("%02d"), n), &m_ppackval[n], 0);
		OPJconfig->Read(wxT("encode/jpwl/sensisel") + wxString::Format(wxT("%02d"), n), &m_sensisel[n], 0);
		OPJconfig->Read(wxT("encode/jpwl/stileval") + wxString::Format(wxT("%02d"), n), &m_stileval[n], 0);
	}
#endif // USE_JPWL

#else
	// set decoding engine parameters
	m_enabledeco = true;
	m_enableparse = true;
	m_resizemethod = 0;
	m_reducefactor = 0;
	m_qualitylayers = 0;
	m_components = 0;
	m_framenum = 0;
#ifdef USE_JPWL
	m_enablejpwl = true;
	m_expcomps = JPWL_EXPECTED_COMPONENTS;
	m_maxtiles = JPWL_MAXIMUM_TILES;
#endif // USE_JPWL
	m_showtoolbar = true;
	m_showbrowser = true;
	m_showpeeker = true;
	m_browserwidth = OPJ_BROWSER_WIDTH;
	m_peekerheight = OPJ_PEEKER_HEIGHT;
	m_framewidth = OPJ_FRAME_WIDTH;
	m_frameheight = OPJ_FRAME_HEIGHT;

	// set encoding engine parameters
	m_subsampling = wxT("1,1");
	m_origin = wxT("0,0");
	m_rates = wxT("20,10,5");
	m_quality = wxT("30,35,40");
	m_enablequality = false;
	m_multicomp = false;
	m_irreversible = false;
	m_resolutions = 6;
	m_progression = 0;
	m_cbsize= wxT("32,32");
	m_prsize= wxT("[128,128],[128,128]");
	m_tsize = wxT("");
	m_torigin = wxT("0,0");
	m_enablesop = false;
	m_enableeph = false;
	m_enablebypass = false;
	m_enablereset = false;
	m_enablerestart = false;
	m_enablevsc = false;
	m_enableerterm = false;
	m_enablesegmark = false;
	m_enableidx = false;
	m_index = wxT("index.txt");
	m_enablecomm = true;
	m_comment = wxT("");
	m_enablepoc = false;
	m_poc = wxT("T1=0,0,1,5,3,CPRL/T1=5,0,1,6,3,CPRL");
#ifdef USE_JPWL
	m_enablejpwle = true;
	for (n = 0; n < MYJPWL_MAX_NO_TILESPECS; n++) {
		m_hprotsel[n] = 0;
		m_htileval[n] = 0;
		m_pprotsel[n] = 0;
		m_ptileval[n] = 0;
		m_sensisel[n] = 0;
		m_stileval[n] = 0;
	}
#endif // USE_JPWL

#endif // OPJ_INICONFIG

	if (m_comment == wxT("")) {
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

	// Create the main frame window
  OPJFrame *frame = new OPJFrame(NULL, wxID_ANY, OPJ_APPLICATION_TITLEBAR,
					  wxDefaultPosition, wxSize(wxGetApp().m_framewidth, wxGetApp().m_frameheight),
                      wxDEFAULT_FRAME_STYLE | wxNO_FULL_REPAINT_ON_RESIZE |
                      wxHSCROLL | wxVSCROLL);

  // Give it an icon (this is ignored in MDI mode: uses resources)
#ifdef __WXMSW__
  frame->SetIcon(wxIcon(wxT("OPJViewer16")));
#endif

  frame->Show(true);

  SetTopWindow(frame);

	// if there are files on the command line, open them
	if (!(m_filelist.IsEmpty())) {
		//wxLogMessage(wxT("Habemus files!!!"));
		wxArrayString paths, filenames;
		for (unsigned int f = 0; f < wxGetApp().m_filelist.GetCount(); f++) {
			paths.Add(wxFileName(wxGetApp().m_filelist[f]).GetFullPath());
			filenames.Add(wxFileName(wxGetApp().m_filelist[f]).GetFullName());
		}
		//wxLogMessage(paths[0]);
		frame->OpenFiles(paths, filenames);
	}

  return true;
}

int OPJViewerApp::OnExit()
{
	int n;

#ifdef OPJ_INICONFIG
	OPJconfig->Write(wxT("decode/enabledeco"), m_enabledeco);
	OPJconfig->Write(wxT("decode/enableparse"), m_enableparse);
	OPJconfig->Write(wxT("decode/resizemethod"), m_resizemethod);
	OPJconfig->Write(wxT("decode/reducefactor"), m_reducefactor);
	OPJconfig->Write(wxT("decode/qualitylayers"), m_qualitylayers);
	OPJconfig->Write(wxT("decode/components"), m_components);
	OPJconfig->Write(wxT("decode/framenum"), m_framenum);
#ifdef USE_JPWL
	OPJconfig->Write(wxT("decode/enablejpwl"), m_enablejpwl);
	OPJconfig->Write(wxT("decode/expcomps"), m_expcomps);
	OPJconfig->Write(wxT("decode/maxtiles"), m_maxtiles);
#endif // USE_JPWL
	OPJconfig->Write(wxT("showtoolbar"), m_showtoolbar);
	OPJconfig->Write(wxT("showbrowser"), m_showbrowser);
	OPJconfig->Write(wxT("showpeeker"), m_showpeeker);
	OPJconfig->Write(wxT("browserwidth"), m_browserwidth);
	OPJconfig->Write(wxT("peekerheight"), m_peekerheight);
	OPJconfig->Write(wxT("framewidth"), m_framewidth);
	OPJconfig->Write(wxT("frameheight"), m_frameheight);

	OPJconfig->Write(wxT("encode/subsampling"), m_subsampling);
	OPJconfig->Write(wxT("encode/origin"), m_origin);
	OPJconfig->Write(wxT("encode/rates"), m_rates);
	OPJconfig->Write(wxT("encode/quality"), m_quality);
	OPJconfig->Write(wxT("encode/enablequality"), m_enablequality);
	OPJconfig->Write(wxT("encode/multicomp"), m_multicomp);
	OPJconfig->Write(wxT("encode/irreversible"), m_irreversible);
	OPJconfig->Write(wxT("encode/resolutions"), m_resolutions);
	OPJconfig->Write(wxT("encode/progression"), m_progression);
	OPJconfig->Write(wxT("encode/cbsize"), m_cbsize);
	OPJconfig->Write(wxT("encode/prsize"), m_prsize);
	OPJconfig->Write(wxT("encode/tiles"), m_tsize);
	OPJconfig->Write(wxT("encode/torigin"), m_torigin);
	OPJconfig->Write(wxT("encode/enablesop"), m_enablesop);
	OPJconfig->Write(wxT("encode/enableeph"), m_enableeph);
	OPJconfig->Write(wxT("encode/enablebypass"), m_enablebypass);
	OPJconfig->Write(wxT("encode/enablereset"), m_enablereset);
	OPJconfig->Write(wxT("encode/enablerestart"), m_enablerestart);
	OPJconfig->Write(wxT("encode/enablevsc"), m_enablevsc);
	OPJconfig->Write(wxT("encode/enableerterm"), m_enableerterm);
	OPJconfig->Write(wxT("encode/enablesegmark"), m_enablesegmark);
	OPJconfig->Write(wxT("encode/enableidx"), m_enableidx);
	OPJconfig->Write(wxT("encode/index"), m_index);
	OPJconfig->Write(wxT("encode/enablecomm"), m_enablecomm);
	OPJconfig->Write(wxT("encode/comment"), m_comment);
	OPJconfig->Write(wxT("encode/enablepoc"), m_enablepoc);
	OPJconfig->Write(wxT("encode/poc"), m_poc);
#ifdef USE_JPWL
	OPJconfig->Write(wxT("encode/enablejpwl"), m_enablejpwle);
	for (n = 0; n < MYJPWL_MAX_NO_TILESPECS; n++) {
		OPJconfig->Write(wxT("encode/jpwl/hprotsel") + wxString::Format(wxT("%02d"), n), m_hprotsel[n]);
		OPJconfig->Write(wxT("encode/jpwl/htileval") + wxString::Format(wxT("%02d"), n), m_htileval[n]);
		OPJconfig->Write(wxT("encode/jpwl/pprotsel") + wxString::Format(wxT("%02d"), n), m_pprotsel[n]);
		OPJconfig->Write(wxT("encode/jpwl/ptileval") + wxString::Format(wxT("%02d"), n), m_ptileval[n]);
		OPJconfig->Write(wxT("encode/jpwl/ppackval") + wxString::Format(wxT("%02d"), n), m_ppackval[n]);
		OPJconfig->Write(wxT("encode/jpwl/sensisel") + wxString::Format(wxT("%02d"), n), m_sensisel[n]);
		OPJconfig->Write(wxT("encode/jpwl/stileval") + wxString::Format(wxT("%02d"), n), m_stileval[n]);
	}
#endif // USE_JPWL

#endif // OPJ_INICONFIG

	return 1;
}

void OPJViewerApp::ShowCmdLine(const wxCmdLineParser& parser)
{
    wxString s = wxT("Command line parsed successfully:\nInput files: ");

    size_t count = parser.GetParamCount();
    for (size_t param = 0; param < count; param++) {
        s << parser.GetParam(param) << ';';
		m_filelist.Add(parser.GetParam(param));
    }

    //wxLogMessage(s);
}

// OPJFrame events

// Event class for sending text messages between worker and GUI threads
BEGIN_EVENT_TABLE(OPJFrame, wxMDIParentFrame)
    EVT_MENU(OPJFRAME_HELPABOUT, OPJFrame::OnAbout)
    EVT_MENU(OPJFRAME_FILEOPEN, OPJFrame::OnFileOpen)
    EVT_MENU(OPJFRAME_FILESAVEAS, OPJFrame::OnFileSaveAs)
    EVT_MENU(OPJFRAME_MEMORYOPEN, OPJFrame::OnMemoryOpen)
    EVT_SIZE(OPJFrame::OnSize)
    EVT_MENU(OPJFRAME_FILEEXIT, OPJFrame::OnQuit)
    EVT_MENU(OPJFRAME_FILECLOSE, OPJFrame::OnClose)
    EVT_MENU(OPJFRAME_VIEWZOOM, OPJFrame::OnZoom)
    EVT_MENU(OPJFRAME_VIEWFIT, OPJFrame::OnFit)
    EVT_MENU(OPJFRAME_VIEWRELOAD, OPJFrame::OnReload)
    EVT_MENU(OPJFRAME_VIEWPREVFRAME, OPJFrame::OnPrevFrame)
    EVT_MENU(OPJFRAME_VIEWHOMEFRAME, OPJFrame::OnHomeFrame)
    EVT_MENU(OPJFRAME_VIEWNEXTFRAME, OPJFrame::OnNextFrame)
    EVT_MENU(OPJFRAME_VIEWLESSLAYERS, OPJFrame::OnLessLayers)
    EVT_MENU(OPJFRAME_VIEWALLLAYERS, OPJFrame::OnAllLayers)
    EVT_MENU(OPJFRAME_VIEWMORELAYERS, OPJFrame::OnMoreLayers)
    EVT_MENU(OPJFRAME_VIEWLESSRES, OPJFrame::OnLessRes)
    EVT_MENU(OPJFRAME_VIEWFULLRES, OPJFrame::OnFullRes)
    EVT_MENU(OPJFRAME_VIEWMORERES, OPJFrame::OnMoreRes)
    EVT_MENU(OPJFRAME_VIEWPREVCOMP, OPJFrame::OnPrevComp)
    EVT_MENU(OPJFRAME_VIEWALLCOMPS, OPJFrame::OnAllComps)
    EVT_MENU(OPJFRAME_VIEWNEXTCOMP, OPJFrame::OnNextComp)
    EVT_MENU(OPJFRAME_FILETOGGLEB, OPJFrame::OnToggleBrowser)
    EVT_MENU(OPJFRAME_FILETOGGLEP, OPJFrame::OnTogglePeeker)
    EVT_MENU(OPJFRAME_FILETOGGLET, OPJFrame::OnToggleToolbar)
    EVT_MENU(OPJFRAME_SETSENCO, OPJFrame::OnSetsEnco)
    EVT_MENU(OPJFRAME_SETSDECO, OPJFrame::OnSetsDeco)
    EVT_SASH_DRAGGED_RANGE(OPJFRAME_BROWSEWIN, OPJFRAME_LOGWIN, OPJFrame::OnSashDrag)
    EVT_NOTEBOOK_PAGE_CHANGED(LEFT_NOTEBOOK_ID, OPJFrame::OnNotebook)
    EVT_MENU(OPJFRAME_THREADLOGMSG, OPJFrame::OnThreadLogmsg)
END_EVENT_TABLE()

// this is the frame constructor
OPJFrame::OPJFrame(wxWindow *parent, const wxWindowID id, const wxString& title,
				   const wxPoint& pos, const wxSize& size, const long style)
		: wxMDIParentFrame(parent, id, title, pos, size, style)
{
	// file menu and its items
	wxMenu *file_menu = new wxMenu;

	file_menu->Append(OPJFRAME_FILEOPEN, wxT("&Open\tCtrl+O"));
	file_menu->SetHelpString(OPJFRAME_FILEOPEN, wxT("Open one or more files"));

	file_menu->Append(OPJFRAME_MEMORYOPEN, wxT("&Memory\tCtrl+M"));
	file_menu->SetHelpString(OPJFRAME_MEMORYOPEN, wxT("Open a memory buffer"));

	file_menu->Append(OPJFRAME_FILECLOSE, wxT("&Close\tCtrl+C"));
	file_menu->SetHelpString(OPJFRAME_FILECLOSE, wxT("Close current image"));

	file_menu->AppendSeparator();

	file_menu->Append(OPJFRAME_FILESAVEAS, wxT("&Save as\tCtrl+S"));
	file_menu->SetHelpString(OPJFRAME_FILESAVEAS, wxT("Save the current image"));
	//file_menu->Enable(OPJFRAME_FILESAVEAS, false);

	file_menu->AppendSeparator();

	file_menu->Append(OPJFRAME_FILETOGGLEB, wxT("Toggle &browser\tCtrl+B"));
	file_menu->SetHelpString(OPJFRAME_FILETOGGLEB, wxT("Toggle the left browsing pane"));

	file_menu->Append(OPJFRAME_FILETOGGLEP, wxT("Toggle &peeker\tCtrl+P"));
	file_menu->SetHelpString(OPJFRAME_FILETOGGLEP, wxT("Toggle the bottom peeking pane"));

	file_menu->Append(OPJFRAME_FILETOGGLET, wxT("Toggle &toolbar\tCtrl+T"));
	file_menu->SetHelpString(OPJFRAME_FILETOGGLET, wxT("Toggle the toolbar"));

	file_menu->AppendSeparator();

	file_menu->Append(OPJFRAME_FILEEXIT, wxT("&Exit\tCtrl+Q"));
	file_menu->SetHelpString(OPJFRAME_FILEEXIT, wxT("Quit this program"));

	// view menu and its items
	wxMenu *view_menu = new wxMenu;

	view_menu->Append(OPJFRAME_VIEWZOOM, wxT("&Zoom\tCtrl+Z"));
	view_menu->SetHelpString(OPJFRAME_VIEWZOOM, wxT("Rescale the image"));

	view_menu->Append(OPJFRAME_VIEWFIT, wxT("Zoom to &fit\tCtrl+F"));
	view_menu->SetHelpString(OPJFRAME_VIEWFIT, wxT("Fit the image in canvas"));

	view_menu->Append(OPJFRAME_VIEWRELOAD, wxT("&Reload image\tCtrl+R"));
	view_menu->SetHelpString(OPJFRAME_VIEWRELOAD, wxT("Reload the current image"));

	view_menu->AppendSeparator();

	view_menu->Append(OPJFRAME_VIEWPREVFRAME, wxT("&Prev frame\tLeft"));
	view_menu->SetHelpString(OPJFRAME_VIEWPREVFRAME, wxT("View previous frame"));

	view_menu->Append(OPJFRAME_VIEWHOMEFRAME, wxT("&Start frame\tHome"));
	view_menu->SetHelpString(OPJFRAME_VIEWHOMEFRAME, wxT("View starting frame"));

	view_menu->Append(OPJFRAME_VIEWNEXTFRAME, wxT("&Next frame\tRight"));
	view_menu->SetHelpString(OPJFRAME_VIEWNEXTFRAME, wxT("View next frame"));

	view_menu->AppendSeparator();

	view_menu->Append(OPJFRAME_VIEWLESSLAYERS, wxT("&Less layers\t-"));
	view_menu->SetHelpString(OPJFRAME_VIEWLESSLAYERS, wxT("Remove a layer"));

	view_menu->Append(OPJFRAME_VIEWALLLAYERS, wxT("&All layers\t0"));
	view_menu->SetHelpString(OPJFRAME_VIEWALLLAYERS, wxT("Show all layers"));

	view_menu->Append(OPJFRAME_VIEWMORELAYERS, wxT("&More layers\t+"));
	view_menu->SetHelpString(OPJFRAME_VIEWMORELAYERS, wxT("Add a layer"));

	view_menu->AppendSeparator();

	view_menu->Append(OPJFRAME_VIEWLESSRES, wxT("&Less resolution\t<"));
	view_menu->SetHelpString(OPJFRAME_VIEWLESSRES, wxT("Reduce the resolution"));

	view_menu->Append(OPJFRAME_VIEWFULLRES, wxT("&Full resolution\tf"));
	view_menu->SetHelpString(OPJFRAME_VIEWFULLRES, wxT("Full resolution"));

	view_menu->Append(OPJFRAME_VIEWMORERES, wxT("&More resolution\t>"));
	view_menu->SetHelpString(OPJFRAME_VIEWMORERES, wxT("Increase the resolution"));

	view_menu->AppendSeparator();

	view_menu->Append(OPJFRAME_VIEWPREVCOMP, wxT("&Prev component\tDown"));
	view_menu->SetHelpString(OPJFRAME_VIEWPREVCOMP, wxT("View previous component"));

	view_menu->Append(OPJFRAME_VIEWALLCOMPS, wxT("&All components\ta"));
	view_menu->SetHelpString(OPJFRAME_VIEWALLCOMPS, wxT("View all components"));

	view_menu->Append(OPJFRAME_VIEWNEXTCOMP, wxT("&Next component\tUp"));
	view_menu->SetHelpString(OPJFRAME_VIEWNEXTCOMP, wxT("View next component"));


	// settings menu and its items
	wxMenu *sets_menu = new wxMenu;

	sets_menu->Append(OPJFRAME_SETSENCO, wxT("&Encoder\tCtrl+E"));
	sets_menu->SetHelpString(OPJFRAME_SETSENCO, wxT("Encoder settings"));

	sets_menu->Append(OPJFRAME_SETSDECO, wxT("&Decoder\tCtrl+D"));
	sets_menu->SetHelpString(OPJFRAME_SETSDECO, wxT("Decoder settings"));

	// help menu and its items
	wxMenu *help_menu = new wxMenu;

	help_menu->Append(OPJFRAME_HELPABOUT, wxT("&About\tF1"));
	help_menu->SetHelpString(OPJFRAME_HELPABOUT, wxT("Basic info on the program"));

	// the whole menubar
	wxMenuBar *menu_bar = new wxMenuBar;
	menu_bar->Append(file_menu, wxT("&File"));
	menu_bar->Append(view_menu, wxT("&View"));
	menu_bar->Append(sets_menu, wxT("&Settings"));
	menu_bar->Append(help_menu, wxT("&Help"));

	// Associate the menu bar with the frame
	SetMenuBar(menu_bar);

	// the status bar
	CreateStatusBar();

	// the toolbar
	tool_bar = new wxToolBar(this, OPJFRAME_TOOLBAR,
								wxDefaultPosition, wxDefaultSize,
								wxTB_HORIZONTAL | wxNO_BORDER);
	wxBitmap bmpOpen = wxArtProvider::GetBitmap(wxART_FILE_OPEN, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpSaveAs = wxArtProvider::GetBitmap(wxART_FILE_SAVE_AS, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpZoom = wxArtProvider::GetBitmap(wxART_FIND, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpFit = wxArtProvider::GetBitmap(wxART_FIND_AND_REPLACE, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpReload = wxArtProvider::GetBitmap(wxART_EXECUTABLE_FILE, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpDecosettings = wxArtProvider::GetBitmap(wxART_REPORT_VIEW, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpEncosettings = wxArtProvider::GetBitmap(wxART_LIST_VIEW, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpPrevframe = wxArtProvider::GetBitmap(wxART_GO_BACK, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpHomeframe = wxArtProvider::GetBitmap(wxART_GO_HOME, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpNextframe = wxArtProvider::GetBitmap(wxART_GO_FORWARD, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpLesslayers = bmpPrevframe;
	wxBitmap bmpAlllayers = wxArtProvider::GetBitmap(wxART_GO_TO_PARENT, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpMorelayers = bmpNextframe;
	wxBitmap bmpLessres = bmpPrevframe;
	wxBitmap bmpFullres = wxArtProvider::GetBitmap(wxART_GO_TO_PARENT, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpMoreres = bmpNextframe;
	wxBitmap bmpPrevcomp = bmpPrevframe;
	wxBitmap bmpAllcomps = wxArtProvider::GetBitmap(wxART_GO_TO_PARENT, wxART_TOOLBAR,
												wxDefaultSize);
	wxBitmap bmpNextcomp = bmpNextframe;

	tool_bar->AddTool(OPJFRAME_FILEOPEN, bmpOpen, wxT("Open"));
	tool_bar->AddTool(OPJFRAME_FILESAVEAS, bmpSaveAs, wxT("Save as "));
	//tool_bar->EnableTool(OPJFRAME_FILESAVEAS, false);
	tool_bar->AddSeparator();
	tool_bar->AddTool(OPJFRAME_VIEWZOOM, bmpZoom, wxT("Zoom"));
	tool_bar->AddTool(OPJFRAME_VIEWFIT, bmpFit, wxT("Zoom to fit"));
	tool_bar->AddTool(OPJFRAME_VIEWRELOAD, bmpReload, wxT("Reload"));
	tool_bar->AddSeparator();
	tool_bar->AddTool(OPJFRAME_SETSDECO, bmpDecosettings, wxT("Decoder settings"));
	tool_bar->AddTool(OPJFRAME_SETSENCO, bmpEncosettings, wxT("Encoder settings"));
	tool_bar->AddSeparator();
	tool_bar->AddTool(OPJFRAME_VIEWPREVFRAME, bmpPrevframe, wxT("Previous frame"));
	tool_bar->AddTool(OPJFRAME_VIEWHOMEFRAME, bmpHomeframe, wxT("Starting frame"));
	tool_bar->AddTool(OPJFRAME_VIEWNEXTFRAME, bmpNextframe, wxT("Next frame"));
	tool_bar->AddSeparator();
	tool_bar->AddTool(OPJFRAME_VIEWLESSLAYERS, bmpLesslayers, wxT("Remove a layer"));
	tool_bar->AddTool(OPJFRAME_VIEWALLLAYERS, bmpAlllayers, wxT("Show all layers"));
	tool_bar->AddTool(OPJFRAME_VIEWMORELAYERS, bmpMorelayers, wxT("Add a layer"));
	tool_bar->AddSeparator();
	tool_bar->AddTool(OPJFRAME_VIEWLESSRES, bmpLessres, wxT("Reduce the resolution"));
	tool_bar->AddTool(OPJFRAME_VIEWFULLRES, bmpFullres, wxT("Full resolution"));
	tool_bar->AddTool(OPJFRAME_VIEWMORERES, bmpMoreres, wxT("Increase the resolution"));
	tool_bar->AddSeparator();
	tool_bar->AddTool(OPJFRAME_VIEWPREVCOMP, bmpPrevcomp, wxT("Previous component"));
	tool_bar->AddTool(OPJFRAME_VIEWALLCOMPS, bmpAllcomps, wxT("All components"));
	tool_bar->AddTool(OPJFRAME_VIEWNEXTCOMP, bmpNextcomp, wxT("Next component"));
	tool_bar->Realize();
	
	// associate the toolbar with the frame
	SetToolBar(tool_bar);

	// show the toolbar?
	if (!wxGetApp().m_showtoolbar)
		tool_bar->Show(false);
	else
		tool_bar->Show(true);

	// the logging window
	loggingWindow = new wxSashLayoutWindow(this, OPJFRAME_LOGWIN,
											wxDefaultPosition, wxSize(400, wxGetApp().m_peekerheight),
											wxNO_BORDER | wxSW_3D | wxCLIP_CHILDREN
											);
	loggingWindow->SetDefaultSize(wxSize(1000, wxGetApp().m_peekerheight));
	loggingWindow->SetOrientation(wxLAYOUT_HORIZONTAL);
	loggingWindow->SetAlignment(wxLAYOUT_BOTTOM);
	//loggingWindow->SetBackgroundColour(wxColour(0, 0, 255));
	loggingWindow->SetSashVisible(wxSASH_TOP, true);

	// show the logging?
	if (!wxGetApp().m_showpeeker)
		loggingWindow->Show(false);
	else
		loggingWindow->Show(true);

	// create the bottom notebook
	m_bookCtrlbottom = new wxNotebook(loggingWindow, BOTTOM_NOTEBOOK_ID,
								wxDefaultPosition, wxDefaultSize,
								wxBK_LEFT);

	// create the text control of the logger
	m_textCtrl = new wxTextCtrl(m_bookCtrlbottom, wxID_ANY, wxT(""),
								wxDefaultPosition, wxDefaultSize,
								wxTE_MULTILINE | wxSUNKEN_BORDER | wxTE_READONLY
								);
	m_textCtrl->SetValue(_T("Logging window\n"));

	// add it to the notebook
	m_bookCtrlbottom->AddPage(m_textCtrl, wxT("Log"));

	// create the text control of the browser
	m_textCtrlbrowse = new wxTextCtrl(m_bookCtrlbottom, wxID_ANY, wxT(""),
								wxDefaultPosition, wxDefaultSize,
								wxTE_MULTILINE | wxSUNKEN_BORDER | wxTE_READONLY | wxTE_RICH
								);
	wxFont *browsefont = new wxFont(wxNORMAL_FONT->GetPointSize(),
		wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    m_textCtrlbrowse->SetDefaultStyle(wxTextAttr(wxNullColour, wxNullColour, *browsefont));
	m_textCtrlbrowse->AppendText(wxT("Browsing window\n"));

	// add it the notebook
	m_bookCtrlbottom->AddPage(m_textCtrlbrowse, wxT("Peek"), false);

	// the browser window
	markerTreeWindow = new wxSashLayoutWindow(this, OPJFRAME_BROWSEWIN,
											  wxDefaultPosition, wxSize(wxGetApp().m_browserwidth, 30),
											  wxNO_BORDER | wxSW_3D | wxCLIP_CHILDREN
											  );
	markerTreeWindow->SetDefaultSize(wxSize(wxGetApp().m_browserwidth, 1000));
	markerTreeWindow->SetOrientation(wxLAYOUT_VERTICAL);
	markerTreeWindow->SetAlignment(wxLAYOUT_LEFT);
	//markerTreeWindow->SetBackgroundColour(wxColour(0, 255, 0));
	markerTreeWindow->SetSashVisible(wxSASH_RIGHT, true);
	markerTreeWindow->SetExtraBorderSize(0);

	// create the browser notebook
	m_bookCtrl = new wxNotebook(markerTreeWindow, LEFT_NOTEBOOK_ID,
								wxDefaultPosition, wxDefaultSize,
								wxBK_TOP);

	// show the browser?
	if (!wxGetApp().m_showbrowser)
		markerTreeWindow->Show(false);
	else
		markerTreeWindow->Show(true);

#ifdef __WXMOTIF__
	// For some reason, we get a memcpy crash in wxLogStream::DoLogStream
	// on gcc/wxMotif, if we use wxLogTextCtl. Maybe it's just gcc?
	delete wxLog::SetActiveTarget(new wxLogStderr);
#else
	// set our text control as the log target
	wxLogTextCtrl *logWindow = new wxLogTextCtrl(m_textCtrl);
	delete wxLog::SetActiveTarget(logWindow);
#endif

	// associate drop targets with the controls
	SetDropTarget(new OPJDnDFile(this));

}

// this is the frame destructor
OPJFrame::~OPJFrame(void)
{
	// save size settings
	GetSize(&(wxGetApp().m_framewidth), &(wxGetApp().m_frameheight));

	// delete all possible things
	delete m_bookCtrl;
	m_bookCtrl = NULL;

	delete markerTreeWindow;
	markerTreeWindow = NULL;

	delete m_textCtrl;
	m_textCtrl = NULL;

	delete m_bookCtrlbottom;
	m_bookCtrlbottom = NULL;

	delete loggingWindow;
	loggingWindow = NULL;
}

void OPJFrame::OnNotebook(wxNotebookEvent& event)
{
	int sel = event.GetSelection();
	long childnum;

	m_bookCtrl->GetPageText(sel).ToLong(&childnum);

	if (m_childhash[childnum])
		m_childhash[childnum]->Activate();

	//wxLogMessage(wxT("Selection changed (now %d --> %d)"), childnum, m_childhash[childnum]->m_winnumber);

}


void OPJFrame::Resize(int number)
{
	wxSize size = GetClientSize();
}

void OPJFrame::OnSetsEnco(wxCommandEvent& event)
{
	int n;

    OPJEncoderDialog dialog(this, event.GetId());

    if (dialog.ShowModal() == wxID_OK) {

		// load settings
		wxGetApp().m_subsampling = dialog.m_subsamplingCtrl->GetValue();
		wxGetApp().m_origin = dialog.m_originCtrl->GetValue();
		wxGetApp().m_rates = dialog.m_rateCtrl->GetValue();
		wxGetApp().m_quality = dialog.m_qualityCtrl->GetValue();
		wxGetApp().m_enablequality = dialog.m_qualityRadio->GetValue();
		wxGetApp().m_multicomp = dialog.m_mctCheck->GetValue();
		wxGetApp().m_irreversible = dialog.m_irrevCheck->GetValue();
		wxGetApp().m_resolutions = dialog.m_resolutionsCtrl->GetValue();
		wxGetApp().m_cbsize = dialog.m_cbsizeCtrl->GetValue();
		wxGetApp().m_prsize = dialog.m_prsizeCtrl->GetValue();
		wxGetApp().m_tsize = dialog.m_tsizeCtrl->GetValue();
		wxGetApp().m_torigin = dialog.m_toriginCtrl->GetValue();
		wxGetApp().m_progression = dialog.progressionBox->GetSelection();
		wxGetApp().m_enablesop = dialog.m_sopCheck->GetValue();
		wxGetApp().m_enableeph = dialog.m_ephCheck->GetValue();
		wxGetApp().m_enablebypass = dialog.m_enablebypassCheck->GetValue();
		wxGetApp().m_enablereset = dialog.m_enableresetCheck->GetValue();
		wxGetApp().m_enablerestart = dialog.m_enablerestartCheck->GetValue();
		wxGetApp().m_enablevsc = dialog.m_enablevscCheck->GetValue();
		wxGetApp().m_enableerterm = dialog.m_enableertermCheck->GetValue();
		wxGetApp().m_enablesegmark = dialog.m_enablesegmarkCheck->GetValue();
		wxGetApp().m_enableidx = dialog.m_enableidxCheck->GetValue();
		wxGetApp().m_index = dialog.m_indexCtrl->GetValue();
		wxGetApp().m_enablecomm = dialog.m_enablecommCheck->GetValue();
		wxGetApp().m_comment = dialog.m_commentCtrl->GetValue();
		wxGetApp().m_enablepoc = dialog.m_enablepocCheck->GetValue();
		wxGetApp().m_poc = dialog.m_pocCtrl->GetValue();
#ifdef USE_JPWL
		wxGetApp().m_enablejpwle = dialog.m_enablejpwlCheck->GetValue();
		for (n = 0; n < MYJPWL_MAX_NO_TILESPECS; n++) {
			wxGetApp().m_hprotsel[n] = dialog.m_hprotChoice[n]->GetSelection();
			wxGetApp().m_htileval[n] = dialog.m_htileCtrl[n]->GetValue();
			wxGetApp().m_pprotsel[n] = dialog.m_pprotChoice[n]->GetSelection();
			wxGetApp().m_ptileval[n] = dialog.m_ptileCtrl[n]->GetValue();
			wxGetApp().m_ppackval[n] = dialog.m_ppackCtrl[n]->GetValue();
			wxGetApp().m_sensisel[n] = dialog.m_sensiChoice[n]->GetSelection();
			wxGetApp().m_stileval[n] = dialog.m_stileCtrl[n]->GetValue();
		}
#endif // USE_JPWL
	};
}

void OPJFrame::OnSetsDeco(wxCommandEvent& event)
{
    OPJDecoderDialog dialog(this, event.GetId());

    if (dialog.ShowModal() == wxID_OK) {

		// load settings
		wxGetApp().m_enabledeco = dialog.m_enabledecoCheck->GetValue();
		wxGetApp().m_enableparse = dialog.m_enableparseCheck->GetValue();
		wxGetApp().m_resizemethod = dialog.m_resizeBox->GetSelection() - 1;
		wxGetApp().m_reducefactor = dialog.m_reduceCtrl->GetValue();
		wxGetApp().m_qualitylayers = dialog.m_layerCtrl->GetValue();
		wxGetApp().m_components = dialog.m_numcompsCtrl->GetValue();
		wxGetApp().m_framenum = dialog.m_framenumCtrl->GetValue();
#ifdef USE_JPWL
		wxGetApp().m_enablejpwl = dialog.m_enablejpwlCheck->GetValue();
		wxGetApp().m_expcomps = dialog.m_expcompsCtrl->GetValue();
		wxGetApp().m_maxtiles = dialog.m_maxtilesCtrl->GetValue();
#endif // USE_JPWL

	};
}

void OPJFrame::OnQuit(wxCommandEvent& WXUNUSED(event))
{
	Close(true);
}

void OPJFrame::OnClose(wxCommandEvent& WXUNUSED(event))
{
	// current frame
	OPJChildFrame *currframe = (OPJChildFrame *) GetActiveChild();

	if (!currframe)
		return;

	wxCloseEvent e;
	currframe->OnClose(e);
}

void OPJFrame::OnFit(wxCommandEvent& event)
{
	OPJChildFrame *currchild;
	wxString eventstring = event.GetString();

	//wxLogMessage(wxT("OnFit:%d:%s"), event.GetInt(), eventstring);

	// current child
	if (event.GetInt() >= 1) {
		currchild = m_childhash[event.GetInt()];
	} else {
		currchild = (OPJChildFrame *) GetActiveChild();
	}

	// problems
	if (!currchild)
		return;

	// current canvas
	OPJCanvas *currcanvas = currchild->m_canvas;

	// find a fit-to-width zoom
	/*int zooml, wzooml, hzooml;
	wxSize clientsize = currcanvas->GetClientSize();
	wzooml = (int) ceil(100.0 * (double) (clientsize.GetWidth() - 2 * OPJ_CANVAS_BORDER) / (double) (currcanvas->m_image100.GetWidth()));
	hzooml = (int) ceil(100.0 * (double) (clientsize.GetHeight() - 2 * OPJ_CANVAS_BORDER) / (double) (currcanvas->m_image100.GetHeight()));
	zooml = wxMin(100, wxMin(wzooml, hzooml));*/

	// fit to width
	Rescale(-1, currchild);
}

void OPJFrame::OnZoom(wxCommandEvent& WXUNUSED(event))
{
	// current frame
	OPJChildFrame *currframe = (OPJChildFrame *) GetActiveChild();

	if (!currframe)
		return;

	// get the preferred zoom
	long zooml = wxGetNumberFromUser(wxT("Choose a scale between 5% and 300%"),
		wxT("Zoom (%)"),
		wxT("Image scale"),
		currframe->m_canvas->m_zooml, 5, 300, NULL, wxDefaultPosition);

	// rescale current frame image if necessary
	if (zooml >= 5) {
		Rescale(zooml, currframe);
		wxLogMessage(wxT("zoom to %d%%"), zooml);
	}
}

void OPJFrame::Rescale(int zooml, OPJChildFrame *currframe)
{
	wxImage new_image = currframe->m_canvas->m_image100.ConvertToImage();

	// resizing enabled?
	if (wxGetApp().m_resizemethod == -1) {

		zooml = 100;

	} else {

		if (zooml < 0) {
			// find a fit-to-width zoom
			int wzooml, hzooml;
			//wxSize clientsize = currframe->m_canvas->GetClientSize();
			wxSize clientsize = currframe->m_frame->GetActiveChild()->GetClientSize();
			wzooml = (int) floor(100.0 * (double) clientsize.GetWidth() / (double) (2 * OPJ_CANVAS_BORDER + currframe->m_canvas->m_image100.GetWidth()));
			hzooml = (int) floor(100.0 * (double) clientsize.GetHeight() / (double) (2 * OPJ_CANVAS_BORDER + currframe->m_canvas->m_image100.GetHeight()));
			zooml = wxMin(100, wxMin(wzooml, hzooml));
		}
	}

	if (zooml != 100)
		new_image.Rescale((int) ((double) zooml * (double) new_image.GetWidth() / 100.0),
			(int) ((double) zooml * (double) new_image.GetHeight() / 100.0),
			wxGetApp().m_resizemethod ? wxIMAGE_QUALITY_HIGH : wxIMAGE_QUALITY_NORMAL);
	currframe->m_canvas->m_image = wxBitmap(new_image);
	currframe->m_canvas->SetScrollbars(20,
										20,
										(int)(0.5 + (double) new_image.GetWidth() / 20.0),
										(int)(0.5 + (double) new_image.GetHeight() / 20.0)
										);

	currframe->m_canvas->Refresh();

	wxLogMessage(wxT("Rescale said %d%%"), zooml);

	// update zoom
	currframe->m_canvas->m_zooml = zooml;
}


void OPJFrame::OnReload(wxCommandEvent& event)
{
	OPJChildFrame *currframe = (OPJChildFrame *) GetActiveChild();

	if (currframe) {
		OPJDecoThread *dthread = currframe->m_canvas->CreateDecoThread();

		if (dthread->Run() != wxTHREAD_NO_ERROR)
			wxLogMessage(wxT("Can't start deco thread!"));
		else
			wxLogMessage(wxT("New deco thread started."));

		currframe->m_canvas->Refresh();

		// update zoom
		//currframe->m_canvas->m_zooml = zooml;
	}
}

void OPJFrame::OnPrevFrame(wxCommandEvent& event)
{
	if (--wxGetApp().m_framenum < 0)
		wxGetApp().m_framenum = 0;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnHomeFrame(wxCommandEvent& event)
{
	wxGetApp().m_framenum = 0;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnNextFrame(wxCommandEvent& event)
{
	++wxGetApp().m_framenum;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnLessLayers(wxCommandEvent& event)
{
	if (--wxGetApp().m_qualitylayers < 1)
		wxGetApp().m_qualitylayers = 1;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnAllLayers(wxCommandEvent& event)
{
	wxGetApp().m_qualitylayers = 0;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnMoreLayers(wxCommandEvent& event)
{
	++wxGetApp().m_qualitylayers;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnLessRes(wxCommandEvent& event)
{
	++wxGetApp().m_reducefactor;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnFullRes(wxCommandEvent& event)
{
	wxGetApp().m_reducefactor = 0;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnMoreRes(wxCommandEvent& event)
{
	if (--wxGetApp().m_reducefactor < 0)
		wxGetApp().m_reducefactor = 0;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnPrevComp(wxCommandEvent& event)
{
	if (--wxGetApp().m_components < 1)
		wxGetApp().m_components = 1;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnAllComps(wxCommandEvent& event)
{
	wxGetApp().m_components = 0;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnNextComp(wxCommandEvent& event)
{
	++wxGetApp().m_components;

	wxCommandEvent e;
	OnReload(e);
}

void OPJFrame::OnToggleBrowser(wxCommandEvent& WXUNUSED(event))
{
    if (markerTreeWindow->IsShown())
        markerTreeWindow->Show(false);
    else
        markerTreeWindow->Show(true);

    wxLayoutAlgorithm layout;
    layout.LayoutMDIFrame(this);

	wxGetApp().m_showbrowser = markerTreeWindow->IsShown();

    // Leaves bits of itself behind sometimes
    GetClientWindow()->Refresh();
}

void OPJFrame::OnTogglePeeker(wxCommandEvent& WXUNUSED(event))
{
    if (loggingWindow->IsShown())
        loggingWindow->Show(false);
    else
        loggingWindow->Show(true);

    wxLayoutAlgorithm layout;
    layout.LayoutMDIFrame(this);

	wxGetApp().m_showpeeker = loggingWindow->IsShown();

    // Leaves bits of itself behind sometimes
    GetClientWindow()->Refresh();
}

void OPJFrame::OnToggleToolbar(wxCommandEvent& WXUNUSED(event))
{
    if (tool_bar->IsShown())
        tool_bar->Show(false);
    else
        tool_bar->Show(true);

    wxLayoutAlgorithm layout;
    layout.LayoutMDIFrame(this);

	wxGetApp().m_showtoolbar = tool_bar->IsShown();

    // Leaves bits of itself behind sometimes
    GetClientWindow()->Refresh();
}

void OPJFrame::OnSashDrag(wxSashEvent& event)
{
	int wid, hei;

    if (event.GetDragStatus() == wxSASH_STATUS_OUT_OF_RANGE)
        return;

    switch (event.GetId()) {
		case OPJFRAME_BROWSEWIN:
		{
			markerTreeWindow->SetDefaultSize(wxSize(event.GetDragRect().width, 1000));
			break;
		}
		case OPJFRAME_LOGWIN:
		{
			loggingWindow->SetDefaultSize(wxSize(1000, event.GetDragRect().height));
			break;
		}
    }

    wxLayoutAlgorithm layout;
    layout.LayoutMDIFrame(this);

    // Leaves bits of itself behind sometimes
    GetClientWindow()->Refresh();

	// update dimensions
	markerTreeWindow->GetSize(&wid, &hei);
	wxGetApp().m_browserwidth = wid;

	loggingWindow->GetSize(&wid, &hei);
	wxGetApp().m_peekerheight = hei;

}

void OPJFrame::OnThreadLogmsg(wxCommandEvent& event)
{
#if 1
    wxLogMessage(wxT("Frame got message from worker thread: %d"), event.GetInt());
    wxLogMessage(event.GetString());
#else
    int n = event.GetInt();
    if ( n == -1 )
    {
        m_dlgProgress->Destroy();
        m_dlgProgress = (wxProgressDialog *)NULL;

        // the dialog is aborted because the event came from another thread, so
        // we may need to wake up the main event loop for the dialog to be
        // really closed
        wxWakeUpIdle();
    }
    else
    {
        if ( !m_dlgProgress->Update(n) )
        {
            wxCriticalSectionLocker lock(m_critsectWork);

            m_cancelled = true;
        }
    }
#endif
}


// physically save the file
void OPJFrame::SaveFile(wxArrayString paths, wxArrayString filenames)
{
	size_t count = paths.GetCount();
	wxString msg, s;

	if (wxFile::Exists(paths[0].c_str())) {

		s.Printf(wxT("File %s already exists. Do you want to overwrite it?\n"), filenames[0].c_str());
		wxMessageDialog dialog3(this, s, _T("File exists"), wxYES_NO);
		if (dialog3.ShowModal() == wxID_NO)
			return;
	}

	/*s.Printf(_T("File %d: %s (%s)\n"), (int)0, paths[0].c_str(), filenames[0].c_str());
	msg += s;

	wxMessageDialog dialog2(this, msg, _T("Selected files"));
	dialog2.ShowModal();*/

	if (!GetActiveChild())
		return;

	((OPJChildFrame *) GetActiveChild())->m_canvas->m_savename = paths[0];

	OPJEncoThread *ethread = ((OPJChildFrame *) GetActiveChild())->m_canvas->CreateEncoThread();

    if (ethread->Run() != wxTHREAD_NO_ERROR)
        wxLogMessage(wxT("Can't start enco thread!"));
    else
		wxLogMessage(wxT("New enco thread started."));


}

// physically open the files
void OPJFrame::OpenFiles(wxArrayString paths, wxArrayString filenames)
{

	size_t count = paths.GetCount();
	for (size_t n = 0; n < count; n++) {

		wxString msg, s;
		s.Printf(_T("File %d: %s (%s)\n"), (int)n, paths[n].c_str(), filenames[n].c_str());

		msg += s;

		/*wxMessageDialog dialog2(this, msg, _T("Selected files"));
		dialog2.ShowModal();*/

		// Make another frame, containing a canvas
		OPJChildFrame *subframe = new OPJChildFrame(this,
													paths[n],
													winNumber,
													wxT("Canvas Frame"),
													wxDefaultPosition, wxSize(300, 300),
													wxDEFAULT_FRAME_STYLE | wxNO_FULL_REPAINT_ON_RESIZE
													);
		m_childhash[winNumber] = subframe;

		// create own marker tree
		m_treehash[winNumber] = new OPJMarkerTree(m_bookCtrl, subframe, paths[n], wxT("Parsing..."), TreeTest_Ctrl,
												  wxDefaultPosition, wxDefaultSize,
												  wxTR_DEFAULT_STYLE | wxSUNKEN_BORDER
												  );

		m_bookCtrl->AddPage(m_treehash[winNumber], wxString::Format(wxT("%u"), winNumber), false);

		for (unsigned int p = 0; p < m_bookCtrl->GetPageCount(); p++) {
			if (m_bookCtrl->GetPageText(p) == wxString::Format(wxT("%u"), winNumber)) {
				m_bookCtrl->ChangeSelection(p);
				break;
			}
		}

		winNumber++;
	}
}

void OPJFrame::OnFileOpen(wxCommandEvent& WXUNUSED(event))
{
    wxString wildcards =
#ifdef __WXMOTIF__
	wxT("JPEG 2000 files (*.jp2,*.j2k,*.j2c,*.mj2)|*.*j*2*");
#else
#if wxUSE_LIBOPENJPEG
	wxT("JPEG 2000 files (*.jp2,*.j2k,*.j2c,*.mj2)|*.jp2;*.j2k;*.j2c;*.mj2")
#endif
#if USE_MXF
	wxT("|MXF JPEG 2000 video (*.mxf)|*.mxf")
#endif // USE_MXF
#if wxUSE_LIBJPEG
		wxT("|JPEG files (*.jpg)|*.jpg")
#endif
#if OPJ_MANYFORMATS
		wxT("|BMP files (*.bmp)|*.bmp")
		wxT("|PNG files (*.png)|*.png")
		wxT("|GIF files (*.gif)|*.gif")
		wxT("|PNM files (*.pnm)|*.pnm")
		wxT("|TIFF files (*.tif,*.tiff)|*.tif*")
#endif
		wxT("|All files|*");
#endif
    wxFileDialog dialog(this, _T("Open image file(s)"),
                        wxEmptyString, wxEmptyString, wildcards,
                        wxFD_OPEN|wxFD_MULTIPLE);

    if (dialog.ShowModal() == wxID_OK) {
        wxArrayString paths, filenames;

        dialog.GetPaths(paths);
        dialog.GetFilenames(filenames);

		OpenFiles(paths, filenames);
    }

}

void OPJFrame::OnFileSaveAs(wxCommandEvent& WXUNUSED(event))
{
    wxString wildcards =
#ifdef wxUSE_LIBOPENJPEG
#ifdef __WXMOTIF__
	wxT("JPEG 2000 codestream (*.j2k)|*.*j*2*");
#else
	wxT("JPEG 2000 codestream (*.j2k)|*.j2k")
	wxT("|JPEG 2000 file format (*.jp2)|*.jp2");
#endif
#endif

    wxFileDialog dialog(this, _T("Save image file"),
                        wxEmptyString, wxEmptyString, wildcards,
                        wxFD_SAVE);

    if (dialog.ShowModal() == wxID_OK) {
        wxArrayString paths, filenames;

        dialog.GetPaths(paths);
        dialog.GetFilenames(filenames);

		SaveFile(paths, filenames);
    }


}

void OPJFrame::OnMemoryOpen(wxCommandEvent& WXUNUSED(event))
{
	// do nothing
	return;
	
	wxTextEntryDialog dialog(this, wxT("Memory HEX address range: start_address-stop_address"),
							wxT("Decode a memory buffer"),
							wxT("0x-0x"),
							wxOK | wxCANCEL | wxCENTRE,
							wxDefaultPosition);

	if (dialog.ShowModal() == wxID_OK) {

	}

}

BEGIN_EVENT_TABLE(OPJCanvas, wxScrolledWindow)
    EVT_MOUSE_EVENTS(OPJCanvas::OnEvent)
    EVT_MENU(OPJCANVAS_THREADSIGNAL, OPJCanvas::OnThreadSignal)
END_EVENT_TABLE()

// Define a constructor for my canvas
OPJCanvas::OPJCanvas(wxFileName fname, wxWindow *parent, const wxPoint& pos, const wxSize& size)
        : wxScrolledWindow(parent, wxID_ANY, pos, size,
                           wxSUNKEN_BORDER | wxNO_FULL_REPAINT_ON_RESIZE)
{
    SetBackgroundColour(OPJ_CANVAS_COLOUR);

	m_fname = fname;
	m_childframe = (OPJChildFrame *) parent;
	// 100% zoom
	m_zooml = 100;


    OPJDecoThread *dthread = CreateDecoThread();

    if (dthread->Run() != wxTHREAD_NO_ERROR)
        wxLogMessage(wxT("Can't start deco thread!"));
    else
		wxLogMessage(wxT("New deco thread started."));

	// 100% zoom
	//m_zooml = 100;

}

OPJDecoThread *OPJCanvas::CreateDecoThread(void)
{
    OPJDecoThread *dthread = new OPJDecoThread(this);

    if (dthread->Create() != wxTHREAD_NO_ERROR)
		wxLogError(wxT("Can't create deco thread!"));

    wxCriticalSectionLocker enter(wxGetApp().m_deco_critsect);
    wxGetApp().m_deco_threads.Add(dthread);

    return dthread;
}

OPJEncoThread *OPJCanvas::CreateEncoThread(void)
{
    OPJEncoThread *ethread = new OPJEncoThread(this);

    if (ethread->Create() != wxTHREAD_NO_ERROR)
		wxLogError(wxT("Can't create enco thread!"));

    wxCriticalSectionLocker enter(wxGetApp().m_enco_critsect);
    wxGetApp().m_enco_threads.Add(ethread);

    return ethread;
}

#define activeoverlay 0
// Define the repainting behaviour
void OPJCanvas::OnDraw(wxDC& dc)
{
	if (m_image.Ok()) {
		dc.DrawBitmap(m_image, OPJ_CANVAS_BORDER, OPJ_CANVAS_BORDER);

		if (activeoverlay) {
			dc.SetPen(*wxRED_PEN);
			dc.SetBrush(*wxTRANSPARENT_BRUSH);
			//int tw, th;
			dc.DrawRectangle(OPJ_CANVAS_BORDER, OPJ_CANVAS_BORDER,
				(unsigned long int) (0.5 + (double) m_zooml * (double) m_childframe->m_twidth / 100.0),
				(unsigned long int) (0.5 + (double) m_zooml * (double) m_childframe->m_theight / 100.0));
		}

	} else {
		dc.SetFont(*wxSWISS_FONT);
		dc.SetPen(*wxBLACK_PEN);
#ifdef __WXGTK__
		dc.DrawText(_T("Decoding image, please wait... (press \"Zoom to Fit\" to show the image)"), 40, 50);
#else
		dc.DrawText(_T("Decoding image, please wait..."), 40, 50);
#endif
	}
}

// This implements a tiny doodling program! Drag the mouse using
// the left button.
void OPJCanvas::OnEvent(wxMouseEvent& event)
{
#if USE_PENCIL_ON_CANVAS
  wxClientDC dc(this);
  PrepareDC(dc);

  wxPoint pt(event.GetLogicalPosition(dc));

  if ((xpos > -1) && (ypos > -1) && event.Dragging()) {
    dc.SetPen(*wxRED_PEN);
    dc.DrawLine(xpos, ypos, pt.x, pt.y);
  }
  xpos = pt.x;
  ypos = pt.y;
#endif
}

void OPJFrame::OnSize(wxSizeEvent& WXUNUSED(event))
{
    wxLayoutAlgorithm layout;
    layout.LayoutMDIFrame(this);
}

void OPJCanvas::OnThreadSignal(wxCommandEvent& event)
{
#if 1
    wxLogMessage(wxT("Canvas got signal from deco thread: %d"), event.GetInt());
    wxLogMessage(event.GetString());
#else
    int n = event.GetInt();
    if ( n == -1 )
    {
        m_dlgProgress->Destroy();
        m_dlgProgress = (wxProgressDialog *)NULL;

        // the dialog is aborted because the event came from another thread, so
        // we may need to wake up the main event loop for the dialog to be
        // really closed
        wxWakeUpIdle();
    }
    else
    {
        if ( !m_dlgProgress->Update(n) )
        {
            wxCriticalSectionLocker lock(m_critsectWork);

            m_cancelled = true;
        }
    }
#endif
}


// Note that OPJFRAME_FILEOPEN and OPJFRAME_HELPABOUT commands get passed
// to the parent window for processing, so no need to
// duplicate event handlers here.

BEGIN_EVENT_TABLE(OPJChildFrame, wxMDIChildFrame)
  /*EVT_MENU(SASHTEST_CHILD_QUIT, OPJChildFrame::OnQuit)*/
  EVT_CLOSE(OPJChildFrame::OnClose)
  EVT_SET_FOCUS(OPJChildFrame::OnGotFocus)
  EVT_KILL_FOCUS(OPJChildFrame::OnLostFocus)
END_EVENT_TABLE()

OPJChildFrame::OPJChildFrame(OPJFrame *parent, wxFileName fname, int winnumber, const wxString& title, const wxPoint& pos, const wxSize& size,
const long style):
  wxMDIChildFrame(parent, wxID_ANY, title, pos, size, style)
{
	m_frame = (OPJFrame  *) parent;
	m_canvas = NULL;
	//my_children.Append(this);
	m_fname = fname;
	m_winnumber = winnumber;
	SetTitle(wxString::Format(_T("%d: "), m_winnumber) + m_fname.GetFullName());

	  // Give it an icon (this is ignored in MDI mode: uses resources)
#ifdef __WXMSW__
	SetIcon(wxIcon(wxT("OPJChild16")));
#endif

	// Give it a status line
	/*CreateStatusBar();*/

	int width, height;
	GetClientSize(&width, &height);

	OPJCanvas *canvas = new OPJCanvas(fname, this, wxPoint(0, 0), wxSize(width, height));
#if USE_PENCIL_ON_CANVAS
	canvas->SetCursor(wxCursor(wxCURSOR_PENCIL));
#endif
	m_canvas = canvas;

	// Give it scrollbars
	canvas->SetScrollbars(20, 20, 5, 5);

	Show(true);
	Maximize(true);

	/*wxLogError(wxString::Format(wxT("Created tree %d (0x%x)"), m_winnumber, m_frame->m_treehash[m_winnumber]));*/

}

OPJChildFrame::~OPJChildFrame(void)
{
  //my_children.DeleteObject(this);
}


void OPJChildFrame::OnClose(wxCloseEvent& event)
{
	for (unsigned int p = 0; p < m_frame->m_bookCtrl->GetPageCount(); p++) {
		if (m_frame->m_bookCtrl->GetPageText(p) == wxString::Format(wxT("%u"), m_winnumber)) {
			m_frame->m_bookCtrl->DeletePage(p);
			break;
		}
	}
	Destroy();

	wxLogMessage(wxT("Closed: %d"), m_winnumber);
}

void OPJChildFrame::OnActivate(wxActivateEvent& event)
{
  /*if (event.GetActive() && m_canvas)
    m_canvas->SetFocus();*/
}

void OPJChildFrame::OnGotFocus(wxFocusEvent& event)
{
	// we need to check if the notebook is being destroyed or not
	if (!m_frame->m_bookCtrl)
		return;

	for (unsigned int p = 0; p < m_frame->m_bookCtrl->GetPageCount(); p++) {

		if (m_frame->m_bookCtrl->GetPageText(p) == wxString::Format(wxT("%u"), m_winnumber)) {
			m_frame->m_bookCtrl->ChangeSelection(p);
			break;
		}

	}

	//wxLogMessage(wxT("Got focus: %d (%x)"), m_winnumber, event.GetWindow());
}

void OPJChildFrame::OnLostFocus(wxFocusEvent& event)
{
	//wxLogMessage(wxT("Lost focus: %d (%x)"), m_winnumber, event.GetWindow());
}


////////////////////////////////
// drag and drop 
////////////////////////////////

bool OPJDnDFile::OnDropFiles(wxCoord, wxCoord, const wxArrayString& filenames)
{
    /*size_t nFiles = filenames.GetCount();
    wxString str;
    str.Printf( _T("%d files dropped\n"), (int)nFiles);
    for ( size_t n = 0; n < nFiles; n++ ) {
        str << filenames[n] << wxT("\n");
    }
    wxLogMessage(str);*/
	m_pOwner->OpenFiles(filenames, filenames);

    return true;
}

