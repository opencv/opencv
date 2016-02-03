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
// Name:        sashtest.h
// Purpose:     Layout window/sash sample
// Author:      Julian Smart
// Modified by:
// Created:     04/01/98
// RCS-ID:      $Id: sashtest.h,v 1.5 2005/06/02 12:04:24 JS Exp $
// Copyright:   (c) Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        treectrl.h
// Purpose:     wxTreeCtrl sample
// Author:      Julian Smart
// Modified by:
// Created:     04/01/98
// RCS-ID:      $Id: treetest.h,v 1.50 2006/11/04 11:26:51 VZ Exp $
// Copyright:   (c) Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Name:        dialogs.h
// Purpose:     Common dialogs demo
// Author:      Julian Smart
// Modified by: ABX (2004) - adjustementd for conditional building
// Created:     04/01/98
// RCS-ID:      $Id: dialogs.h,v 1.50 2006/10/08 14:12:59 VZ Exp $
// Copyright:   (c) Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////

#ifndef __OPJ_VIEWER_H__
#define __OPJ_VIEWER_H__

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#include "wx/mdi.h"
#endif

#include "wx/toolbar.h"
#include "wx/laywin.h"
#include "wx/treectrl.h"

#include "icon1.xpm"
#include "icon2.xpm"
#include "icon3.xpm"
#include "icon4.xpm"
#include "icon5.xpm"

#include "wx/filedlg.h"
#include "wx/toolbar.h"
#include <wx/filename.h>
#include <wx/busyinfo.h>
#include <wx/cmdline.h>
#include <wx/file.h>
#include "wx/notebook.h"
#include <wx/numdlg.h>

#include "wx/propdlg.h"
#include "wx/spinctrl.h"

#include <wx/dnd.h>
#include "wx/wxhtml.h"
#include "wx/statline.h"
#include <wx/fs_mem.h>

#include <wx/imaglist.h>

#include "wx/toolbar.h"
#include "wx/artprov.h"

#include "libopenjpeg/openjpeg.h"

//#include "imagj2k.h"
//#include "imagjp2.h"
//#include "imagmj2.h"
#include "imagjpeg2000.h"
#ifdef USE_MXF
#include "imagmxf.h"
#endif // USE_MXF

#ifdef __WXMSW__
typedef unsigned __int64 int8byte;
#endif // __WXMSW__

#ifdef __WXGTK__
typedef unsigned long long int8byte;
#endif // __WXGTK__

#define USE_GENERIC_TREECTRL 0
#define USE_PENCIL_ON_CANVAS 0

#if USE_GENERIC_TREECTRL
#include "wx/generic/treectlg.h"
#ifndef wxTreeCtrl
#define wxTreeCtrl wxGenericTreeCtrl
#define sm_classwxTreeCtrl sm_classwxGenericTreeCtrl
#endif
#endif

#define OPJ_APPLICATION             wxT("OPJViewer")
#define OPJ_APPLICATION_NAME		wxT("OpenJPEG Viewer")
#define OPJ_APPLICATION_VERSION		wxT("0.4 beta")
#define OPJ_APPLICATION_TITLEBAR	OPJ_APPLICATION_NAME wxT(" ") OPJ_APPLICATION_VERSION
#define OPJ_APPLICATION_COPYRIGHT	wxT("(C) 2007-2008, Giuseppe Baruffa")
#define OPJ_APPLICATION_VENDOR      wxT("OpenJPEG")

#ifdef __WXMSW__
#define OPJ_APPLICATION_PLATFORM    wxT("Windows")
#endif

#ifdef __WXGTK__
#define OPJ_APPLICATION_PLATFORM    wxT("Linux")
#endif

#define OPJ_FRAME_WIDTH   800
#define OPJ_FRAME_HEIGHT  600

#define OPJ_BROWSER_WIDTH 300
#define OPJ_PEEKER_HEIGHT 130

#define OPJ_CANVAS_BORDER 10
#define OPJ_CANVAS_COLOUR *wxWHITE



#ifdef USE_JPWL

//#define MYJPWL_MAX_NO_TILESPECS JPWL_MAX_NO_TILESPECS
#define MYJPWL_MAX_NO_TILESPECS 4

#endif // USE_JPWL


class OPJDecoThread;
class OPJEncoThread;
class OPJParseThread;
WX_DEFINE_ARRAY_PTR(wxThread *, wxArrayThread);
class OPJChildFrame;

//////////////////////////////////
// this is our main application //
//////////////////////////////////
class OPJViewerApp: public wxApp
{
	// public methods and variables
	public:

		// class constructor
		OPJViewerApp() { m_showImages = true; m_showButtons = false; }

		// other methods
		bool OnInit(void);
		int OnExit(void);
		void SetShowImages(bool show) { m_showImages = show; }
		bool ShowImages() const { return m_showImages; }
		void ShowCmdLine(const wxCmdLineParser& parser);

		// all the threads currently alive - as soon as the thread terminates, it's
		// removed from the array
		wxArrayThread m_deco_threads, m_parse_threads, m_enco_threads;

		// crit section protects access to all of the arrays below
		wxCriticalSection m_deco_critsect, m_parse_critsect, m_enco_critsect;

		// semaphore used to wait for the threads to exit, see OPJFrame::OnQuit()
		wxSemaphore m_deco_semAllDone, m_parse_semAllDone, m_enco_semAllDone;

		// the last exiting thread should post to m_semAllDone if this is true
		// (protected by the same m_critsect)
		bool m_deco_waitingUntilAllDone, m_parse_waitingUntilAllDone, m_enco_waitingUntilAllDone;

		// the list of all filenames written in the command line
		wxArrayString m_filelist;

		// displaying engine parameters
		int m_resizemethod;

		// decoding engine parameters
		bool m_enabledeco, m_enableparse;
		int m_reducefactor, m_qualitylayers, m_components, m_framenum;
#ifdef USE_JPWL
		bool m_enablejpwl, m_enablejpwle;
		int m_expcomps, m_maxtiles;
		int m_framewidth, m_frameheight;
#endif // USE_JPWL

		// encoding engine parameters
		wxString m_subsampling, m_origin, m_rates, m_comment, m_index, m_quality;
		wxString m_cbsize, m_prsize, m_tsize, m_torigin, m_poc;
		bool m_enablecomm, m_enableidx, m_multicomp, m_irreversible, m_enablesop, m_enableeph;
		bool m_enablebypass, m_enablereset, m_enablerestart, m_enablevsc, m_enableerterm;
		bool m_enablesegmark, m_enablepoc;
		bool m_enablequality;
		int m_resolutions, m_progression;
#ifdef USE_JPWL
		int m_hprotsel[MYJPWL_MAX_NO_TILESPECS], m_pprotsel[MYJPWL_MAX_NO_TILESPECS];
		int m_htileval[MYJPWL_MAX_NO_TILESPECS], m_ptileval[MYJPWL_MAX_NO_TILESPECS],
			m_ppackval[MYJPWL_MAX_NO_TILESPECS];
		int m_sensisel[MYJPWL_MAX_NO_TILESPECS], m_stileval[MYJPWL_MAX_NO_TILESPECS];
#endif // USE_JPWL

		// some layout settings
		bool m_showtoolbar, m_showbrowser, m_showpeeker;
		int m_browserwidth, m_peekerheight;

		// application configuration
		wxConfig *OPJconfig; 

	// private methods and variables
	private:
		bool m_showImages, m_showButtons;

};

DECLARE_APP(OPJViewerApp)

///////////////////////////////////////////
// this canvas is used to draw the image //
///////////////////////////////////////////
class OPJCanvas: public wxScrolledWindow
{
	// public methods and variables
	public:

		// class constructor
		OPJCanvas(wxFileName fname, wxWindow *parent, const wxPoint& pos, const wxSize& size);

		virtual void OnDraw(wxDC& dc);
		void OnEvent(wxMouseEvent& event);
		void WriteText(const wxString& text) {
#ifndef __WXGTK__ 
			wxMutexGuiEnter();
#endif //__WXGTK__
			wxLogMessage(text);
#ifndef __WXGTK__ 
			wxMutexGuiLeave();
#endif //__WXGTK__
		}

		void OnThreadSignal(wxCommandEvent& event);

		OPJDecoThread *CreateDecoThread(void);
		OPJEncoThread *CreateEncoThread(void);


		OPJChildFrame *m_childframe;

		wxBitmap  m_image, m_image100;
		wxFileName m_fname, m_savename;
		long m_zooml;

	DECLARE_EVENT_TABLE()
};

///////////////////////////////////////////////////
// the data associated to each tree leaf or node //
///////////////////////////////////////////////////
class OPJMarkerData : public wxTreeItemData
{
	// public methods and variables
	public:

		// class constructor
		OPJMarkerData(const wxString& desc, const wxString& fname = wxT(""), wxFileOffset start = 0, wxFileOffset length = 0) : m_desc(desc), m_filestring(fname) { m_start = start; m_length = length; }

		void ShowInfo(wxTreeCtrl *tree);
		const wxChar *GetDesc1() const { return m_desc.c_str(); }
		const wxChar *GetDesc2() const { return m_filestring.c_str(); }
		wxFileOffset m_start, m_length;
		wxString m_desc;

	// private methods and variables
	private:
		wxString m_filestring;
};


class OPJMarkerTree : public wxTreeCtrl
{
public:
    enum
    {
        TreeCtrlIcon_File,
        TreeCtrlIcon_FileSelected,
        TreeCtrlIcon_Folder,
        TreeCtrlIcon_FolderSelected,
        TreeCtrlIcon_FolderOpened
    };

    OPJMarkerTree() { };
    OPJMarkerTree(wxWindow *parent, OPJChildFrame *subframe, wxFileName fname, wxString name, const wxWindowID id,
               const wxPoint& pos, const wxSize& size,
               long style);
    virtual ~OPJMarkerTree(){};
	OPJParseThread *CreateParseThread(wxTreeItemId parentid = 0x00, OPJChildFrame *subframe = NULL);
    void WriteText(const wxString& text) { wxMutexGuiEnter(); wxLogMessage(text); wxMutexGuiLeave(); }

	wxFileName m_fname;
	wxTextCtrl *m_peektextCtrl;
	OPJChildFrame *m_childframe;

    /*void OnBeginDrag(wxTreeEvent& event);
    void OnBeginRDrag(wxTreeEvent& event);
    void OnEndDrag(wxTreeEvent& event);*/
    /*void OnBeginLabelEdit(wxTreeEvent& event);
    void OnEndLabelEdit(wxTreeEvent& event);*/
    /*void OnDeleteItem(wxTreeEvent& event);*/
    /*void OnContextMenu(wxContextMenuEvent& event);*/
    void OnItemMenu(wxTreeEvent& event);
    /*void OnGetInfo(wxTreeEvent& event);
    void OnSetInfo(wxTreeEvent& event);*/
    /*void OnItemExpanded(wxTreeEvent& event);*/
    void OnItemExpanding(wxTreeEvent& event);
    /*void OnItemCollapsed(wxTreeEvent& event);
    void OnItemCollapsing(wxTreeEvent& event);*/
    void OnSelChanged(wxTreeEvent& event);
    /*void OnSelChanging(wxTreeEvent& event);*/
    /*void OnTreeKeyDown(wxTreeEvent& event);*/
    /*void OnItemActivated(wxTreeEvent& event);*/
    /*void OnItemRClick(wxTreeEvent& event);*/
    /*void OnRMouseDown(wxMouseEvent& event);
    void OnRMouseUp(wxMouseEvent& event);
    void OnRMouseDClick(wxMouseEvent& event);*/
    /*void GetItemsRecursively(const wxTreeItemId& idParent,
                             wxTreeItemIdValue cookie = 0);*/

    void CreateImageList(int size = 16);
    void CreateButtonsImageList(int size = 11);

    /*void AddTestItemsToTree(size_t numChildren, size_t depth);*/
    /*void DoSortChildren(const wxTreeItemId& item, bool reverse = false)
        { m_reverseSort = reverse; wxTreeCtrl::SortChildren(item); }*/
    /*void DoEnsureVisible() { if (m_lastItem.IsOk()) EnsureVisible(m_lastItem); }*/
    /*void DoToggleIcon(const wxTreeItemId& item);*/
    /*void ShowMenu(wxTreeItemId id, const wxPoint& pt);*/

    int ImageSize(void) const { return m_imageSize; }

    void SetLastItem(wxTreeItemId id) { m_lastItem = id; }

protected:
    /*virtual int OnCompareItems(const wxTreeItemId& i1, const wxTreeItemId& i2);*/

    // is this the test item which we use in several event handlers?
    /*bool IsTestItem(const wxTreeItemId& item)
    {
        // the test item is the first child folder
        return GetItemParent(item) == GetRootItem() && !GetPrevSibling(item);
    }*/

private:
    /*void AddItemsRecursively(const wxTreeItemId& idParent,
                             size_t nChildren,
                             size_t depth,
                             size_t folder);*/

    void LogEvent(const wxChar *name, const wxTreeEvent& event);

    int          m_imageSize;               // current size of images
    bool         m_reverseSort;             // flag for OnCompareItems
    wxTreeItemId m_lastItem,                // for OnEnsureVisible()
                 m_draggedItem;             // item being dragged right now

    // NB: due to an ugly wxMSW hack you _must_ use DECLARE_DYNAMIC_CLASS()
    //     if you want your overloaded OnCompareItems() to be called.
    //     OTOH, if you don't want it you may omit the next line - this will
    //     make default (alphabetical) sorting much faster under wxMSW.
    DECLARE_DYNAMIC_CLASS(OPJMarkerTree)
    DECLARE_EVENT_TABLE()
};

// this hash map stores all the trees of currently opened images, with an integer key
WX_DECLARE_HASH_MAP(int, OPJMarkerTree*, wxIntegerHash, wxIntegerEqual, OPJMarkerTreeHash);

// this hash map stores all the children of currently opened images, with an integer key
WX_DECLARE_HASH_MAP(int, OPJChildFrame*, wxIntegerHash, wxIntegerEqual, OPJChildFrameHash);

// Define a new frame
class OPJFrame: public wxMDIParentFrame
{
  public:

    OPJFrame(wxWindow *parent, const wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, const long style);

    ~OPJFrame(void);
	void OnSize(wxSizeEvent& WXUNUSED(event));
    void OnAbout(wxCommandEvent& WXUNUSED(event));
    void OnFileOpen(wxCommandEvent& WXUNUSED(event));
    void OnFileSaveAs(wxCommandEvent& WXUNUSED(event));
    void OnMemoryOpen(wxCommandEvent& WXUNUSED(event));
    void OnQuit(wxCommandEvent& WXUNUSED(event));
    void OnClose(wxCommandEvent& WXUNUSED(event));
    void OnZoom(wxCommandEvent& WXUNUSED(event));
	void OnFit(wxCommandEvent& event);
	void OnToggleBrowser(wxCommandEvent& WXUNUSED(event));
	void OnTogglePeeker(wxCommandEvent& WXUNUSED(event));
	void OnToggleToolbar(wxCommandEvent& WXUNUSED(event));
	void OnReload(wxCommandEvent& event);
	void OnPrevFrame(wxCommandEvent& event);
	void OnHomeFrame(wxCommandEvent& event);
	void OnNextFrame(wxCommandEvent& event);
	void OnLessLayers(wxCommandEvent& event);
	void OnAllLayers(wxCommandEvent& event);
	void OnMoreLayers(wxCommandEvent& event);
	void OnLessRes(wxCommandEvent& event);
	void OnFullRes(wxCommandEvent& event);
	void OnMoreRes(wxCommandEvent& event);
	void OnPrevComp(wxCommandEvent& event);
	void OnAllComps(wxCommandEvent& event);
	void OnNextComp(wxCommandEvent& event);
	void OnSetsEnco(wxCommandEvent& event);
	void OnSetsDeco(wxCommandEvent& event);
	void OnSashDrag(wxSashEvent& event);
	void OpenFiles(wxArrayString paths, wxArrayString filenames);
	void SaveFile(wxArrayString paths, wxArrayString filenames);
	void OnNotebook(wxNotebookEvent& event);
	void Rescale(int scale, OPJChildFrame *child);
	void OnThreadLogmsg(wxCommandEvent& event);

	OPJMarkerTreeHash m_treehash;
	OPJChildFrameHash m_childhash;
    wxSashLayoutWindow* markerTreeWindow;
    wxSashLayoutWindow* loggingWindow;
	wxToolBar* tool_bar;
    void Resize(int number);
	wxNotebook *m_bookCtrl;
	wxNotebook *m_bookCtrlbottom;
    wxTextCtrl *m_textCtrlbrowse;

  private:
    void TogStyle(int id, long flag);

    void DoSort(bool reverse = false);

    wxPanel *m_panel;
    wxTextCtrl *m_textCtrl;

    void DoSetBold(bool bold = true);

protected:
    wxSashLayoutWindow* m_topWindow;
    wxSashLayoutWindow* m_leftWindow2;

DECLARE_EVENT_TABLE()
};

class OPJChildFrame: public wxMDIChildFrame
{
  public:
    OPJCanvas *m_canvas;
    OPJChildFrame(OPJFrame *parent, wxFileName fname, int winnumber, const wxString& title, const wxPoint& pos, const wxSize& size, const long style);
    ~OPJChildFrame(void);
    void OnActivate(wxActivateEvent& event);
	/*void OnQuit(wxCommandEvent& WXUNUSED(event));*/
	void OnClose(wxCloseEvent& event);
	void OnGotFocus(wxFocusEvent& event);
	void OnLostFocus(wxFocusEvent& event);
    OPJFrame *m_frame;
	wxFileName m_fname;
	int m_winnumber;

	unsigned long  m_twidth, m_theight, m_tx, m_ty;

	DECLARE_EVENT_TABLE()
};

// frame and main menu ids
enum {
	OPJFRAME_FILEEXIT = wxID_EXIT,
	OPJFRAME_HELPABOUT = wxID_ABOUT,
	OPJFRAME_FILEOPEN,
	OPJFRAME_MEMORYOPEN,
	OPJFRAME_FILESAVEAS,
	OPJFRAME_FILETOGGLEB,
	OPJFRAME_FILETOGGLEP,
	OPJFRAME_FILETOGGLET,
	OPJFRAME_VIEWZOOM,
	OPJFRAME_VIEWFIT,
	OPJFRAME_VIEWRELOAD,
	OPJFRAME_VIEWPREVFRAME,
	OPJFRAME_VIEWHOMEFRAME,
	OPJFRAME_VIEWNEXTFRAME,
	OPJFRAME_VIEWLESSLAYERS,
	OPJFRAME_VIEWALLLAYERS,
	OPJFRAME_VIEWMORELAYERS,
	OPJFRAME_VIEWLESSRES,
	OPJFRAME_VIEWFULLRES,
	OPJFRAME_VIEWMORERES,
	OPJFRAME_VIEWPREVCOMP,
	OPJFRAME_VIEWALLCOMPS,
	OPJFRAME_VIEWNEXTCOMP,
	OPJFRAME_FILECLOSE,
	OPJFRAME_SETSENCO,
	OPJFRAME_SETSDECO,

	OPJFRAME_BROWSEWIN = 10000,
	OPJFRAME_LOGWIN,
	OPJFRAME_TOOLBAR,

	OPJFRAME_THREADLOGMSG,
	OPJCANVAS_THREADSIGNAL
};


// menu and control ids
enum
{
    TreeTest_Quit = wxID_EXIT,
    TreeTest_About = wxID_ABOUT,
    TreeTest_TogButtons = wxID_HIGHEST,
    TreeTest_TogTwist,
    TreeTest_TogLines,
    TreeTest_TogEdit,
    TreeTest_TogHideRoot,
    TreeTest_TogRootLines,
    TreeTest_TogBorder,
    TreeTest_TogFullHighlight,
    TreeTest_SetFgColour,
    TreeTest_SetBgColour,
    TreeTest_ResetStyle,
    TreeTest_Highlight,
    TreeTest_Dump,
    TreeTest_DumpSelected,
    TreeTest_Count,
    TreeTest_CountRec,
    TreeTest_Sort,
    TreeTest_SortRev,
    TreeTest_SetBold,
    TreeTest_ClearBold,
    TreeTest_Rename,
    TreeTest_Delete,
    TreeTest_DeleteChildren,
    TreeTest_DeleteAll,
    TreeTest_Recreate,
    TreeTest_ToggleImages,
    TreeTest_ToggleButtons,
    TreeTest_SetImageSize,
    TreeTest_ToggleSel,
    TreeTest_CollapseAndReset,
    TreeTest_EnsureVisible,
    TreeTest_AddItem,
    TreeTest_InsertItem,
    TreeTest_IncIndent,
    TreeTest_DecIndent,
    TreeTest_IncSpacing,
    TreeTest_DecSpacing,
    TreeTest_ToggleIcon,
    TreeTest_Select,
    TreeTest_Unselect,
    TreeTest_SelectRoot,
    TreeTest_Ctrl = 1000,
	BOTTOM_NOTEBOOK_ID,
	LEFT_NOTEBOOK_ID
};

class OPJEncoThread : public wxThread
{
public:
    OPJEncoThread(OPJCanvas *canvas);

    // thread execution starts here
    virtual void *Entry();

    // called when the thread exits - whether it terminates normally or is
    // stopped with Delete() (but not when it is Kill()ed!)
    virtual void OnExit();

    // write something to the text control
    void WriteText(const wxString& text);

public:
    unsigned m_count;
    OPJCanvas *m_canvas;
};

class OPJDecoThread : public wxThread
{
public:
    OPJDecoThread(OPJCanvas *canvas);

    // thread execution starts here
    virtual void *Entry();

    // called when the thread exits - whether it terminates normally or is
    // stopped with Delete() (but not when it is Kill()ed!)
    virtual void OnExit();

    // write something to the text control
    void WriteText(const wxString& text);

public:
    unsigned m_count;
    OPJCanvas *m_canvas;
};

class OPJParseThread : public wxThread
{
public:
    OPJParseThread(OPJMarkerTree *tree, wxTreeItemId parentid = 0x00);

    // thread execution starts here
    virtual void *Entry();

    // called when the thread exits - whether it terminates normally or is
    // stopped with Delete() (but not when it is Kill()ed!)
    virtual void OnExit();

    // write something to the text control
    void WriteText(const wxString& text);
	void LoadFile(wxFileName fname);
	void ParseJ2KFile(wxFile *m_file, wxFileOffset offset, wxFileOffset length, wxTreeItemId parentid);
	void ParseJP2File(wxFile *fileid, wxFileOffset filepoint, wxFileOffset filelimit, wxTreeItemId parentid);

    unsigned m_count;
    OPJMarkerTree *m_tree;
	wxTreeItemId m_parentid;

private:
	int jpeg2000parse(wxFile *fileid, wxFileOffset filepoint, wxFileOffset filelimit,
			wxTreeItemId parentid, int level, char *scansign, unsigned long int *scanpoint);
	int box_handler_function(int boxtype, wxFile *fileid, wxFileOffset filepoint, wxFileOffset filelimit,
			wxTreeItemId parentid, int level, char *scansign, unsigned long int *scanpoint);

};


// Drag and drop files target
class OPJDnDFile: public wxFileDropTarget
{
public:
    OPJDnDFile(OPJFrame *pOwner) { m_pOwner = pOwner; }
    virtual bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames);

private:
    OPJFrame *m_pOwner;
};



// Property sheet dialog: encoder
class OPJEncoderDialog: public wxPropertySheetDialog
{
DECLARE_CLASS(OPJEncoderDialog)
public:
    OPJEncoderDialog(wxWindow* parent, int dialogType);
    ~OPJEncoderDialog();

	wxBookCtrlBase* m_settingsNotebook;

    wxPanel* CreateMainSettingsPage(wxWindow* parent);
    wxPanel* CreatePart1_1SettingsPage(wxWindow* parent);
    wxPanel* CreatePart1_2SettingsPage(wxWindow* parent);
/*    wxPanel* CreatePart3SettingsPage(wxWindow* parent);*/
	void OnEnableComm(wxCommandEvent& event);
	void OnEnableIdx(wxCommandEvent& event);
	void OnEnablePoc(wxCommandEvent& event);
	void OnRadioQualityRate(wxCommandEvent& event);
#ifdef USE_JPWL
	void OnEnableJPWL(wxCommandEvent& event);
	wxPanel* CreatePart11SettingsPage(wxWindow* parent);
	/*wxCheckBox *m_enablejpwlCheck;*/
	wxChoice *m_hprotChoice[MYJPWL_MAX_NO_TILESPECS];
	wxSpinCtrl *m_htileCtrl[MYJPWL_MAX_NO_TILESPECS];
	wxChoice *m_pprotChoice[MYJPWL_MAX_NO_TILESPECS];
	wxSpinCtrl *m_ptileCtrl[MYJPWL_MAX_NO_TILESPECS];
	wxSpinCtrl *m_ppackCtrl[MYJPWL_MAX_NO_TILESPECS];
	wxChoice *m_sensiChoice[MYJPWL_MAX_NO_TILESPECS];
	wxSpinCtrl *m_stileCtrl[MYJPWL_MAX_NO_TILESPECS];
	void OnHprotSelect(wxCommandEvent& event);
	void OnPprotSelect(wxCommandEvent& event);
	void OnSensiSelect(wxCommandEvent& event);
#endif // USE_JPWL

	wxTextCtrl *m_subsamplingCtrl, *m_originCtrl, *m_rateCtrl, *m_commentCtrl;
	wxRadioButton *m_rateRadio, *m_qualityRadio;
	wxTextCtrl *m_indexCtrl, *m_qualityCtrl, *m_cbsizeCtrl, *m_prsizeCtrl, *m_pocCtrl;
	wxTextCtrl *m_tsizeCtrl, *m_toriginCtrl;
	wxRadioBox *progressionBox;
	wxCheckBox *m_enablecommCheck, *m_enableidxCheck, *m_mctCheck, *m_irrevCheck;
	wxCheckBox *m_sopCheck, *m_ephCheck, *m_enablebypassCheck, *m_enableresetCheck,
		*m_enablerestartCheck, *m_enablevscCheck, *m_enableertermCheck, *m_enablesegmarkCheck;
	wxCheckBox *m_enablepocCheck, *m_enablejpwlCheck;
	wxSpinCtrl *m_resolutionsCtrl;

protected:

    enum {
		OPJENCO_ENABLEJPWL = 100,
		OPJENCO_RATEFACTOR,
		OPJENCO_RATERADIO,
		OPJENCO_QUALITYFACTOR,
		OPJENCO_QUALITYRADIO,
		OPJENCO_RESNUMBER,
		OPJENCO_CODEBLOCKSIZE,
		OPJENCO_PRECINCTSIZE,
		OPJENCO_TILESIZE,
		OPJENCO_PROGRESSION,
		OPJENCO_SUBSAMPLING,
		OPJENCO_ENABLESOP,
		OPJENCO_ENABLEEPH,
		OPJENCO_ENABLEBYPASS,
		OPJENCO_ENABLERESET,
		OPJENCO_ENABLERESTART,
		OPJENCO_ENABLEVSC,
		OPJENCO_ENABLEERTERM,
		OPJENCO_ENABLESEGMARK,
		OPJENCO_ENABLEPOC,
		OPJENCO_ROICOMP,
		OPJENCO_ROISHIFT,
		OPJENCO_IMORIG,
		OPJENCO_TILORIG,
		OPJENCO_ENABLEMCT,
		OPJENCO_ENABLEIRREV,
		OPJENCO_ENABLEINDEX,
		OPJENCO_INDEXNAME,
		OPJENCO_POCSPEC,
		OPJENCO_ENABLECOMM,
		OPJENCO_COMMENTTEXT,
		OPJENCO_HPROT,
		OPJENCO_HTILE,
		OPJENCO_PPROT,
		OPJENCO_PTILE,
		OPJENCO_PPACK,
		OPJENCO_SENSI,
		OPJENCO_STILE
    };

DECLARE_EVENT_TABLE()
};

// Property sheet dialog: decoder
class OPJDecoderDialog: public wxPropertySheetDialog
{
DECLARE_CLASS(OPJDecoderDialog)
public:
    OPJDecoderDialog(wxWindow* parent, int dialogType);
    ~OPJDecoderDialog();

	wxBookCtrlBase* m_settingsNotebook;
	wxCheckBox *m_enabledecoCheck, *m_enableparseCheck;
	wxSpinCtrl *m_reduceCtrl, *m_layerCtrl, *m_numcompsCtrl;
	wxRadioBox* m_resizeBox;

	void OnEnableDeco(wxCommandEvent& event);

    wxPanel* CreateMainSettingsPage(wxWindow* parent);
    wxPanel* CreatePart1SettingsPage(wxWindow* parent);
    wxPanel* CreatePart3SettingsPage(wxWindow* parent);
#ifdef USE_JPWL
	void OnEnableJPWL(wxCommandEvent& event);
    wxPanel* CreatePart11SettingsPage(wxWindow* parent);
	wxSpinCtrl *m_expcompsCtrl, *m_framenumCtrl, *m_maxtilesCtrl;
	wxCheckBox *m_enablejpwlCheck;
#endif // USE_JPWL


protected:

    enum {
		OPJDECO_RESMETHOD = 100,
		OPJDECO_REDUCEFACTOR,
		OPJDECO_QUALITYLAYERS,
		OPJDECO_NUMCOMPS,
		OPJDECO_ENABLEDECO,
		OPJDECO_ENABLEPARSE,
		OPJDECO_ENABLEJPWL,
		OPJDECO_EXPCOMPS,
		OPJDECO_MAXTILES,
		OPJDECO_FRAMENUM
    };

DECLARE_EVENT_TABLE()
};

#endif //__OPJ_VIEWER_H__



