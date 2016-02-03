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
#include "OPJViewer.h"


/////////////////////////////////////////////////////////////////////
// Encoding thread class
/////////////////////////////////////////////////////////////////////

OPJEncoThread::OPJEncoThread(OPJCanvas *canvas)
        : wxThread()
{
    m_count = 0;
    m_canvas = canvas;
}

void OPJEncoThread::WriteText(const wxString& text)
{
    wxString msg;

    // before doing any GUI calls we must ensure that this thread is the only
    // one doing it!

#ifndef __WXGTK__ 
    wxMutexGuiEnter();
#endif // __WXGTK__

    msg << text;
    m_canvas->WriteText(msg);

#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif // __WXGTK__
}

void OPJEncoThread::OnExit()
{
    wxCriticalSectionLocker locker(wxGetApp().m_enco_critsect);

    wxArrayThread& ethreads = wxGetApp().m_enco_threads;
    ethreads.Remove(this);

    if (ethreads.IsEmpty() )
    {
        // signal the main thread that there are no more threads left if it is
        // waiting for us
        if (wxGetApp().m_enco_waitingUntilAllDone) {
            wxGetApp().m_enco_waitingUntilAllDone = false;
            wxGetApp().m_enco_semAllDone.Post();
        }
    }
}

void *OPJEncoThread::Entry()
{
    wxString text;

	srand(GetId());
	//int m_countnum = rand() % 9;
    //text.Printf(wxT("Deco thread 0x%lx started (priority = %u, time = %d)."),
    //            GetId(), GetPriority(), m_countnum);
    text.Printf(wxT("Enco thread %d started"), m_canvas->m_childframe->m_winnumber);
    WriteText(text);

	// set handler properties
	wxJPEG2000Handler *jpeg2000handler = (wxJPEG2000Handler *) wxImage::FindHandler(wxBITMAP_TYPE_JPEG2000);
	jpeg2000handler->m_subsampling = wxGetApp().m_subsampling;
	jpeg2000handler->m_origin = wxGetApp().m_origin;
	jpeg2000handler->m_rates = wxGetApp().m_rates;
	jpeg2000handler->m_quality = wxGetApp().m_quality;
	jpeg2000handler->m_enablequality = wxGetApp().m_enablequality;
	jpeg2000handler->m_multicomp = wxGetApp().m_multicomp;
	jpeg2000handler->m_irreversible = wxGetApp().m_irreversible;
	jpeg2000handler->m_resolutions = wxGetApp().m_resolutions;
	jpeg2000handler->m_progression = wxGetApp().m_progression;
	jpeg2000handler->m_cbsize = wxGetApp().m_cbsize;
	jpeg2000handler->m_prsize = wxGetApp().m_prsize;
	jpeg2000handler->m_tsize = wxGetApp().m_tsize;
	jpeg2000handler->m_torigin = wxGetApp().m_torigin;
	jpeg2000handler->m_enablesop = wxGetApp().m_enablesop;
	jpeg2000handler->m_enableeph = wxGetApp().m_enableeph;
	jpeg2000handler->m_enablebypass = wxGetApp().m_enablebypass;
	jpeg2000handler->m_enablerestart = wxGetApp().m_enablerestart;
	jpeg2000handler->m_enablereset = wxGetApp().m_enablereset;
	jpeg2000handler->m_enablesegmark = wxGetApp().m_enablesegmark;
	jpeg2000handler->m_enableerterm = wxGetApp().m_enableerterm;
	jpeg2000handler->m_enablevsc = wxGetApp().m_enablevsc;
	jpeg2000handler->m_enableidx = wxGetApp().m_enableidx;
	jpeg2000handler->m_index = m_canvas->m_savename.GetPath(wxPATH_GET_VOLUME | wxPATH_GET_SEPARATOR) + wxGetApp().m_index;
	jpeg2000handler->m_enablecomm = wxGetApp().m_enablecomm;
	jpeg2000handler->m_comment = wxGetApp().m_comment;
	jpeg2000handler->m_enablepoc = wxGetApp().m_enablepoc;
	jpeg2000handler->m_poc = wxGetApp().m_poc;

	// save the file
	if (!m_canvas->m_image100.SaveFile(m_canvas->m_savename.GetFullPath(), (wxBitmapType) wxBITMAP_TYPE_JPEG2000)) {
		WriteText(wxT("Can't save image"));
		return NULL;
	}

    text.Printf(wxT("Enco thread %d finished"), m_canvas->m_childframe->m_winnumber);
    WriteText(text);
    return NULL;
}


/////////////////////////////////////////////////////////////////////
// Decoding thread class
/////////////////////////////////////////////////////////////////////
OPJDecoThread::OPJDecoThread(OPJCanvas *canvas)
        : wxThread()
{
    m_count = 0;
    m_canvas = canvas;
}

void OPJDecoThread::WriteText(const wxString& text)
{
    wxString msg;
	
	// we use a fake event and post it for inter-thread gui communication
    wxCommandEvent event(wxEVT_COMMAND_MENU_SELECTED, OPJFRAME_THREADLOGMSG);
    event.SetInt(-1); 
	msg << text;
	event.SetString(msg);
    wxPostEvent(this->m_canvas->m_childframe->m_frame, event);

/*
    // before doing any GUI calls we must ensure that this thread is the only
    // one doing it!

#ifndef __WXGTK__ 
    wxMutexGuiEnter();
#endif // __WXGTK__

    msg << text;
    m_canvas->WriteText(msg);

#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif // __WXGTK__
*/
}

void OPJDecoThread::OnExit()
{
    wxCriticalSectionLocker locker(wxGetApp().m_deco_critsect);

    wxArrayThread& dthreads = wxGetApp().m_deco_threads;
    dthreads.Remove(this);

    if (dthreads.IsEmpty() )
    {
        // signal the main thread that there are no more threads left if it is
        // waiting for us
        if (wxGetApp().m_deco_waitingUntilAllDone) {
            wxGetApp().m_deco_waitingUntilAllDone = false;
            wxGetApp().m_deco_semAllDone.Post();
        }
    }
}

void *OPJDecoThread::Entry()
{

    wxString text;

	//srand(GetId());
	//int m_countnum = rand() % 9;
    //text.Printf(wxT("Deco thread 0x%lx started (priority = %u, time = %d)."),
    //            GetId(), GetPriority(), m_countnum);

	// we have started
    text.Printf(wxT("Deco thread %d started"), m_canvas->m_childframe->m_winnumber);
    WriteText(text);

	// prepare dummy wximage
    wxBitmap bitmap(100, 100);
    wxImage image(100, 100, true); //= bitmap.ConvertToImage();
    image.Destroy();

	// show image full name
	WriteText(m_canvas->m_fname.GetFullPath());

	// set handler properties
	wxJPEG2000Handler *jpeg2000handler = (wxJPEG2000Handler *) wxImage::FindHandler(wxBITMAP_TYPE_JPEG2000);
	jpeg2000handler->m_reducefactor = wxGetApp().m_reducefactor;
	jpeg2000handler->m_qualitylayers = wxGetApp().m_qualitylayers;
	jpeg2000handler->m_components = wxGetApp().m_components;
	jpeg2000handler->m_framenum = wxGetApp().m_framenum;
#ifdef USE_JPWL
	jpeg2000handler->m_enablejpwl = wxGetApp().m_enablejpwl;
	jpeg2000handler->m_expcomps = wxGetApp().m_expcomps;
	jpeg2000handler->m_maxtiles = wxGetApp().m_maxtiles;
#endif // USE_JPWL

#ifdef USE_MXF
	wxMXFHandler *mxfffhandler = (wxMXFHandler *) wxImage::FindHandler(wxBITMAP_TYPE_MXF);
	mxfffhandler->m_reducefactor = wxGetApp().m_reducefactor;
	mxfffhandler->m_qualitylayers = wxGetApp().m_qualitylayers;
	mxfffhandler->m_components = wxGetApp().m_components;
	mxfffhandler->m_framenum = wxGetApp().m_framenum;
	mxfffhandler->m_filename = m_canvas->m_fname;
#ifdef USE_JPWL
	mxfffhandler->m_enablejpwl = wxGetApp().m_enablejpwl;
	mxfffhandler->m_expcomps = wxGetApp().m_expcomps;
	mxfffhandler->m_maxtiles = wxGetApp().m_maxtiles;
#endif // USE_JPWL
#endif // USE_MXF

	// if decoding is enabled...
	if (wxGetApp().m_enabledeco) {

		// load the file
		if (!image.LoadFile(m_canvas->m_fname.GetFullPath(), wxBITMAP_TYPE_ANY, 0)) {
			WriteText(wxT("Can't load image!"));
			return NULL;
		}

	} else {

		// display a warning
		if (!image.Create(300, 5, false)) {
			WriteText(wxT("Can't create image!"));
			return NULL;
		}

	}

	// assign 100% image
    m_canvas->m_image100 = wxBitmap(image);

	// signal the frame to refresh the canvas
    wxCommandEvent event(wxEVT_COMMAND_MENU_SELECTED, OPJFRAME_VIEWFIT);
	event.SetString(wxT("Fit me"));
    event.SetInt(m_canvas->m_childframe->m_winnumber); 
    wxPostEvent(m_canvas->m_childframe->m_frame, event);

	// find a fit-to-width zoom
	/*int zooml, wzooml, hzooml;
	wxSize clientsize = m_canvas->GetClientSize();
	wzooml = (int) floor(100.0 * (double) clientsize.GetWidth() / (double) (2 * OPJ_CANVAS_BORDER + image.GetWidth()));
	hzooml = (int) floor(100.0 * (double) clientsize.GetHeight() / (double) (2 * OPJ_CANVAS_BORDER + image.GetHeight()));
	zooml = wxMin(100, wxMin(wzooml, hzooml));*/

	// fit to width
#ifndef __WXGTK__
	//m_canvas->m_childframe->m_frame->Rescale(zooml, m_canvas->m_childframe);
#endif // __WXGTK__

	//m_canvas->m_image = m_canvas->m_image100;
	//m_canvas->Refresh();
	//m_canvas->SetScrollbars(20, 20, (int)(0.5 + (double) image.GetWidth() / 20.0), (int)(0.5 + (double) image.GetHeight() / 20.0));

    //text.Printf(wxT("Deco thread 0x%lx finished."), GetId());
    text.Printf(wxT("Deco thread %d finished"), m_canvas->m_childframe->m_winnumber);
    WriteText(text);
    return NULL;

}

/////////////////////////////////////////////////////////////////////
// Parsing thread class
/////////////////////////////////////////////////////////////////////

OPJParseThread::OPJParseThread(OPJMarkerTree *tree, wxTreeItemId parentid)
        : wxThread()
{
    m_count = 0;
    m_tree = tree;
	m_parentid = parentid;
}

void OPJParseThread::WriteText(const wxString& text)
{
    wxString msg;
	
	// we use a fake event and post it for inter-thread gui communication
    wxCommandEvent event(wxEVT_COMMAND_MENU_SELECTED, OPJFRAME_THREADLOGMSG);
    event.SetInt(-1); 
	msg << text;
	event.SetString(msg);
    wxPostEvent(this->m_tree->m_childframe->m_frame, event);

/*    // before doing any GUI calls we must ensure that this thread is the only
    // one doing it!

#ifndef __WXGTK__ 
    wxMutexGuiEnter();
#endif // __WXGTK

    msg << text;
    m_tree->WriteText(msg);

#ifndef __WXGTK__ 
    wxMutexGuiLeave();
#endif // __WXGTK*/
}

void OPJParseThread::OnExit()
{
    wxCriticalSectionLocker locker(wxGetApp().m_parse_critsect);

    wxArrayThread& threads = wxGetApp().m_parse_threads;
    threads.Remove(this);

    if (threads.IsEmpty()) {
        // signal the main thread that there are no more threads left if it is
        // waiting for us
        if (wxGetApp().m_parse_waitingUntilAllDone) {
            wxGetApp().m_parse_waitingUntilAllDone = false;
            wxGetApp().m_parse_semAllDone.Post();
        }
    }
}

void *OPJParseThread::Entry()
{

	printf("Entering\n\n");

    wxString text;

	srand(GetId());
	int m_countnum = rand() % 9;
    text.Printf(wxT("Parse thread 0x%lx started (priority = %u, time = %d)."),
            GetId(), GetPriority(), m_countnum);
    WriteText(text);
    LoadFile(m_tree->m_fname);
    text.Printf(wxT("Parse thread 0x%lx finished."), GetId());
    WriteText(text);


    //wxLogMessage(wxT("Entering\n")); //test wxLog thread safeness

	//wxBusyCursor wait;
	//wxBusyInfo wait(wxT("Decoding image ..."));


    /*for ( m_count = 0; m_count < m_countnum; m_count++ )
    {
        // check if we were asked to exit
        if ( TestDestroy() )
            break;

        text.Printf(wxT("[%u] Parse thread 0x%lx here."), m_count, GetId());
        WriteText(text);

        // wxSleep() can't be called from non-GUI thread!
        wxThread::Sleep(10);
    }*/

    // wxLogMessage(text); -- test wxLog thread safeness

	printf("Exiting\n\n");

    return NULL;
}


///////////////////////////////////////////
// Parsing hread and related
///////////////////////////////////////////

#if USE_GENERIC_TREECTRL
BEGIN_EVENT_TABLE(OPJMarkerTree, wxGenericTreeCtrl)
#else
BEGIN_EVENT_TABLE(OPJMarkerTree, wxTreeCtrl)
#endif
    /*EVT_TREE_BEGIN_DRAG(TreeTest_Ctrl, OPJMarkerTree::OnBeginDrag)
    EVT_TREE_BEGIN_RDRAG(TreeTest_Ctrl, OPJMarkerTree::OnBeginRDrag)
    EVT_TREE_END_DRAG(TreeTest_Ctrl, OPJMarkerTree::OnEndDrag)*/
    /*EVT_TREE_BEGIN_LABEL_EDIT(TreeTest_Ctrl, OPJMarkerTree::OnBeginLabelEdit)
    EVT_TREE_END_LABEL_EDIT(TreeTest_Ctrl, OPJMarkerTree::OnEndLabelEdit)*/
    /*EVT_TREE_DELETE_ITEM(TreeTest_Ctrl, OPJMarkerTree::OnDeleteItem)*/
#if 0       // there are so many of those that logging them causes flicker
    /*EVT_TREE_GET_INFO(TreeTest_Ctrl, OPJMarkerTree::OnGetInfo)*/
#endif
    /*EVT_TREE_SET_INFO(TreeTest_Ctrl, OPJMarkerTree::OnSetInfo)
    EVT_TREE_ITEM_EXPANDED(TreeTest_Ctrl, OPJMarkerTree::OnItemExpanded)*/
    EVT_TREE_ITEM_EXPANDING(TreeTest_Ctrl, OPJMarkerTree::OnItemExpanding)
    /*EVT_TREE_ITEM_COLLAPSED(TreeTest_Ctrl, OPJMarkerTree::OnItemCollapsed)
    EVT_TREE_ITEM_COLLAPSING(TreeTest_Ctrl, OPJMarkerTree::OnItemCollapsing)*/

    EVT_TREE_SEL_CHANGED(TreeTest_Ctrl, OPJMarkerTree::OnSelChanged)
    /*EVT_TREE_SEL_CHANGING(TreeTest_Ctrl, OPJMarkerTree::OnSelChanging)*/
    /*EVT_TREE_KEY_DOWN(TreeTest_Ctrl, OPJMarkerTree::OnTreeKeyDown)*/
    /*EVT_TREE_ITEM_ACTIVATED(TreeTest_Ctrl, OPJMarkerTree::OnItemActivated)*/

    // so many differents ways to handle right mouse button clicks...
    /*EVT_CONTEXT_MENU(OPJMarkerTree::OnContextMenu)*/
    // EVT_TREE_ITEM_MENU is the preferred event for creating context menus
    // on a tree control, because it includes the point of the click or item,
    // meaning that no additional placement calculations are required.
    EVT_TREE_ITEM_MENU(TreeTest_Ctrl, OPJMarkerTree::OnItemMenu)
    /*EVT_TREE_ITEM_RIGHT_CLICK(TreeTest_Ctrl, OPJMarkerTree::OnItemRClick)*/

    /*EVT_RIGHT_DOWN(OPJMarkerTree::OnRMouseDown)
    EVT_RIGHT_UP(OPJMarkerTree::OnRMouseUp)
    EVT_RIGHT_DCLICK(OPJMarkerTree::OnRMouseDClick)*/
END_EVENT_TABLE()

// OPJMarkerTree implementation
#if USE_GENERIC_TREECTRL
IMPLEMENT_DYNAMIC_CLASS(OPJMarkerTree, wxGenericTreeCtrl)
#else
IMPLEMENT_DYNAMIC_CLASS(OPJMarkerTree, wxTreeCtrl)
#endif

OPJMarkerTree::OPJMarkerTree(wxWindow *parent, OPJChildFrame *subframe, wxFileName fname, wxString name, const wxWindowID id,
           const wxPoint& pos, const wxSize& size, long style)
          : wxTreeCtrl(parent, id, pos, size, style)
{
    m_reverseSort = false;
	m_fname = fname;

	m_peektextCtrl = ((OPJFrame *) (parent->GetParent()->GetParent()))->m_textCtrlbrowse;
    CreateImageList();

    // Add some items to the tree
    //AddTestItemsToTree(5, 5);
    int image = wxGetApp().ShowImages() ? OPJMarkerTree::TreeCtrlIcon_Folder : -1;
    wxTreeItemId rootId = AddRoot(name,
                                  image, image,
                                  new OPJMarkerData(name));

    OPJParseThread *pthread = CreateParseThread(0x00, subframe);
    if (pthread->Run() != wxTHREAD_NO_ERROR)
        wxLogMessage(wxT("Can't start parse thread!"));
    else
		wxLogMessage(wxT("New parse thread started."));

	m_childframe = subframe;
}

void OPJMarkerTree::CreateImageList(int size)
{
    if (size == -1) {
        SetImageList(NULL);
        return;
    }
    if (size == 0)
        size = m_imageSize;
    else
        m_imageSize = size;

    // Make an image list containing small icons
    wxImageList *images = new wxImageList(size, size, true);

    // should correspond to TreeCtrlIcon_xxx enum
    wxBusyCursor wait;
    wxIcon icons[5];
    icons[0] = wxIcon(icon1_xpm);
    icons[1] = wxIcon(icon2_xpm);
    icons[2] = wxIcon(icon3_xpm);
    icons[3] = wxIcon(icon4_xpm);
    icons[4] = wxIcon(icon5_xpm);

    int sizeOrig = icons[0].GetWidth();
    for (size_t i = 0; i < WXSIZEOF(icons); i++) {
        if (size == sizeOrig) {
            images->Add(icons[i]);
        } else {
            images->Add(wxBitmap(wxBitmap(icons[i]).ConvertToImage().Rescale(size, size)));
        }
    }

    AssignImageList(images);
}

#if USE_GENERIC_TREECTRL || !defined(__WXMSW__)
void OPJMarkerTree::CreateButtonsImageList(int size)
{
    if ( size == -1 ) {
        SetButtonsImageList(NULL);
        return;
    }

    // Make an image list containing small icons
    wxImageList *images = new wxImageList(size, size, true);

    // should correspond to TreeCtrlIcon_xxx enum
    wxBusyCursor wait;
    wxIcon icons[4];
    icons[0] = wxIcon(icon3_xpm);   // closed
    icons[1] = wxIcon(icon3_xpm);   // closed, selected
    icons[2] = wxIcon(icon5_xpm);   // open
    icons[3] = wxIcon(icon5_xpm);   // open, selected

    for ( size_t i = 0; i < WXSIZEOF(icons); i++ ) {
        int sizeOrig = icons[i].GetWidth();
        if ( size == sizeOrig ) {
            images->Add(icons[i]);
        } else {
            images->Add(wxBitmap(wxBitmap(icons[i]).ConvertToImage().Rescale(size, size)));
        }
    }

    AssignButtonsImageList(images);
#else
void OPJMarkerTree::CreateButtonsImageList(int WXUNUSED(size))
{
#endif
}

void OPJParseThread::LoadFile(wxFileName fname)
{
	wxTreeItemId rootid;

	// this is the root node
	int image = wxGetApp().ShowImages() ? m_tree->TreeCtrlIcon_Folder : -1;

	if (this->m_parentid) {
		// leaf of a tree
		rootid = m_parentid;
		m_tree->SetItemText(rootid, wxT("Parsing..."));

	} else {

		// delete the existing tree hierarchy
		m_tree->DeleteAllItems();

		// new tree
		rootid = m_tree->AddRoot(wxT("Parsing..."),
			image,
			image,
			new OPJMarkerData(fname.GetFullPath())
			);
		//m_tree->SetItemFont(rootid, *wxITALIC_FONT);
		m_tree->SetItemBold(rootid);
	}

	// open the file
	wxFile m_file(fname.GetFullPath().c_str(), wxFile::read);

	// parsing enabled?
	if (wxGetApp().m_enableparse) {

		// what is the extension?
		if ((fname.GetExt() == wxT("j2k")) || (fname.GetExt() == wxT("j2c"))) {

			// parse the file
			ParseJ2KFile(&m_file, 0, m_file.Length(), rootid);

		} else if ((fname.GetExt() == wxT("jp2")) || (fname.GetExt() == wxT("mj2"))) {

			// parse the file
			if (this->m_parentid) {
				//WriteText(wxT("Only a subsection of jp2"));
				OPJMarkerData *data = (OPJMarkerData *) m_tree->GetItemData(rootid);
				ParseJ2KFile(&m_file, data->m_start, data->m_length, rootid);
				m_tree->Expand(rootid);

			} else {
				// as usual
				ParseJP2File(&m_file, 0, m_file.Length(), rootid);
			}

		} else {

			// unknown extension
			WriteText(wxT("Unknown file format!"));

		}

	}

	// this is the root node
	if (this->m_parentid)
		m_tree->SetItemText(rootid, wxT("Codestream"));
	else
		//m_tree->SetItemText(rootid, wxString::Format(wxT("%s (%d B)"), fname.GetFullName(), m_file.Length()));
		m_tree->SetItemText(rootid, fname.GetFullName());

	// close the file
	m_file.Close();

	WriteText(wxT("Parsing finished!"));
}

/*int OPJMarkerTree::OnCompareItems(const wxTreeItemId& item1,
                               const wxTreeItemId& item2)
{
    if ( m_reverseSort )
    {
        // just exchange 1st and 2nd items
        return wxTreeCtrl::OnCompareItems(item2, item1);
    }
    else
    {
        return wxTreeCtrl::OnCompareItems(item1, item2);
    }
}*/

/*void OPJMarkerTree::AddItemsRecursively(const wxTreeItemId& idParent,
                                     size_t numChildren,
                                     size_t depth,
                                     size_t folder)
{
    if ( depth > 0 )
    {
        bool hasChildren = depth > 1;

        wxString str;
        for ( size_t n = 0; n < numChildren; n++ )
        {
            // at depth 1 elements won't have any more children
            if ( hasChildren )
                str.Printf(wxT("%s child %u"), wxT("Folder"), unsigned(n + 1));
            else
                str.Printf(wxT("%s child %u.%u"), wxT("File"), unsigned(folder), unsigned(n + 1));

            // here we pass to AppendItem() normal and selected item images (we
            // suppose that selected image follows the normal one in the enum)
            int image, imageSel;
            if ( wxGetApp().ShowImages() )
            {
                image = depth == 1 ? TreeCtrlIcon_File : TreeCtrlIcon_Folder;
                imageSel = image + 1;
            }
            else
            {
                image = imageSel = -1;
            }
            wxTreeItemId id = AppendItem(idParent, str, image, imageSel,
                                         new OPJMarkerData(str));

            // and now we also set the expanded one (only for the folders)
            if ( hasChildren && wxGetApp().ShowImages() )
            {
                SetItemImage(id, TreeCtrlIcon_FolderOpened,
                             wxTreeItemIcon_Expanded);
            }

            // remember the last child for OnEnsureVisible()
            if ( !hasChildren && n == numChildren - 1 )
            {
                m_lastItem = id;
            }

            AddItemsRecursively(id, numChildren, depth - 1, n + 1);
        }
    }
    //else: done!
}*/

/*void OPJMarkerTree::AddTestItemsToTree(size_t numChildren,
                                    size_t depth)
{
    int image = wxGetApp().ShowImages() ? OPJMarkerTree::TreeCtrlIcon_Folder : -1;
    wxTreeItemId rootId = AddRoot(wxT("Root"),
                                  image, image,
                                  new OPJMarkerData(wxT("Root item")));
    if ( image != -1 )
    {
        SetItemImage(rootId, TreeCtrlIcon_FolderOpened, wxTreeItemIcon_Expanded);
    }

    AddItemsRecursively(rootId, numChildren, depth, 0);

    // set some colours/fonts for testing
    SetItemFont(rootId, *wxITALIC_FONT);

    wxTreeItemIdValue cookie;
    wxTreeItemId id = GetFirstChild(rootId, cookie);
    SetItemTextColour(id, *wxBLUE);

    id = GetNextChild(rootId, cookie);
    id = GetNextChild(rootId, cookie);
    SetItemTextColour(id, *wxRED);
    SetItemBackgroundColour(id, *wxLIGHT_GREY);
}*/

/*void OPJMarkerTree::GetItemsRecursively(const wxTreeItemId& idParent,
                                     wxTreeItemIdValue cookie)
{
    wxTreeItemId id;

    if ( !cookie )
        id = GetFirstChild(idParent, cookie);
    else
        id = GetNextChild(idParent, cookie);

    if ( !id.IsOk() )
        return;

    wxString text = GetItemText(id);
    wxLogMessage(text);

    if (ItemHasChildren(id))
        GetItemsRecursively(id);

    GetItemsRecursively(idParent, cookie);
}*/

/*void OPJMarkerTree::DoToggleIcon(const wxTreeItemId& item)
{
    int image = (GetItemImage(item) == TreeCtrlIcon_Folder)
                    ? TreeCtrlIcon_File
                    : TreeCtrlIcon_Folder;
    SetItemImage(item, image, wxTreeItemIcon_Normal);

    image = (GetItemImage(item) == TreeCtrlIcon_FolderSelected)
                    ? TreeCtrlIcon_FileSelected
                    : TreeCtrlIcon_FolderSelected;
    SetItemImage(item, image, wxTreeItemIcon_Selected);
}*/

void OPJMarkerTree::LogEvent(const wxChar *name, const wxTreeEvent& event)
{
    wxTreeItemId item = event.GetItem();
    wxString text;
    if ( item.IsOk() )
        text << wxT('"') << GetItemText(item).c_str() << wxT('"');
    else
        text = wxT("invalid item");
    wxLogMessage(wxT("%s(%s)"), name, text.c_str());
}

OPJParseThread *OPJMarkerTree::CreateParseThread(wxTreeItemId parentid, OPJChildFrame *subframe)
{
    OPJParseThread *pthread = new OPJParseThread(this, parentid);

    if (pthread->Create() != wxTHREAD_NO_ERROR)
		wxLogError(wxT("Can't create parse thread!"));

    wxCriticalSectionLocker enter(wxGetApp().m_parse_critsect);
    wxGetApp().m_parse_threads.Add(pthread);

    return pthread;
}


/*// avoid repetition
#define TREE_EVENT_HANDLER(name)                                 \
void OPJMarkerTree::name(wxTreeEvent& event)                        \
{                                                                \
    LogEvent(_T(#name), event);                                  \
    SetLastItem(wxTreeItemId());                                 \
    event.Skip();                                                \
}*/

/*TREE_EVENT_HANDLER(OnBeginRDrag)*/
/*TREE_EVENT_HANDLER(OnDeleteItem)*/
/*TREE_EVENT_HANDLER(OnGetInfo)
TREE_EVENT_HANDLER(OnSetInfo)*/
/*TREE_EVENT_HANDLER(OnItemExpanded)
TREE_EVENT_HANDLER(OnItemExpanding)*/
/*TREE_EVENT_HANDLER(OnItemCollapsed)*/
/*TREE_EVENT_HANDLER(OnSelChanged)
TREE_EVENT_HANDLER(OnSelChanging)*/

/*#undef TREE_EVENT_HANDLER*/

void OPJMarkerTree::OnItemExpanding(wxTreeEvent& event)
{
	wxTreeItemId item = event.GetItem();
	OPJMarkerData* data = (OPJMarkerData *) GetItemData(item);
	wxString text;

	if (item.IsOk())
		text << wxT('"') << GetItemText(item).c_str() << wxT('"');
	else
		text = wxT("invalid item");

	if (wxStrcmp(data->GetDesc1(), wxT("INFO-CSTREAM")))
		return;

	wxLogMessage(wxT("Expanding... (%s -> %s, %s, %d, %d)"),
		text.c_str(), data->GetDesc1(), data->GetDesc2(),
		data->m_start, data->m_length);

	// the codestream box is being asked for expansion
	wxTreeItemIdValue cookie;
	if (!GetFirstChild(item, cookie).IsOk()) {
		OPJParseThread *pthread = CreateParseThread(item);
		if (pthread->Run() != wxTHREAD_NO_ERROR)
			wxLogMessage(wxT("Can't start parse thread!"));
		else
			wxLogMessage(wxT("New parse thread started."));
	}
}

void OPJMarkerTree::OnSelChanged(wxTreeEvent& event)
{
	int bunch_linesize = 16;
	int bunch_numlines = 7;

	wxTreeItemId item = event.GetItem();
	OPJMarkerData* data = (OPJMarkerData *) GetItemData(item);
	wxString text;
	int l, c, pos = 0, pre_pos;

	m_peektextCtrl->Clear();

	/*text << wxString::Format(wxT("Selected... (%s -> %s, %s, %d, %d)"),
		text.c_str(), data->GetDesc1(), data->GetDesc2(),
		data->m_start, data->m_length) << wxT("\n");*/

	// open the file and browse a little
	wxFile *fp = new wxFile(m_fname.GetFullPath().c_str(), wxFile::read);

	// go to position claimed
	fp->Seek(data->m_start, wxFromStart);

	// read a bunch
	int max_read = wxMin(wxFileOffset(bunch_linesize * bunch_numlines), data->m_length - data->m_start + 1);
	if (data->m_desc == wxT("MARK (65380)")) {
		/*wxLogMessage(data->m_desc);*/
		max_read = data->m_length - data->m_start + 1;
		bunch_numlines = (int) ceil((float) max_read / (float) bunch_linesize);
	}
	unsigned char *buffer = new unsigned char[bunch_linesize * bunch_numlines];
	fp->Read(buffer, max_read);

	// write the file data between start and stop
	pos = 0;
	for (l = 0; l < bunch_numlines; l++) {

		text << wxString::Format(wxT("%010d:"), data->m_start + pos);

		pre_pos = pos;

		// add hex browsing text
		for (c = 0; c < bunch_linesize; c++) {

			if (!(c % 8))
				text << wxT(" ");

			if (pos < max_read) {
				text << wxString::Format(wxT("%02X "), buffer[pos]);
			} else
				text << wxT("   ");
			pos++;
		}

		text << wxT("    ");

		// add char browsing text
		for (c = 0; c < bunch_linesize; c++) {

			if (pre_pos < max_read) {
				if ((buffer[pre_pos] == '\n') ||
					(buffer[pre_pos] == '\t') ||
					(buffer[pre_pos] == '\0') ||
					(buffer[pre_pos] == 0x0D) ||
					(buffer[pre_pos] == 0x0B))
					buffer[pre_pos] = ' ';
				text << wxString::FromAscii((char) buffer[pre_pos]) << wxT(".");
			} else
				text << wxT("  ");
			pre_pos++;
		}

		text << wxT("\n");

	}

	// close the file
	fp->Close();

	m_peektextCtrl->WriteText(text);

	delete buffer;
}

/*void LogKeyEvent(const wxChar *name, const wxKeyEvent& event)
{
    wxString key;
    long keycode = event.GetKeyCode();
    {
        switch ( keycode )
        {
            case WXK_BACK: key = wxT("BACK"); break;
            case WXK_TAB: key = wxT("TAB"); break;
            case WXK_RETURN: key = wxT("RETURN"); break;
            case WXK_ESCAPE: key = wxT("ESCAPE"); break;
            case WXK_SPACE: key = wxT("SPACE"); break;
            case WXK_DELETE: key = wxT("DELETE"); break;
            case WXK_START: key = wxT("START"); break;
            case WXK_LBUTTON: key = wxT("LBUTTON"); break;
            case WXK_RBUTTON: key = wxT("RBUTTON"); break;
            case WXK_CANCEL: key = wxT("CANCEL"); break;
            case WXK_MBUTTON: key = wxT("MBUTTON"); break;
            case WXK_CLEAR: key = wxT("CLEAR"); break;
            case WXK_SHIFT: key = wxT("SHIFT"); break;
            case WXK_ALT: key = wxT("ALT"); break;
            case WXK_CONTROL: key = wxT("CONTROL"); break;
            case WXK_MENU: key = wxT("MENU"); break;
            case WXK_PAUSE: key = wxT("PAUSE"); break;
            case WXK_CAPITAL: key = wxT("CAPITAL"); break;
            case WXK_END: key = wxT("END"); break;
            case WXK_HOME: key = wxT("HOME"); break;
            case WXK_LEFT: key = wxT("LEFT"); break;
            case WXK_UP: key = wxT("UP"); break;
            case WXK_RIGHT: key = wxT("RIGHT"); break;
            case WXK_DOWN: key = wxT("DOWN"); break;
            case WXK_SELECT: key = wxT("SELECT"); break;
            case WXK_PRINT: key = wxT("PRINT"); break;
            case WXK_EXECUTE: key = wxT("EXECUTE"); break;
            case WXK_SNAPSHOT: key = wxT("SNAPSHOT"); break;
            case WXK_INSERT: key = wxT("INSERT"); break;
            case WXK_HELP: key = wxT("HELP"); break;
            case WXK_NUMPAD0: key = wxT("NUMPAD0"); break;
            case WXK_NUMPAD1: key = wxT("NUMPAD1"); break;
            case WXK_NUMPAD2: key = wxT("NUMPAD2"); break;
            case WXK_NUMPAD3: key = wxT("NUMPAD3"); break;
            case WXK_NUMPAD4: key = wxT("NUMPAD4"); break;
            case WXK_NUMPAD5: key = wxT("NUMPAD5"); break;
            case WXK_NUMPAD6: key = wxT("NUMPAD6"); break;
            case WXK_NUMPAD7: key = wxT("NUMPAD7"); break;
            case WXK_NUMPAD8: key = wxT("NUMPAD8"); break;
            case WXK_NUMPAD9: key = wxT("NUMPAD9"); break;
            case WXK_MULTIPLY: key = wxT("MULTIPLY"); break;
            case WXK_ADD: key = wxT("ADD"); break;
            case WXK_SEPARATOR: key = wxT("SEPARATOR"); break;
            case WXK_SUBTRACT: key = wxT("SUBTRACT"); break;
            case WXK_DECIMAL: key = wxT("DECIMAL"); break;
            case WXK_DIVIDE: key = wxT("DIVIDE"); break;
            case WXK_F1: key = wxT("F1"); break;
            case WXK_F2: key = wxT("F2"); break;
            case WXK_F3: key = wxT("F3"); break;
            case WXK_F4: key = wxT("F4"); break;
            case WXK_F5: key = wxT("F5"); break;
            case WXK_F6: key = wxT("F6"); break;
            case WXK_F7: key = wxT("F7"); break;
            case WXK_F8: key = wxT("F8"); break;
            case WXK_F9: key = wxT("F9"); break;
            case WXK_F10: key = wxT("F10"); break;
            case WXK_F11: key = wxT("F11"); break;
            case WXK_F12: key = wxT("F12"); break;
            case WXK_F13: key = wxT("F13"); break;
            case WXK_F14: key = wxT("F14"); break;
            case WXK_F15: key = wxT("F15"); break;
            case WXK_F16: key = wxT("F16"); break;
            case WXK_F17: key = wxT("F17"); break;
            case WXK_F18: key = wxT("F18"); break;
            case WXK_F19: key = wxT("F19"); break;
            case WXK_F20: key = wxT("F20"); break;
            case WXK_F21: key = wxT("F21"); break;
            case WXK_F22: key = wxT("F22"); break;
            case WXK_F23: key = wxT("F23"); break;
            case WXK_F24: key = wxT("F24"); break;
            case WXK_NUMLOCK: key = wxT("NUMLOCK"); break;
            case WXK_SCROLL: key = wxT("SCROLL"); break;
            case WXK_PAGEUP: key = wxT("PAGEUP"); break;
            case WXK_PAGEDOWN: key = wxT("PAGEDOWN"); break;
            case WXK_NUMPAD_SPACE: key = wxT("NUMPAD_SPACE"); break;
            case WXK_NUMPAD_TAB: key = wxT("NUMPAD_TAB"); break;
            case WXK_NUMPAD_ENTER: key = wxT("NUMPAD_ENTER"); break;
            case WXK_NUMPAD_F1: key = wxT("NUMPAD_F1"); break;
            case WXK_NUMPAD_F2: key = wxT("NUMPAD_F2"); break;
            case WXK_NUMPAD_F3: key = wxT("NUMPAD_F3"); break;
            case WXK_NUMPAD_F4: key = wxT("NUMPAD_F4"); break;
            case WXK_NUMPAD_HOME: key = wxT("NUMPAD_HOME"); break;
            case WXK_NUMPAD_LEFT: key = wxT("NUMPAD_LEFT"); break;
            case WXK_NUMPAD_UP: key = wxT("NUMPAD_UP"); break;
            case WXK_NUMPAD_RIGHT: key = wxT("NUMPAD_RIGHT"); break;
            case WXK_NUMPAD_DOWN: key = wxT("NUMPAD_DOWN"); break;
            case WXK_NUMPAD_PAGEUP: key = wxT("NUMPAD_PAGEUP"); break;
            case WXK_NUMPAD_PAGEDOWN: key = wxT("NUMPAD_PAGEDOWN"); break;
            case WXK_NUMPAD_END: key = wxT("NUMPAD_END"); break;
            case WXK_NUMPAD_BEGIN: key = wxT("NUMPAD_BEGIN"); break;
            case WXK_NUMPAD_INSERT: key = wxT("NUMPAD_INSERT"); break;
            case WXK_NUMPAD_DELETE: key = wxT("NUMPAD_DELETE"); break;
            case WXK_NUMPAD_EQUAL: key = wxT("NUMPAD_EQUAL"); break;
            case WXK_NUMPAD_MULTIPLY: key = wxT("NUMPAD_MULTIPLY"); break;
            case WXK_NUMPAD_ADD: key = wxT("NUMPAD_ADD"); break;
            case WXK_NUMPAD_SEPARATOR: key = wxT("NUMPAD_SEPARATOR"); break;
            case WXK_NUMPAD_SUBTRACT: key = wxT("NUMPAD_SUBTRACT"); break;
            case WXK_NUMPAD_DECIMAL: key = wxT("NUMPAD_DECIMAL"); break;

            default:
            {
               if ( keycode < 128 && wxIsprint((int)keycode) )
                   key.Printf(wxT("'%c'"), (char)keycode);
               else if ( keycode > 0 && keycode < 27 )
                   key.Printf(_("Ctrl-%c"), wxT('A') + keycode - 1);
               else
                   key.Printf(wxT("unknown (%ld)"), keycode);
            }
        }
    }

    wxLogMessage(wxT("%s event: %s (flags = %c%c%c%c)"),
                  name,
                  key.c_str(),
                  event.ControlDown() ? wxT('C') : wxT('-'),
                  event.AltDown() ? wxT('A') : wxT('-'),
                  event.ShiftDown() ? wxT('S') : wxT('-'),
                  event.MetaDown() ? wxT('M') : wxT('-'));
}

void OPJMarkerTree::OnTreeKeyDown(wxTreeEvent& event)
{
    LogKeyEvent(wxT("Tree key down "), event.GetKeyEvent());

    event.Skip();
}*/

/*void OPJMarkerTree::OnBeginDrag(wxTreeEvent& event)
{
    // need to explicitly allow drag
    if ( event.GetItem() != GetRootItem() )
    {
        m_draggedItem = event.GetItem();

        wxLogMessage(wxT("OnBeginDrag: started dragging %s"),
                     GetItemText(m_draggedItem).c_str());

        event.Allow();
    }
    else
    {
        wxLogMessage(wxT("OnBeginDrag: this item can't be dragged."));
    }
}

void OPJMarkerTree::OnEndDrag(wxTreeEvent& event)
{
    wxTreeItemId itemSrc = m_draggedItem,
                 itemDst = event.GetItem();
    m_draggedItem = (wxTreeItemId)0l;

    // where to copy the item?
    if ( itemDst.IsOk() && !ItemHasChildren(itemDst) )
    {
        // copy to the parent then
        itemDst = GetItemParent(itemDst);
    }

    if ( !itemDst.IsOk() )
    {
        wxLogMessage(wxT("OnEndDrag: can't drop here."));

        return;
    }

    wxString text = GetItemText(itemSrc);
    wxLogMessage(wxT("OnEndDrag: '%s' copied to '%s'."),
                 text.c_str(), GetItemText(itemDst).c_str());

    // just do append here - we could also insert it just before/after the item
    // on which it was dropped, but this requires slightly more work... we also
    // completely ignore the client data and icon of the old item but could
    // copy them as well.
    //
    // Finally, we only copy one item here but we might copy the entire tree if
    // we were dragging a folder.
    int image = wxGetApp().ShowImages() ? TreeCtrlIcon_File : -1;
    AppendItem(itemDst, text, image);
}*/

/*void OPJMarkerTree::OnBeginLabelEdit(wxTreeEvent& event)
{
    wxLogMessage(wxT("OnBeginLabelEdit"));

    // for testing, prevent this item's label editing
    wxTreeItemId itemId = event.GetItem();
    if ( IsTestItem(itemId) )
    {
        wxMessageBox(wxT("You can't edit this item."));

        event.Veto();
    }
    else if ( itemId == GetRootItem() )
    {
        // test that it is possible to change the text of the item being edited
        SetItemText(itemId, _T("Editing root item"));
    }
}

void OPJMarkerTree::OnEndLabelEdit(wxTreeEvent& event)
{
    wxLogMessage(wxT("OnEndLabelEdit"));

    // don't allow anything except letters in the labels
    if ( !event.GetLabel().IsWord() )
    {
        wxMessageBox(wxT("The new label should be a single word."));

        event.Veto();
    }
}*/

/*void OPJMarkerTree::OnItemCollapsing(wxTreeEvent& event)
{
    wxLogMessage(wxT("OnItemCollapsing"));

    // for testing, prevent the user from collapsing the first child folder
    wxTreeItemId itemId = event.GetItem();
    if ( IsTestItem(itemId) )
    {
        wxMessageBox(wxT("You can't collapse this item."));

        event.Veto();
    }
}*/

/*void OPJMarkerTree::OnItemActivated(wxTreeEvent& event)
{
    // show some info about this item
    wxTreeItemId itemId = event.GetItem();
    OPJMarkerData *item = (OPJMarkerData *)GetItemData(itemId);

    if ( item != NULL )
    {
        item->ShowInfo(this);
    }

    wxLogMessage(wxT("OnItemActivated"));
}*/

void OPJMarkerTree::OnItemMenu(wxTreeEvent& event)
{
    /*wxTreeItemId itemId = event.GetItem();
    OPJMarkerData *item = itemId.IsOk() ? (OPJMarkerData *)GetItemData(itemId)
                                         : NULL;

    wxLogMessage(wxT("OnItemMenu for item \"%s\""), item ? item->GetDesc()
                                                         : _T(""));*/

	//wxLogMessage(wxT("EEEEEEEEEE"));

    //event.Skip();
}

/*void OPJMarkerTree::OnContextMenu(wxContextMenuEvent& event)
{
    wxPoint pt = event.GetPosition();
    wxTreeItemId item;
    wxLogMessage(wxT("OnContextMenu at screen coords (%i, %i)"), pt.x, pt.y);

    // check if event was generated by keyboard (MSW-specific?)
    if ( pt.x == -1 && pt.y == -1 ) //(this is how MSW indicates it)
    {
        if ( !HasFlag(wxTR_MULTIPLE) )
            item = GetSelection();

        // attempt to guess where to show the menu
        if ( item.IsOk() )
        {
            // if an item was clicked, show menu to the right of it
            wxRect rect;
            GetBoundingRect(item, rect, true );// only the label
            pt = wxPoint(rect.GetRight(), rect.GetTop());
        }
        else
        {
            pt = wxPoint(0, 0);
        }
    }
    else // event was generated by mouse, use supplied coords
    {
        pt = ScreenToClient(pt);
        item = HitTest(pt);
    }

    ShowMenu(item, pt);
}*/

/*void OPJMarkerTree::ShowMenu(wxTreeItemId id, const wxPoint& pt)
{
    wxString title;
    if ( id.IsOk() )
    {
        title << wxT("Menu for ") << GetItemText(id);
    }
    else
    {
        title = wxT("Menu for no particular item");
    }

#if wxUSE_MENUS
    wxMenu menu(title);
    menu.Append(TreeTest_About, wxT("&About..."));
    menu.AppendSeparator();
    menu.Append(TreeTest_Highlight, wxT("&Highlight item"));
    menu.Append(TreeTest_Dump, wxT("&Dump"));

    PopupMenu(&menu, pt);
#endif // wxUSE_MENUS
}*/

/*void OPJMarkerTree::OnItemRClick(wxTreeEvent& event)
{
    wxTreeItemId itemId = event.GetItem();
    OPJMarkerData *item = itemId.IsOk() ? (OPJMarkerData *)GetItemData(itemId)
                                         : NULL;

    wxLogMessage(wxT("Item \"%s\" right clicked"), item ? item->GetDesc()
                                                        : _T(""));

    event.Skip();
}*/

/*
void OPJMarkerTree::OnRMouseDown(wxMouseEvent& event)
{
    wxLogMessage(wxT("Right mouse button down"));

    event.Skip();
}

void OPJMarkerTree::OnRMouseUp(wxMouseEvent& event)
{
    wxLogMessage(wxT("Right mouse button up"));

    event.Skip();
}

void OPJMarkerTree::OnRMouseDClick(wxMouseEvent& event)
{
    wxTreeItemId id = HitTest(event.GetPosition());
    if ( !id )
        wxLogMessage(wxT("No item under mouse"));
    else
    {
        OPJMarkerData *item = (OPJMarkerData *)GetItemData(id);
        if ( item )
            wxLogMessage(wxT("Item '%s' under mouse"), item->GetDesc());
    }

    event.Skip();
}
*/

static inline const wxChar *Bool2String(bool b)
{
    return b ? wxT("") : wxT("not ");
}

void OPJMarkerData::ShowInfo(wxTreeCtrl *tree)
{
    wxLogMessage(wxT("Item '%s': %sselected, %sexpanded, %sbold,\n")
                 wxT("%u children (%u immediately under this item)."),
                 m_desc.c_str(),
                 Bool2String(tree->IsSelected(GetId())),
                 Bool2String(tree->IsExpanded(GetId())),
                 Bool2String(tree->IsBold(GetId())),
                 unsigned(tree->GetChildrenCount(GetId())),
                 unsigned(tree->GetChildrenCount(GetId(), false)));
}


