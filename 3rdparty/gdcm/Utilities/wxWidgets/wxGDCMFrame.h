#ifndef WXGDCMFRAME_H
#define WXGDCMFRAME_H

#include "wxGDCMFrameBase.h"
class vtkImageColorViewer;
class vtkImageViewer;
class vtkGDCMImageReader;
class wxGDCMFrame: public wxGDCMFrameBase
{
public:
    wxGDCMFrame(wxWindow* parent, int id, const wxString& title, const wxPoint& pos=wxDefaultPosition, const wxSize& size=wxDefaultSize, long style=wxDEFAULT_FRAME_STYLE);
    ~wxGDCMFrame();


    void OnQuit( wxCommandEvent& event );
    void OnOpen(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnCloseFrame( wxCloseEvent& event );

private:
    wxString	      directory;
    wxString        filename;
    vtkImageColorViewer *imageViewer;
    //vtkImageViewer *imageViewer;
    vtkGDCMImageReader  *Reader;

    DECLARE_EVENT_TABLE( );
}; // wxGlade: end class


#endif // WXGDCMFRAME_H
