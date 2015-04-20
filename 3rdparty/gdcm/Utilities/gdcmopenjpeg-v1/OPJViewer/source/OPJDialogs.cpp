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

// ----------------------------------------------------------------------------
// OPJDecoderDialog
// ----------------------------------------------------------------------------

IMPLEMENT_CLASS(OPJDecoderDialog, wxPropertySheetDialog)

BEGIN_EVENT_TABLE(OPJDecoderDialog, wxPropertySheetDialog)
	EVT_CHECKBOX(OPJDECO_ENABLEDECO, OPJDecoderDialog::OnEnableDeco)
#ifdef USE_JPWL
	EVT_CHECKBOX(OPJDECO_ENABLEJPWL, OPJDecoderDialog::OnEnableJPWL)
#endif // USE_JPWL
END_EVENT_TABLE()

OPJDecoderDialog::OPJDecoderDialog(wxWindow* win, int dialogType)
{
	SetExtraStyle(wxDIALOG_EX_CONTEXTHELP|wxWS_EX_VALIDATE_RECURSIVELY);

	Create(win, wxID_ANY, wxT("Decoder settings"),
		wxDefaultPosition, wxDefaultSize,
		wxDEFAULT_DIALOG_STYLE| (int) wxPlatform::IfNot(wxOS_WINDOWS_CE, wxRESIZE_BORDER)
		);

	CreateButtons(wxOK | wxCANCEL | (int)wxPlatform::IfNot(wxOS_WINDOWS_CE, wxHELP));

	m_settingsNotebook = GetBookCtrl();

	wxPanel* mainSettings = CreateMainSettingsPage(m_settingsNotebook);
	wxPanel* jpeg2000Settings = CreatePart1SettingsPage(m_settingsNotebook);
	if (!wxGetApp().m_enabledeco)
		jpeg2000Settings->Enable(false);
	wxPanel* mjpeg2000Settings = CreatePart3SettingsPage(m_settingsNotebook);
	if (!wxGetApp().m_enabledeco)
		mjpeg2000Settings->Enable(false);
#ifdef USE_JPWL
	wxPanel* jpwlSettings = CreatePart11SettingsPage(m_settingsNotebook);
	if (!wxGetApp().m_enabledeco)
		jpwlSettings->Enable(false);
#endif // USE_JPWL

	m_settingsNotebook->AddPage(mainSettings, wxT("Display"), false);
	m_settingsNotebook->AddPage(jpeg2000Settings, wxT("JPEG 2000"), false);
	m_settingsNotebook->AddPage(mjpeg2000Settings, wxT("MJPEG 2000"), false);
#ifdef USE_JPWL
	m_settingsNotebook->AddPage(jpwlSettings, wxT("JPWL"), false);
#endif // USE_JPWL

	LayoutDialog();
}

OPJDecoderDialog::~OPJDecoderDialog()
{
}

wxPanel* OPJDecoderDialog::CreateMainSettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

		// sub top sizer
		wxBoxSizer *subtopSizer = new wxBoxSizer(wxVERTICAL);

		// add decoding enabling check box
		subtopSizer->Add(
			m_enabledecoCheck = new wxCheckBox(panel, OPJDECO_ENABLEDECO, wxT("Enable decoding"), wxDefaultPosition, wxDefaultSize),
			0, wxGROW | wxALL, 5);
		m_enabledecoCheck->SetValue(wxGetApp().m_enabledeco);

		// add parsing enabling check box
		subtopSizer->Add(
			m_enableparseCheck = new wxCheckBox(panel, OPJDECO_ENABLEPARSE, wxT("Enable parsing"), wxDefaultPosition, wxDefaultSize),
			0, wxGROW | wxALL, 5);
		m_enableparseCheck->SetValue(wxGetApp().m_enableparse);

			// resize settings, column
			wxString choices[] = {wxT("Don't resize"), wxT("Low quality"), wxT("High quality")};
			m_resizeBox = new wxRadioBox(panel, OPJDECO_RESMETHOD,
				wxT("Resize method"),
				wxDefaultPosition, wxDefaultSize,
				WXSIZEOF(choices),
				choices,
				1,
				wxRA_SPECIFY_ROWS);
			m_resizeBox->SetSelection(wxGetApp().m_resizemethod + 1);

		subtopSizer->Add(m_resizeBox, 0, wxGROW | wxALL, 5);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}

wxPanel* OPJDecoderDialog::CreatePart3SettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

	// add some space
	//topSizer->AddSpacer(5);

		// sub top sizer
		wxBoxSizer *subtopSizer = new wxBoxSizer(wxVERTICAL);

			// frame settings, column
			wxStaticBox* frameBox = new wxStaticBox(panel, wxID_ANY, wxT("Frame"));
			wxBoxSizer* frameSizer = new wxStaticBoxSizer(frameBox, wxVERTICAL);

				// selected frame number, row
				wxBoxSizer* framenumSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				framenumSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Displayed frame:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

				// add some horizontal space
				framenumSizer->Add(5, 5, 1, wxALL, 0);

				// add the value control
				framenumSizer->Add(
					m_framenumCtrl = new wxSpinCtrl(panel, OPJDECO_FRAMENUM,
								wxString::Format(wxT("%d"), wxGetApp().m_framenum),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								1, 100000, wxGetApp().m_framenum),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 5);

			frameSizer->Add(framenumSizer, 0, wxGROW | wxALL, 5);

		subtopSizer->Add(frameSizer, 0, wxGROW | wxALL, 5);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}

wxPanel* OPJDecoderDialog::CreatePart1SettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

	// add some space
	//topSizer->AddSpacer(5);

		// sub top sizer
		wxBoxSizer *subtopSizer = new wxBoxSizer(wxVERTICAL);

			// resolutions settings, column
			wxStaticBox* resolutionBox = new wxStaticBox(panel, wxID_ANY, wxT("Resolutions"));
			wxBoxSizer* resolutionSizer = new wxStaticBoxSizer(resolutionBox, wxVERTICAL);

				// reduce factor sizer, row
				wxBoxSizer* reduceSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				reduceSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Reduce factor:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

				// add some horizontal space
				reduceSizer->Add(5, 5, 1, wxALL, 0);

				// add the value control
				reduceSizer->Add(
					m_reduceCtrl = new wxSpinCtrl(panel, OPJDECO_REDUCEFACTOR,
					wxString::Format(wxT("%d"), wxGetApp().m_reducefactor),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								0, 10000, wxGetApp().m_reducefactor),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 5);

			resolutionSizer->Add(reduceSizer, 0, wxGROW | wxALL, 5);

		subtopSizer->Add(resolutionSizer, 0, wxGROW | wxALL, 5);

			// quality layer settings, column
			wxStaticBox* layerBox = new wxStaticBox(panel, wxID_ANY, wxT("Layers"));
			wxBoxSizer* layerSizer = new wxStaticBoxSizer(layerBox, wxVERTICAL);

				// quality layers sizer, row
				wxBoxSizer* qualitySizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				qualitySizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Quality layers:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

				// add some horizontal space
				qualitySizer->Add(5, 5, 1, wxALL, 0);

				// add the value control
				qualitySizer->Add(
					m_layerCtrl = new wxSpinCtrl(panel, OPJDECO_QUALITYLAYERS,
								wxString::Format(wxT("%d"), wxGetApp().m_qualitylayers),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								0, 100000, wxGetApp().m_qualitylayers),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 5);

			layerSizer->Add(qualitySizer, 0, wxGROW | wxALL, 5);

		subtopSizer->Add(layerSizer, 0, wxGROW | wxALL, 5);

			// component settings, column
			wxStaticBox* compoBox = new wxStaticBox(panel, wxID_ANY, wxT("Components"));
			wxBoxSizer* compoSizer = new wxStaticBoxSizer(compoBox, wxVERTICAL);

				// quality layers sizer, row
				wxBoxSizer* numcompsSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				numcompsSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Component displayed:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

				// add some horizontal space
				numcompsSizer->Add(5, 5, 1, wxALL, 0);

				// add the value control
				numcompsSizer->Add(
					m_numcompsCtrl = new wxSpinCtrl(panel, OPJDECO_NUMCOMPS,
								wxString::Format(wxT("%d"), wxGetApp().m_components),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								0, 100000, wxGetApp().m_components),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 5);
				m_numcompsCtrl->Enable(true);

			compoSizer->Add(numcompsSizer, 0, wxGROW | wxALL, 5);

		subtopSizer->Add(compoSizer, 0, wxGROW | wxALL, 5);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}

#ifdef USE_JPWL
wxPanel* OPJDecoderDialog::CreatePart11SettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

	// add some space
	//topSizer->AddSpacer(5);

		// sub top sizer
		wxBoxSizer *subtopSizer = new wxBoxSizer(wxVERTICAL);

		// add JPWL enabling check box
		subtopSizer->Add(
			m_enablejpwlCheck = new wxCheckBox(panel, OPJDECO_ENABLEJPWL, wxT("Enable JPWL"), wxDefaultPosition, wxDefaultSize),
			0, wxGROW | wxALL, 5);
		m_enablejpwlCheck->SetValue(wxGetApp().m_enablejpwl);

			// component settings, column
			wxStaticBox* compoBox = new wxStaticBox(panel, wxID_ANY, wxT("Components"));
			wxBoxSizer* compoSizer = new wxStaticBoxSizer(compoBox, wxVERTICAL);

				// expected components sizer, row
				wxBoxSizer* expcompsSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				expcompsSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Expected comps.:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

				// add some horizontal space
				expcompsSizer->Add(5, 5, 1, wxALL, 0);

				// add the value control
				expcompsSizer->Add(
					m_expcompsCtrl = new wxSpinCtrl(panel, OPJDECO_EXPCOMPS,
								wxString::Format(wxT("%d"), wxGetApp().m_expcomps),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								1, 100000, wxGetApp().m_expcomps),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 5);
				m_expcompsCtrl->Enable(wxGetApp().m_enablejpwl);

			compoSizer->Add(expcompsSizer, 0, wxGROW | wxALL, 5);

		subtopSizer->Add(compoSizer, 0, wxGROW | wxALL, 5);

			// tiles settings, column
			wxStaticBox* tileBox = new wxStaticBox(panel, wxID_ANY, wxT("Tiles"));
			wxBoxSizer* tileSizer = new wxStaticBoxSizer(tileBox, wxVERTICAL);

				// maximum tiles sizer, row
				wxBoxSizer* maxtileSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				maxtileSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Max. no. of tiles:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

				// add some horizontal space
				maxtileSizer->Add(5, 5, 1, wxALL, 0);

				// add the value control
				maxtileSizer->Add(
					m_maxtilesCtrl = new wxSpinCtrl(panel, OPJDECO_MAXTILES,
								wxString::Format(wxT("%d"), wxGetApp().m_maxtiles),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								1, 100000, wxGetApp().m_maxtiles),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 5);
				m_maxtilesCtrl->Enable(wxGetApp().m_enablejpwl);

			tileSizer->Add(maxtileSizer, 0, wxGROW | wxALL, 5);

		subtopSizer->Add(tileSizer, 0, wxGROW | wxALL, 5);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}

void OPJDecoderDialog::OnEnableDeco(wxCommandEvent& event)
{
	size_t pp;

	if (event.IsChecked()) {
		wxLogMessage(wxT("Decoding enabled"));
		m_resizeBox->Enable(true);
		// enable all tabs except ourselves
		for (pp = 0; pp < m_settingsNotebook->GetPageCount(); pp++) {
			if (m_settingsNotebook->GetPageText(pp) != wxT("Display"))
				m_settingsNotebook->GetPage(pp)->Enable(true);
		}
	} else {
		wxLogMessage(wxT("Decoding disabled"));
		m_resizeBox->Enable(false);
		// disable all tabs except ourselves
		for (pp = 0; pp < m_settingsNotebook->GetPageCount(); pp++) {
			if (m_settingsNotebook->GetPageText(pp) != wxT("Display"))
				m_settingsNotebook->GetPage(pp)->Enable(false);
		}
	}

}

void OPJDecoderDialog::OnEnableJPWL(wxCommandEvent& event)
{
	if (event.IsChecked()) {
		wxLogMessage(wxT("JPWL enabled"));
		m_expcompsCtrl->Enable(true);
		m_maxtilesCtrl->Enable(true);
	} else {
		wxLogMessage(wxT("JPWL disabled"));
		m_expcompsCtrl->Enable(false);
		m_maxtilesCtrl->Enable(false);
	}

}

#endif // USE_JPWL




// ----------------------------------------------------------------------------
// OPJEncoderDialog
// ----------------------------------------------------------------------------

IMPLEMENT_CLASS(OPJEncoderDialog, wxPropertySheetDialog)

BEGIN_EVENT_TABLE(OPJEncoderDialog, wxPropertySheetDialog)
	EVT_CHECKBOX(OPJENCO_ENABLECOMM, OPJEncoderDialog::OnEnableComm)
	EVT_CHECKBOX(OPJENCO_ENABLEINDEX, OPJEncoderDialog::OnEnableIdx)
	EVT_CHECKBOX(OPJENCO_ENABLEPOC, OPJEncoderDialog::OnEnablePoc)
	EVT_RADIOBUTTON(OPJENCO_RATERADIO, OPJEncoderDialog::OnRadioQualityRate)
	EVT_RADIOBUTTON(OPJENCO_QUALITYRADIO, OPJEncoderDialog::OnRadioQualityRate)
#ifdef USE_JPWL
	EVT_CHECKBOX(OPJENCO_ENABLEJPWL, OPJEncoderDialog::OnEnableJPWL)
	EVT_CHOICE(OPJENCO_HPROT, OPJEncoderDialog::OnHprotSelect)
	EVT_CHOICE(OPJENCO_PPROT, OPJEncoderDialog::OnPprotSelect)
	EVT_CHOICE(OPJENCO_SENSI, OPJEncoderDialog::OnSensiSelect)
#endif // USE_JPWL
END_EVENT_TABLE()

OPJEncoderDialog::OPJEncoderDialog(wxWindow* win, int dialogType)
{
	SetExtraStyle(wxDIALOG_EX_CONTEXTHELP|wxWS_EX_VALIDATE_RECURSIVELY);

	Create(win, wxID_ANY, wxT("Encoder settings"),
		wxDefaultPosition, wxDefaultSize,
		wxDEFAULT_DIALOG_STYLE| (int) wxPlatform::IfNot(wxOS_WINDOWS_CE, wxRESIZE_BORDER)
		);

	CreateButtons(wxOK | wxCANCEL | (int)wxPlatform::IfNot(wxOS_WINDOWS_CE, wxHELP));

	m_settingsNotebook = GetBookCtrl();

	wxPanel* jpeg2000_1Settings = CreatePart1_1SettingsPage(m_settingsNotebook);
	wxPanel* jpeg2000_2Settings = CreatePart1_2SettingsPage(m_settingsNotebook);
	wxPanel* mainSettings = CreateMainSettingsPage(m_settingsNotebook);
#ifdef USE_JPWL
	wxPanel* jpwlSettings = CreatePart11SettingsPage(m_settingsNotebook);
#endif // USE_JPWL

#ifdef USE_JPWL
	m_settingsNotebook->AddPage(jpwlSettings, wxT("JPWL"), false);
#endif // USE_JPWL
	m_settingsNotebook->AddPage(jpeg2000_1Settings, wxT("JPEG 2000 - 1"), false);
	m_settingsNotebook->AddPage(jpeg2000_2Settings, wxT("JPEG 2000 - 2"), false);
	m_settingsNotebook->AddPage(mainSettings, wxT("General"), false);

	LayoutDialog();
}

OPJEncoderDialog::~OPJEncoderDialog()
{
}

wxPanel* OPJEncoderDialog::CreateMainSettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

		// sub top sizer
		wxBoxSizer *subtopSizer = new wxBoxSizer(wxVERTICAL);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}

#ifdef USE_JPWL
wxPanel* OPJEncoderDialog::CreatePart11SettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);
	int specno;

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

		// add JPWL enabling check box
		topSizer->Add(
			m_enablejpwlCheck = new wxCheckBox(panel, OPJENCO_ENABLEJPWL, wxT("Enable JPWL"),
			wxDefaultPosition, wxDefaultSize),
			0, wxGROW | wxALL | wxALIGN_CENTER, 5);
		m_enablejpwlCheck->SetValue(wxGetApp().m_enablejpwle);

		// sub top sizer
		wxFlexGridSizer *subtopSizer = new wxFlexGridSizer(2, 3, 3);

			// header settings, column
			wxStaticBox* headerBox = new wxStaticBox(panel, wxID_ANY, wxT("Header protection"));
			wxBoxSizer* headerSizer = new wxStaticBoxSizer(headerBox, wxVERTICAL);

				// info sizer, row
				wxBoxSizer* info1Sizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				info1Sizer->Add(new wxStaticText(panel, wxID_ANY,
								wxT("Type")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 1);

				// add some horizontal space
				info1Sizer->Add(3, 3, 1, wxALL, 0);

				// add some text
				info1Sizer->Add(new wxStaticText(panel, wxID_ANY,
								wxT("Tile part")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 1);

			headerSizer->Add(info1Sizer, 0, wxGROW | wxALL, 0);

			// specify specs
			wxString hprotvalues[] = {wxT("None"), wxT("Pred."), wxT("CRC16"), wxT("CRC32"),
				wxT("RS37"), wxT("RS38"), wxT("RS40"), wxT("RS43"), wxT("RS45"), wxT("RS48"),
				wxT("RS51"), wxT("RS53"), wxT("RS56"), wxT("RS64"), wxT("RS75"), wxT("RS80"),
				wxT("RS85"), wxT("RS96"), wxT("RS112"), wxT("RS128")};
			for (specno = 0; specno < MYJPWL_MAX_NO_TILESPECS; specno++) {

					// tile+hprot sizer, row
					wxBoxSizer* tilehprotSizer = new wxBoxSizer(wxHORIZONTAL);

					// add the value selection
					tilehprotSizer->Add(
						m_hprotChoice[specno] = new wxChoice(panel, OPJENCO_HPROT,
							wxDefaultPosition, wxSize(60, wxDefaultCoord),
							WXSIZEOF(hprotvalues), hprotvalues),
						0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 1);
					m_hprotChoice[specno]->SetSelection(wxGetApp().m_hprotsel[specno]);

					// add some horizontal space
					tilehprotSizer->Add(3, 3, 1, wxALL, 0);

					// add the value control
					tilehprotSizer->Add(
						m_htileCtrl[specno] = new wxSpinCtrl(panel, OPJENCO_HTILE,
							wxString::Format(wxT("%d"), wxGetApp().m_htileval[specno]),
							wxDefaultPosition, wxSize(45, wxDefaultCoord),
							wxSP_ARROW_KEYS,
							0, JPWL_MAXIMUM_TILES - 1, 0),
						0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 1);

				headerSizer->Add(tilehprotSizer, 0, wxGROW | wxALL, 0);
			}

			wxCommandEvent event1;
			OnHprotSelect(event1);

		subtopSizer->Add(headerSizer, 0, wxGROW | wxALL, 3);

			// packet settings, column
			wxStaticBox* packetBox = new wxStaticBox(panel, wxID_ANY, wxT("Packet protection"));
			wxBoxSizer* packetSizer = new wxStaticBoxSizer(packetBox, wxVERTICAL);

				// info sizer, row
				wxBoxSizer* info2Sizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				info2Sizer->Add(new wxStaticText(panel, wxID_ANY,
								wxT("Type")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 1);

				// add some horizontal space
				info2Sizer->Add(3, 3, 1, wxALL, 0);

				// add some text
				info2Sizer->Add(new wxStaticText(panel, wxID_ANY,
								wxT("Tile part")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 1);

				// add some horizontal space
				info2Sizer->Add(3, 3, 1, wxALL, 0);

				// add some text
				info2Sizer->Add(new wxStaticText(panel, wxID_ANY,
								wxT("Packet")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 1);

			packetSizer->Add(info2Sizer, 0, wxGROW | wxALL, 0);

			// specify specs
			wxString pprotvalues[] = {wxT("None"), wxT("Pred."), wxT("CRC16"), wxT("CRC32"),
				wxT("RS37"), wxT("RS38"), wxT("RS40"), wxT("RS43"), wxT("RS45"), wxT("RS48"),
				wxT("RS51"), wxT("RS53"), wxT("RS56"), wxT("RS64"), wxT("RS75"), wxT("RS80"),
				wxT("RS85"), wxT("RS96"), wxT("RS112"), wxT("RS128")};
			for (specno = 0; specno < MYJPWL_MAX_NO_TILESPECS; specno++) {

					// tile+pprot sizer, row
					wxBoxSizer* tilepprotSizer = new wxBoxSizer(wxHORIZONTAL);

					// add the value selection
					tilepprotSizer->Add(
						m_pprotChoice[specno] = new wxChoice(panel, OPJENCO_PPROT,
							wxDefaultPosition, wxSize(60, wxDefaultCoord),
							WXSIZEOF(pprotvalues), pprotvalues),
						0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 1);
					m_pprotChoice[specno]->SetSelection(wxGetApp().m_pprotsel[specno]);

					// add some horizontal space
					tilepprotSizer->Add(3, 3, 1, wxALL, 0);

					// add the value control
					tilepprotSizer->Add(
						m_ptileCtrl[specno] = new wxSpinCtrl(panel, OPJENCO_PTILE,
							wxString::Format(wxT("%d"), wxGetApp().m_ptileval[specno]),
							wxDefaultPosition, wxSize(45, wxDefaultCoord),
							wxSP_ARROW_KEYS,
							0, JPWL_MAXIMUM_TILES - 1, 0),
						0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 1);

					// add some horizontal space
					tilepprotSizer->Add(3, 3, 1, wxALL, 0);

					// add the value control
					tilepprotSizer->Add(
						m_ppackCtrl[specno] = new wxSpinCtrl(panel, OPJENCO_PPACK,
							wxString::Format(wxT("%d"), wxGetApp().m_ppackval[specno]),
							wxDefaultPosition, wxSize(50, wxDefaultCoord),
							wxSP_ARROW_KEYS,
							0, 2047, 0),
						0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 1);

				packetSizer->Add(tilepprotSizer, 0, wxGROW | wxALL, 0);
			}

			wxCommandEvent event2;
			OnPprotSelect(event2);

		subtopSizer->Add(packetSizer, 0, wxGROW | wxALL, 3);

			// sensitivity settings, column
			wxStaticBox* sensiBox = new wxStaticBox(panel, wxID_ANY, wxT("Sensitivity"));
			wxBoxSizer* sensiSizer = new wxStaticBoxSizer(sensiBox, wxVERTICAL);

				// info sizer, row
				wxBoxSizer* info3Sizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				info3Sizer->Add(new wxStaticText(panel, wxID_ANY,
								wxT("Type")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 1);

				// add some horizontal space
				info3Sizer->Add(3, 3, 1, wxALL, 0);

				// add some text
				info3Sizer->Add(new wxStaticText(panel, wxID_ANY,
								wxT("Tile part")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 1);

			sensiSizer->Add(info3Sizer, 0, wxGROW | wxALL, 0);

			// specify specs
			wxString sensivalues[] = {wxT("None"), wxT("RELATIVE ERROR"), wxT("MSE"),
				wxT("MSE REDUCTION"), wxT("PSNR INCREMENT"), wxT("MAXERR"), wxT("TSE")};
			for (specno = 0; specno < MYJPWL_MAX_NO_TILESPECS; specno++) {

					// tile+sensi sizer, row
					wxBoxSizer* tilesensiSizer = new wxBoxSizer(wxHORIZONTAL);

					// add the value selection
					tilesensiSizer->Add(
						m_sensiChoice[specno] = new wxChoice(panel, OPJENCO_SENSI,
							wxDefaultPosition, wxSize(110, wxDefaultCoord),
							WXSIZEOF(sensivalues), sensivalues),
						0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 1);
					m_sensiChoice[specno]->SetSelection(wxGetApp().m_sensisel[specno]);

					// add some horizontal space
					tilesensiSizer->Add(3, 3, 1, wxALL, 0);

					// add the value control
					tilesensiSizer->Add(
						m_stileCtrl[specno] = new wxSpinCtrl(panel, OPJENCO_STILE,
							wxString::Format(wxT("%d"), wxGetApp().m_stileval[specno]),
							wxDefaultPosition, wxSize(45, wxDefaultCoord),
							wxSP_ARROW_KEYS,
							0, JPWL_MAXIMUM_TILES - 1, 0),
						0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 1);

				sensiSizer->Add(tilesensiSizer, 0, wxGROW | wxALL, 0);
			}

			wxCommandEvent event3;
			OnSensiSelect(event3);

		subtopSizer->Add(sensiSizer, 0, wxGROW | wxALL, 3);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}
#endif // USE_JPWL

wxPanel* OPJEncoderDialog::CreatePart1_1SettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

	// add some space
	//topSizer->AddSpacer(5);

		// sub top sizer
		wxFlexGridSizer *subtopSizer = new wxFlexGridSizer(2, 3, 3);

			// image settings, column
			wxStaticBox* imageBox = new wxStaticBox(panel, wxID_ANY, wxT("Image"));
			wxBoxSizer* imageSizer = new wxStaticBoxSizer(imageBox, wxVERTICAL);

				// subsampling factor sizer, row
				wxBoxSizer* subsSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				subsSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Subsampling:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				subsSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				subsSizer->Add(
					m_subsamplingCtrl = new wxTextCtrl(panel, OPJENCO_SUBSAMPLING,
								wxGetApp().m_subsampling,
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			imageSizer->Add(subsSizer, 0, wxGROW | wxALL, 3);

				// origin sizer, row
				wxBoxSizer* imorigSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				imorigSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Origin:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				imorigSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				imorigSizer->Add(
					m_originCtrl = new wxTextCtrl(panel, OPJENCO_IMORIG,
								wxGetApp().m_origin,
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			imageSizer->Add(imorigSizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(imageSizer, 0, wxGROW | wxALL, 3);

			// layer settings, column
			wxStaticBox* layerBox = new wxStaticBox(panel, wxID_ANY, wxT("Layers/compression"));
			wxBoxSizer* layerSizer = new wxStaticBoxSizer(layerBox, wxVERTICAL);

				// rate factor sizer, row
				wxBoxSizer* rateSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				/*rateSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Rate values:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);*/

				// add the radio button
				rateSizer->Add(
					m_rateRadio = new wxRadioButton(panel, OPJENCO_RATERADIO, wxT("&Rate values"),
								wxDefaultPosition, wxDefaultSize,
								wxRB_GROUP),
								0, wxALL | wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL
								);
				m_rateRadio->SetValue(!(wxGetApp().m_enablequality));

				// add some horizontal space
				rateSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				rateSizer->Add(
					m_rateCtrl = new wxTextCtrl(panel, OPJENCO_RATEFACTOR,
								wxGetApp().m_rates,
								wxDefaultPosition, wxSize(100, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);
				if (wxGetApp().m_enablequality == true)
					m_rateCtrl->Enable(false);

			layerSizer->Add(rateSizer, 0, wxGROW | wxALL, 3);

				// quality factor sizer, row
				wxBoxSizer* qualitySizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				/*qualitySizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Quality values:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);*/

				// add the radio button
				qualitySizer->Add(
					m_qualityRadio = new wxRadioButton(panel, OPJENCO_QUALITYRADIO, wxT("&Quality values"),
								wxDefaultPosition, wxDefaultSize),
								0, wxALL | wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL
								);
				m_qualityRadio->SetValue(wxGetApp().m_enablequality);

				// add some horizontal space
				qualitySizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				qualitySizer->Add(
					m_qualityCtrl = new wxTextCtrl(panel, OPJENCO_QUALITYFACTOR,
								wxGetApp().m_quality,
								wxDefaultPosition, wxSize(100, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);
				if (wxGetApp().m_enablequality == false)
					m_qualityCtrl->Enable(false);

			layerSizer->Add(qualitySizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(layerSizer, 0, wxGROW | wxALL, 3);

			// wavelet settings, column
			wxStaticBox* transformBox = new wxStaticBox(panel, wxID_ANY, wxT("Transforms"));
			wxBoxSizer* transformSizer = new wxStaticBoxSizer(transformBox, wxVERTICAL);

			// multiple component check box
			transformSizer->Add(
				m_mctCheck = new wxCheckBox(panel, OPJENCO_ENABLEMCT, wxT("Multiple component"),
				wxDefaultPosition, wxDefaultSize),
				0, wxGROW | wxALL, 3);
			m_mctCheck->SetValue(wxGetApp().m_multicomp);

			// irreversible wavelet check box
			transformSizer->Add(
				m_irrevCheck = new wxCheckBox(panel, OPJENCO_ENABLEIRREV, wxT("Irreversible wavelet"),
				wxDefaultPosition, wxDefaultSize),
				0, wxGROW | wxALL, 3);
			m_irrevCheck->SetValue(wxGetApp().m_irreversible);

				// resolution number sizer, row
				wxBoxSizer* resnumSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				resnumSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Resolutions:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				resnumSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				resnumSizer->Add(
					m_resolutionsCtrl = new wxSpinCtrl(panel, OPJENCO_RESNUMBER,
								wxString::Format(wxT("%d"), wxGetApp().m_resolutions),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								1, 256, 6),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			transformSizer->Add(resnumSizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(transformSizer, 0, wxGROW | wxALL, 3);

			// codestream settings, column
			wxStaticBox* codestreamBox = new wxStaticBox(panel, wxID_ANY, wxT("Codestream"));
			wxBoxSizer* codestreamSizer = new wxStaticBoxSizer(codestreamBox, wxVERTICAL);

				// codeblock sizer, row
				wxBoxSizer* codeblockSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				codeblockSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Codeblocks size:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				codeblockSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				codeblockSizer->Add(
					m_cbsizeCtrl = new wxTextCtrl(panel, OPJENCO_CODEBLOCKSIZE,
								wxGetApp().m_cbsize,
								wxDefaultPosition, wxSize(100, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			codestreamSizer->Add(codeblockSizer, 0, wxGROW | wxALL, 3);

				// precinct sizer, row
				wxBoxSizer* precinctSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				precinctSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Precincts size:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				precinctSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				precinctSizer->Add(
					m_prsizeCtrl = new wxTextCtrl(panel, OPJENCO_PRECINCTSIZE,
								wxGetApp().m_prsize,
								wxDefaultPosition, wxSize(100, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			codestreamSizer->Add(precinctSizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(codestreamSizer, 0, wxGROW | wxALL, 3);

			// tile settings, column
			wxStaticBox* tileBox = new wxStaticBox(panel, wxID_ANY, wxT("Tiles"));
			wxBoxSizer* tileSizer = new wxStaticBoxSizer(tileBox, wxVERTICAL);

				// tile size sizer, row
				wxBoxSizer* tilesizeSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				tilesizeSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Size:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				tilesizeSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				tilesizeSizer->Add(
					m_tsizeCtrl = new wxTextCtrl(panel, OPJENCO_TILESIZE,
								wxGetApp().m_tsize,
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			tileSizer->Add(tilesizeSizer, 0, wxGROW | wxALL, 3);

				// tile origin sizer, row
				wxBoxSizer* tilorigSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				tilorigSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Origin:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				tilorigSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				tilorigSizer->Add(
					m_toriginCtrl = new wxTextCtrl(panel, OPJENCO_TILORIG,
								wxGetApp().m_torigin,
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			tileSizer->Add(tilorigSizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(tileSizer, 0, wxGROW | wxALL, 3);

			// progression and profile settings, column
			wxString choices[] = {wxT("LRCP"), wxT("RLCP"), wxT("RPCL"), wxT("PCRL"), wxT("CPRL"),
				wxT("DCI2K24"), wxT("DCI2K48"), wxT("DCI4K")};
			progressionBox = new wxRadioBox(panel, OPJENCO_PROGRESSION,
				wxT("Progression order/profile"),
				wxDefaultPosition, wxDefaultSize,
				WXSIZEOF(choices),
				choices,
				3,
				wxRA_SPECIFY_COLS);
			progressionBox->SetSelection(wxGetApp().m_progression);

		subtopSizer->Add(progressionBox, 0, wxGROW | wxALL, 3);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}

wxPanel* OPJEncoderDialog::CreatePart1_2SettingsPage(wxWindow* parent)
{
    wxPanel* panel = new wxPanel(parent, wxID_ANY);

	// top sizer
    wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);

	// add some space
	//topSizer->AddSpacer(5);

		// sub top sizer
		wxFlexGridSizer *subtopSizer = new wxFlexGridSizer(2, 3, 3);
			
			// resilience settings, column
			wxStaticBox* resilBox = new wxStaticBox(panel, wxID_ANY, wxT("Error resilience"));
			wxBoxSizer* resilSizer = new wxStaticBoxSizer(resilBox, wxVERTICAL);

				// resil2 sizer, row
				wxBoxSizer* resil2Sizer = new wxBoxSizer(wxHORIZONTAL);

				// SOP check box
				resil2Sizer->Add(
					m_sopCheck = new wxCheckBox(panel, OPJENCO_ENABLESOP, wxT("SOP"),
						wxDefaultPosition, wxDefaultSize),
						0, wxGROW | wxALL, 3);
				m_sopCheck->SetValue(wxGetApp().m_enablesop);

				// EPH check box
				resil2Sizer->Add(
					m_ephCheck = new wxCheckBox(panel, OPJENCO_ENABLEEPH, wxT("EPH"),
						wxDefaultPosition, wxDefaultSize),
						0, wxGROW | wxALL, 3);
				m_ephCheck->SetValue(wxGetApp().m_enableeph);

			resilSizer->Add(resil2Sizer, 0, wxGROW | wxALL, 3);

			// separation
			resilSizer->Add(new wxStaticLine(panel, wxID_ANY), 0, wxEXPAND | wxLEFT | wxRIGHT, 3);

				// resil3 sizer, row
				wxFlexGridSizer* resil3Sizer = new wxFlexGridSizer(3, 3, 3);

				// BYPASS check box
				resil3Sizer->Add(
					m_enablebypassCheck = new wxCheckBox(panel, OPJENCO_ENABLEBYPASS, wxT("BYPASS"),
					wxDefaultPosition, wxDefaultSize),
					0, wxGROW | wxALL, 3);
				m_enablebypassCheck->SetValue(wxGetApp().m_enablebypass);

				// RESET check box
				resil3Sizer->Add(
					m_enableresetCheck = new wxCheckBox(panel, OPJENCO_ENABLERESET, wxT("RESET"),
					wxDefaultPosition, wxDefaultSize),
					0, wxGROW | wxALL, 3);
				m_enableresetCheck->SetValue(wxGetApp().m_enablereset);

				// RESTART check box
				resil3Sizer->Add(
					m_enablerestartCheck = new wxCheckBox(panel, OPJENCO_ENABLERESTART, wxT("RESTART"),
					wxDefaultPosition, wxDefaultSize),
					0, wxGROW | wxALL, 3);
				m_enablerestartCheck->SetValue(wxGetApp().m_enablerestart);

				// VSC check box
				resil3Sizer->Add(
					m_enablevscCheck = new wxCheckBox(panel, OPJENCO_ENABLEVSC, wxT("VSC"),
					wxDefaultPosition, wxDefaultSize),
					0, wxGROW | wxALL, 3);
				m_enablevscCheck->SetValue(wxGetApp().m_enablevsc);

				// ERTERM check box
				resil3Sizer->Add(
					m_enableertermCheck = new wxCheckBox(panel, OPJENCO_ENABLEERTERM, wxT("ERTERM"),
					wxDefaultPosition, wxDefaultSize),
					0, wxGROW | wxALL, 3);
				m_enableertermCheck->SetValue(wxGetApp().m_enableerterm);

				// SEGMARK check box
				resil3Sizer->Add(
					m_enablesegmarkCheck = new wxCheckBox(panel, OPJENCO_ENABLESEGMARK, wxT("SEGMARK"),
					wxDefaultPosition, wxDefaultSize),
					0, wxGROW | wxALL, 3);
				m_enablesegmarkCheck->SetValue(wxGetApp().m_enablesegmark);

			resilSizer->Add(resil3Sizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(resilSizer, 0, wxGROW | wxALL, 3);

			// ROI settings, column
			wxStaticBox* roiBox = new wxStaticBox(panel, wxID_ANY, wxT("Region Of Interest"));
			wxBoxSizer* roiSizer = new wxStaticBoxSizer(roiBox, wxVERTICAL);

				// component number sizer, row
				wxBoxSizer* roicompSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				roicompSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Component:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				roicompSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				roicompSizer->Add(
					/*m_layerCtrl =*/ new wxSpinCtrl(panel, OPJENCO_ROICOMP,
								wxT("0"),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								0, 256, 0),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			roiSizer->Add(roicompSizer, 0, wxGROW | wxALL, 3);

				// upshift sizer, row
				wxBoxSizer* roishiftSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				roishiftSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Upshift:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				roishiftSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				roishiftSizer->Add(
					/*m_layerCtrl =*/ new wxSpinCtrl(panel, OPJENCO_ROISHIFT,
								wxT("0"),
								wxDefaultPosition, wxSize(80, wxDefaultCoord),
								wxSP_ARROW_KEYS,
								0, 37, 0),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);

			roiSizer->Add(roishiftSizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(roiSizer, 0, wxGROW | wxALL, 3);

			// POC settings, column
			wxStaticBox* pocBox = new wxStaticBox(panel, wxID_ANY, wxT("POC"));
			wxBoxSizer* pocSizer = new wxStaticBoxSizer(pocBox, wxVERTICAL);

			// POC check box
			pocSizer->Add(
				m_enablepocCheck = new wxCheckBox(panel, OPJENCO_ENABLEPOC, wxT("Enabled (tn=rs,cs,le,re,ce,pr)"),
				wxDefaultPosition, wxDefaultSize),
				0, wxGROW | wxALL, 3);
			m_enablepocCheck->SetValue(wxGetApp().m_enablepoc);

				// POC sizer, row
				wxBoxSizer* pocspecSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				pocspecSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&Changes:")),
								0, wxALL | wxALIGN_TOP, 3);

				// add some horizontal space
				pocspecSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				pocspecSizer->Add(
					m_pocCtrl = new wxTextCtrl(panel, OPJENCO_POCSPEC,
								wxGetApp().m_poc,
								wxDefaultPosition, wxSize(140, 60),
								wxTE_LEFT | wxTE_MULTILINE),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);
				m_pocCtrl->Enable(wxGetApp().m_enablepoc);

			pocSizer->Add(pocspecSizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(pocSizer, 0, wxGROW | wxALL, 3);
			
			// Comment settings, column
			wxStaticBox* commentBox = new wxStaticBox(panel, wxID_ANY, wxT("Comment"));
			wxBoxSizer* commentSizer = new wxStaticBoxSizer(commentBox, wxVERTICAL);

			// commenting check box
			commentSizer->Add(
				m_enablecommCheck = new wxCheckBox(panel, OPJENCO_ENABLECOMM, wxT("Enabled (empty to reset)"),
				wxDefaultPosition, wxDefaultSize),
				0, wxGROW | wxALL, 3);
			m_enablecommCheck->SetValue(wxGetApp().m_enablecomm);

			// add some horizontal space
			commentSizer->Add(3, 3, 1, wxALL, 0);

			// add the value control
			commentSizer->Add(
				m_commentCtrl = new wxTextCtrl(panel, OPJENCO_COMMENTTEXT,
							wxGetApp().m_comment,
							wxDefaultPosition, wxSize(wxDefaultCoord, 60),
							wxTE_LEFT | wxTE_MULTILINE),
				0, wxGROW | wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);
			m_commentCtrl->Enable(wxGetApp().m_enablecomm);

		subtopSizer->Add(commentSizer, 0, wxGROW | wxALL, 3);

			// Index file settings, column
			wxStaticBox* indexBox = new wxStaticBox(panel, wxID_ANY, wxT("Indexing"));
			wxBoxSizer* indexSizer = new wxStaticBoxSizer(indexBox, wxVERTICAL);

			// indexing check box
			indexSizer->Add(
				m_enableidxCheck = new wxCheckBox(panel, OPJENCO_ENABLEINDEX, wxT("Enabled"),
				wxDefaultPosition, wxDefaultSize),
				0, wxGROW | wxALL, 3);
			m_enableidxCheck->SetValue(wxGetApp().m_enableidx);

				// index file sizer, row
				wxBoxSizer* indexnameSizer = new wxBoxSizer(wxHORIZONTAL);

				// add some text
				indexnameSizer->Add(new wxStaticText(panel, wxID_ANY, wxT("&File name:")),
								0, wxALL | wxALIGN_CENTER_VERTICAL, 3);

				// add some horizontal space
				indexnameSizer->Add(3, 3, 1, wxALL, 0);

				// add the value control
				indexnameSizer->Add(
					m_indexCtrl = new wxTextCtrl(panel, OPJENCO_INDEXNAME,
								wxGetApp().m_index,
								wxDefaultPosition, wxSize(120, wxDefaultCoord),
								wxTE_LEFT),
					0, wxALL | wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL, 3);
				m_indexCtrl->Enable(wxGetApp().m_enableidx);

			indexSizer->Add(indexnameSizer, 0, wxGROW | wxALL, 3);

		subtopSizer->Add(indexSizer, 0, wxGROW | wxALL, 3);

	topSizer->Add(subtopSizer, 1, wxGROW | wxALIGN_CENTRE | wxALL, 5);

	// assign top and fit it
    panel->SetSizer(topSizer);
    topSizer->Fit(panel);

    return panel;
}

void OPJEncoderDialog::OnEnableComm(wxCommandEvent& event)
{
	if (event.IsChecked()) {
		wxLogMessage(wxT("Comment enabled"));
		m_commentCtrl->Enable(true);
	} else {
		wxLogMessage(wxT("Comment disabled"));
		m_commentCtrl->Enable(false);
	}

}

void OPJEncoderDialog::OnEnableIdx(wxCommandEvent& event)
{
	if (event.IsChecked()) {
		wxLogMessage(wxT("Index enabled"));
		m_indexCtrl->Enable(true);
	} else {
		wxLogMessage(wxT("Index disabled"));
		m_indexCtrl->Enable(false);
	}

}

void OPJEncoderDialog::OnEnablePoc(wxCommandEvent& event)
{
	if (event.IsChecked()) {
		wxLogMessage(wxT("POC enabled"));
		m_pocCtrl->Enable(true);
	} else {
		wxLogMessage(wxT("POC disabled"));
		m_pocCtrl->Enable(false);
	}

}

void OPJEncoderDialog::OnRadioQualityRate(wxCommandEvent& event)
{
	if (event.GetId() == OPJENCO_QUALITYRADIO) {
		wxLogMessage(wxT("Quality selected"));
		m_rateCtrl->Enable(false);
		m_qualityCtrl->Enable(true);
	} else {
		wxLogMessage(wxT("Rate selected"));
		m_rateCtrl->Enable(true);
		m_qualityCtrl->Enable(false);
	}
}

#ifdef USE_JPWL
void OPJEncoderDialog::OnEnableJPWL(wxCommandEvent& event)
{
	int specno;

	if (event.IsChecked()) {
		wxLogMessage(wxT("JPWL enabled"));
		for (specno = 0; specno < MYJPWL_MAX_NO_TILESPECS; specno++) {
			m_hprotChoice[specno]->Enable(true);
			m_htileCtrl[specno]->Enable(true);
			m_pprotChoice[specno]->Enable(true);
			m_ptileCtrl[specno]->Enable(true);
			m_ppackCtrl[specno]->Enable(true);
			m_sensiChoice[specno]->Enable(true);
			m_stileCtrl[specno]->Enable(true);
		}
		OnHprotSelect(event);
		OnPprotSelect(event);
		OnSensiSelect(event);
	} else {
		wxLogMessage(wxT("JPWL disabled"));
		for (specno = 0; specno < MYJPWL_MAX_NO_TILESPECS; specno++) {
			m_hprotChoice[specno]->Enable(false);
			m_htileCtrl[specno]->Enable(false);
			m_pprotChoice[specno]->Enable(false);
			m_ptileCtrl[specno]->Enable(false);
			m_ppackCtrl[specno]->Enable(false);
			m_sensiChoice[specno]->Enable(false);
			m_stileCtrl[specno]->Enable(false);
		}
	}

}

void OPJEncoderDialog::OnHprotSelect(wxCommandEvent& event)
{
	int specno;

	// deactivate properly
	for (specno = MYJPWL_MAX_NO_TILESPECS - 1; specno >= 0; specno--) {
		if (!m_hprotChoice[specno]->GetSelection()) {
			m_hprotChoice[specno]->Enable(false);
			m_htileCtrl[specno]->Enable(false);
		} else
			break;
	}
	if (specno < (MYJPWL_MAX_NO_TILESPECS - 1)) {
		m_hprotChoice[specno + 1]->Enable(true);
		m_htileCtrl[specno + 1]->Enable(true);
	}

	//wxLogMessage(wxT("hprot changed: %d"), specno);
}

void OPJEncoderDialog::OnPprotSelect(wxCommandEvent& event)
{
	int specno;

	// deactivate properly
	for (specno = MYJPWL_MAX_NO_TILESPECS - 1; specno >= 0; specno--) {
		if (!m_pprotChoice[specno]->GetSelection()) {
			m_pprotChoice[specno]->Enable(false);
			m_ptileCtrl[specno]->Enable(false);
			m_ppackCtrl[specno]->Enable(false);
		} else
			break;
	}
	if (specno < (MYJPWL_MAX_NO_TILESPECS - 1)) {
		m_pprotChoice[specno + 1]->Enable(true);
		m_ptileCtrl[specno + 1]->Enable(true);
		m_ppackCtrl[specno + 1]->Enable(true);
	}

	//wxLogMessage(wxT("pprot changed: %d"), specno);
}

void OPJEncoderDialog::OnSensiSelect(wxCommandEvent& event)
{
	int specno;

	// deactivate properly
	for (specno = MYJPWL_MAX_NO_TILESPECS - 1; specno >= 0; specno--) {
		if (!m_sensiChoice[specno]->GetSelection()) {
			m_sensiChoice[specno]->Enable(false);
			m_stileCtrl[specno]->Enable(false);
		} else
			break;
	}
	if (specno < (MYJPWL_MAX_NO_TILESPECS - 1)) {
		m_sensiChoice[specno + 1]->Enable(true);
		m_stileCtrl[specno + 1]->Enable(true);
	}

	//wxLogMessage(wxT("sprot changed: %d"), specno);
}


#endif // USE_JPWL

