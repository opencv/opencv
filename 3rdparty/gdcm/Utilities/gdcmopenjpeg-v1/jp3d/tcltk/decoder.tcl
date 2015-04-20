
namespace eval VMDecoder {
	variable var
	variable JP3Ddecoder "../bin/jp3d_vm_dec.exe"
	#variable JP3Ddecoder "jp3d_vm_dec.exe"
}


proc VMDecoder::create { nb } {
	variable var

	set frameD [$nb insert end VMDecoder -text "Decoder"]
	set topfD	[frame $frameD.topfD]
	set medfD	[frame $frameD.medfD]
	set bottomfD	[frame $frameD.bottomfD]
	set srcfD [TitleFrame $topfD.srcfD -text "Source"]
	set dstfD [TitleFrame $topfD.dstfD -text "Destination"]
	set paramfD [TitleFrame $medfD.paramfD -text "Decoding parameters"]
	set infofD [TitleFrame $medfD.infofD -text "Distortion measures"]

	set frame1 [$srcfD getframe]
	_sourceD  $frame1
	set frame2  [$dstfD getframe]
	_destinationD  $frame2
	set frame3  [$infofD getframe]
	_originalD $frame3
	set frame4  [$paramfD getframe]
	_paramsD $frame4

	set butD   [Button $bottomfD.butD -text "Decode!" \
		   -command  "VMDecoder::_decode $frame1 $frame2 $frame3" \
		   -helptext "Decoding trigger button"]
	set butR   [Button $bottomfD.butR -text "Save info" \
		   -command  "VMDecoder::_save $frame3" \
		   -helptext "Save information"]
	
	pack $srcfD $dstfD -side left -fill both -padx 10 -ipadx 5 -expand yes
	pack $topfD -pady 4 -fill x
	
	pack $paramfD $infofD -side left -fill both -padx 10 -pady 2 -ipadx 5 -expand yes
	pack $medfD -pady 4 -fill x 

	pack $butD $butR -side left -padx 4 -pady 5 -expand yes
	pack $bottomfD -pady 4 -fill x

return $frameD
}


proc fileDialogD {w ent operation} {

	variable file
	
	if {$operation == "open"} {
		#-----Type names---------Extension(s)---
		set types {
		{"JP3D Files"	   {.jp3d}	}
		{"All files"		*}
		}
		set file [tk_getOpenFile -filetypes $types -parent $w ]
	} elseif {$operation == "original"} {
		#-----Type names---------Extension(s)---
		set types {
		{"BIN Raw Image Files"  {.bin}  }
		{"PGX Raw Image Files"	{.pgx}	}
		{"All files"		*}
		}
		set file [tk_getOpenFile -filetypes $types -parent $w ]
	} else {
		#-----Type names---------Extension(s)---
		set types {
		{"BIN Raw Image Files"  {.bin}  }
		{"PGX Raw Image Files"	{.pgx}	}
		{"All files"		*}
		}
		set file [tk_getSaveFile -filetypes $types -parent $w -initialfile Untitled -defaultextension "*.bin"]
	}
	if {[string compare $file ""]} {
		$ent delete 0 end
		$ent insert end $file
		$ent xview moveto 1
	}
}

proc VMDecoder::_sourceD { parent } {
	
	variable var
	
	set labsrcD [LabelFrame $parent.labsrcD -text "Select compressed file: " -side top \
			-anchor w -relief flat -borderwidth 0]
	set subsrcD [$labsrcD getframe]
	set listD [entry $subsrcD.entrysrcD -width 40 -textvariable VMDecoder::var(sourceD)]
	
	set labbrw [LabelFrame $parent.labbrw -side top -anchor w -relief flat -borderwidth 0]
	set subbrw [$labbrw getframe]
	set butbrw [button $subbrw.butbrw -image [Bitmap::get open] \
		-relief raised -borderwidth 1 -padx 1 -pady 1 \
		-command "fileDialogD . $subsrcD.entrysrcD open"]
	
	pack $listD -side top
	pack $butbrw -side top
	pack $labsrcD $labbrw -side left -fill both -expand yes


}

proc VMDecoder::_destinationD { parent } {
	
	variable var
	
	set labdstD [LabelFrame $parent.labdstD -text "Save decompressed volume file(s) as: " -side top \
			-anchor w -relief flat -borderwidth 0]
	set subdstD [$labdstD getframe]
	set listD [entry $subdstD.entrydstD -width 40 -textvariable VMDecoder::var(destinationD)]
	
	set labbrw [LabelFrame $parent.labbrw -side top -anchor w -relief flat -borderwidth 0]
	set subbrw [$labbrw getframe]
	set butbrw [button $subbrw.butbrw -image [Bitmap::get save] \
		-relief raised -borderwidth 1 -padx 1 -pady 1 \
		-command "fileDialogD . $subdstD.entrydstD save"]

	pack $listD -side top
	pack $butbrw -side top
	pack $labdstD $labbrw -side left -fill both -expand yes
}

proc VMDecoder::_originalD { parent } {
	
	variable var
	
	set laborgD [LabelFrame $parent.laborgD -text "Select original file: " -side top \
			-anchor w -relief flat -borderwidth 0]
	set suborgD [$laborgD getframe]
	set listorgD [entry $suborgD.entryorgD -width 30 -textvariable VMDecoder::var(originalD)]
	
	set labbrw2 [LabelFrame $parent.labbrw2 -side top -anchor w -relief flat -borderwidth 0]
	set subbrw2 [$labbrw2 getframe]
	set butbrw2 [button $subbrw2.butbrw2 -image [Bitmap::get open] \
		-relief raised -borderwidth 1 -padx 1 -pady 1 \
		-command "fileDialogD . $suborgD.entryorgD original"]
	
	set infoD [Label $parent.infoD -relief sunken -textvariable VMDecoder::var(decodinfo) -justify left]

	pack $listorgD -side left -anchor n
	pack $butbrw2 -side left -anchor n
	pack $infoD -side bottom -anchor nw -pady 4 -ipadx 150 -ipady 20 -expand yes
	pack $laborgD $labbrw2 -side left -fill both 


}

proc VMDecoder::_paramsD { parent } {
	
	variable var
	
	########### DECODING  #############
	set labcod [LabelFrame $parent.labcod -side top -anchor w -relief sunken -borderwidth 1]
	set subcod  [$labcod getframe]

		set frameres [frame $subcod.frameres -borderwidth 1]
		set labres [LabelEntry $frameres.labres -label "Resolutions to discard: " -labelwidth 20 -labelanchor w \
                   -textvariable VMDecoder::var(resdiscard) -editable 1 \
                   -helptext "Number of highest resolution levels to be discarded on each dimension" ]
	set VMDecoder::var(resdiscard) "0,0,0"
	
		set framelayer [frame $subcod.framelayer -borderwidth 1]
		set lablayer [LabelEntry $framelayer.lablayer -label "Layers to decode: " -labelwidth 20 -labelanchor w \
                   -textvariable VMDecoder::var(layer) -editable 1 \
                   -helptext "Maximum number of quality layers to decode" ]
	set VMDecoder::var(layer) "All"

	set framebe [frame $subcod.framebe -borderwidth 1]
	set chkbe [checkbutton $framebe.chkbe -text "Write decoded file with BigEndian byte order" \
		   -variable VMDecoder::var(be) -onvalue 1 -offvalue 0 ]

		pack $labres -side left -padx 2 -anchor n
		pack $lablayer -side left -padx 2 -anchor n
		pack $chkbe -side left -padx 2 -anchor w
		pack $frameres $framelayer $framebe -side top -anchor w

	pack $subcod -anchor n
	pack $labcod -side left -fill both -padx 4 -expand yes
}


proc VMDecoder::_decode { framesrc framedst frameinfo} {

	variable var

	set sourceD [$framesrc.labsrcD.f.entrysrcD get ]
	set destinationD [$framedst.labdstD.f.entrydstD get ]
	set originD [$frameinfo.laborgD.f.entryorgD get ]
	set cond1 [string match *.pgx [string tolower $destinationD]]
	set cond2 [string match *\**.pgx [string tolower $destinationD]]
	set cond3 [string match *.bin [string tolower $destinationD]]
	
	#comprobamos datos son correctos
	if {($cond1 == 1) && ($cond2 == 0)} {
		set pgx "*.pgx"
		set pattern [string range $destinationD 0 [expr [string length $destinationD]-5]]
		set destinationD $pattern$img
	} elseif {$sourceD == ""} {
	  MessageDlg .msgdlg -parent . -message "Error : Source file is not defined !" -type ok -icon error 
	} elseif {$destinationD == ""} {
	  MessageDlg .msgdlg -parent . -message "Error : Destination file is not defined !" -type ok -icon error 
	} else {

		#creamos datain a partir de los parametros de entrada
		#set dirJP3Ddecoder [mk_relativepath $VMDecoder::JP3Ddecoder]
		set dirJP3Ddecoder $VMDecoder::JP3Ddecoder
		set datain [concat " $dirJP3Ddecoder -i [mk_relativepath $sourceD] "]
		set datain [concat " $datain -o [mk_relativepath $destinationD] "]
		if {$originD != ""} {
			set datain [concat " $datain -O [mk_relativepath $originD] "]
			if {$cond3 == 1} {
				set img ".img"
				set pattern [string range $originD 0 [expr [string length $originD]-5]]
				set pattern $pattern$img
				if {[file exists $pattern]} {
				  set datain [concat " $datain -m [mk_relativepath $pattern] "]
				} else {
				  MessageDlg .msgdlg -parent . -message "Error : IMG file associated to original BIN volume file not found in same directory !" -type ok -icon info 
				}
			}
		}
		if {$VMDecoder::var(resdiscard) != "0,0,0"} {
			set datain [concat " $datain -r $VMDecoder::var(resdiscard) "]
		}
		if {$VMDecoder::var(layer) != "All" && $VMDecoder::var(layer) > 0} {
			set datain [concat " $datain -l $VMDecoder::var(layer) "]
		}
		if {$VMDecoder::var(be) == 1} {
			set datain [concat " $datain -BE"]
		}
		
		set VMDecoder::var(progval) 10
		ProgressDlg .progress -parent . -title "Wait..." \
			-type         infinite \
			-width        20 \
			-textvariable "Compute in progress..."\
			-variable     VMDecoder::progval \
			-stop         "Stop" \
			-command      {destroy .progress}

		after 200 set VMDecoder::var(progval) 2

		set fp [open "| $datain " r+] 
		fconfigure $fp -buffering line 
		set jp3dVM::dataout [concat "EXECUTED PROGRAM:\n\t$datain"]
		while {-1 != [gets $fp tmp]} {
			set jp3dVM::dataout [concat "$jp3dVM::dataout\n$tmp"]
		}
		close $fp
		destroy .progress
		set cond [string first "ERROR" $jp3dVM::dataout]
		set cond2 [string first "PSNR" $jp3dVM::dataout]
		set cond3 [string first "RESULT" $jp3dVM::dataout]
		if {$cond != -1} {
		   MessageDlg .msgdlg -parent . -message [string range $jp3dVM::dataout [expr $cond-1] end] -type ok -icon error
		} elseif {$cond3 != -1} {
			if {$cond2 != -1} {
				set VMDecoder::var(decodinfo) [string range $jp3dVM::dataout [expr $cond2-1] end]
			}
			MessageDlg .msgdlg -parent . -message [string range $jp3dVM::dataout [expr $cond3-1] end] -type ok -icon info
		}
	}
}

proc VMDecoder::_save { frameinfo } {

}

