
namespace eval VMEncoder {
	variable var
	variable JP3Dencoder "../bin/jp3d_vm_enc.exe"
}

proc VMEncoder::create { nb } {

	set frame [$nb insert end VMEncoder -text "Encoder"]
	set topf	[frame $frame.topf]
	set midf	[frame $frame.midf]
	set bottomf	[frame $frame.bottomf]
	set srcf [TitleFrame $topf.srcf -text "Source"]
	set dstf [TitleFrame $topf.dstf -text "Destination"]
	set Tparf [TitleFrame $midf.parfT -text "Transform Parameters"]
	set Cparf [TitleFrame $midf.parfC -text "Coding Parameters"]

	set frame1 [$srcf getframe]
		VMEncoder::_sourceE  $frame1
	
	set frame2  [$dstf getframe]
		VMEncoder::_destinationE  $frame2
	
	set frame3  [$Tparf getframe]
		VMEncoder::_transformE $frame3

	set frame4  [$Cparf getframe]
		VMEncoder::_codingE $frame4
	
	set butE  [Button $bottomf.butE -text "Encode!" \
		   -command  "VMEncoder::_encode $frame1 $frame2" \
		   -helptext "Encoding trigger button"]
	set butR  [Button $bottomf.butR -text "Restore defaults" \
		   -command  "VMEncoder::_reset $frame1 $frame2 $frame3 $frame4" \
		   -helptext "Reset to default values"]

	pack $srcf $dstf -side left -fill y -padx 4 -expand yes
	pack $topf -pady 2 -fill x

	pack $Tparf $Cparf -side left -fill both -padx 4 -expand yes
	pack $midf -pady 2 -fill x
	
	pack $butE $butR -side left -padx 40 -pady 5 -fill y -expand yes
	pack $bottomf -pady 2 -fill x

	return $frame
}

proc VMEncoder::_sourceE { parent } {

	variable var

	set labsrc [LabelFrame $parent.labsrc -text "Select volume file to encode: " -side top \
			-anchor w -relief flat -borderwidth 0]
	set subsrc [$labsrc getframe]
	set list [entry $subsrc.entrysrc -width 30 -textvariable VMDecoder::var(source)]
	
	set labbrw [LabelFrame $parent.labbrw -side top -anchor w -relief flat -borderwidth 0]
	set subbrw [$labbrw getframe]
	set butbrw [button $subbrw.butbrw -image [Bitmap::get open] \
		-relief raised -borderwidth 1 -padx 1 -pady 1 \
		-command "fileDialogE . $subsrc.entrysrc open"]
	
	pack $list -side top
	pack $butbrw -side top
	pack $labsrc $labbrw -side left -fill both -expand yes
}

proc VMEncoder::_destinationE { parent } {

	variable var

	set labdst [LabelFrame $parent.labdst -text "Save compressed volume as: " -side top \
			-anchor w -relief flat -borderwidth 0]
	set subdst [$labdst getframe]
	set list [entry $subdst.entrydst -width 30 -textvariable VMDecoder::var(destination)]
	
	set labbrw [LabelFrame $parent.labbrw -side top -anchor w -relief flat -borderwidth 0]
	set subbrw [$labbrw getframe]
	set butbrw [button $subbrw.butbrw -image [Bitmap::get save] \
		-relief raised -borderwidth 1 -padx 1 -pady 1 \
		-command "fileDialogE . $subdst.entrydst save"]
	
	pack $list -side top
	pack $butbrw -side top
	pack $labdst $labbrw -side left -fill both -expand yes
}

proc VMEncoder::_codingE { parent } {

	
	########### CODING  #############
	set labcod [LabelFrame $parent.labcod -side top -anchor w -relief sunken -borderwidth 1]
	set subcod  [$labcod getframe]

		set framerate [frame $subcod.framerate -borderwidth 1]
		set labrate [LabelEntry $framerate.labrate -label "Rates: " -labelwidth 9 -labelanchor w \
                   -textvariable VMEncoder::var(rate) -editable 1 \
                   -helptext "Compression ratios for different layers (R1, R2, R3,...). If R=1, lossless coding" ]
	set VMEncoder::var(rate) "1"

		set framecblk [frame $subcod.framecblk -borderwidth 1]
		set labcblk [LabelEntry $framecblk.labcblk -label "Codeblock: " -labelwidth 9 -labelanchor w \
                   -textvariable VMEncoder::var(cblksize) -editable 1 \
                   -helptext "Codeblock size (X, Y, Z)" ]
	set VMEncoder::var(cblksize) "64,64,64"

		set frametile [frame $subcod.frametile -borderwidth 1]
		set labtile [LabelEntry $frametile.labtile -label "Tile size: " -labelwidth 9 -labelanchor w \
                   -textvariable VMEncoder::var(tilesize) -editable 1 \
                   -helptext "Tile size (X, Y, Z)" ]
	set VMEncoder::var(tilesize) "512,512,512"

		set framesop [frame $subcod.framesop -borderwidth 1]
		set chksop [checkbutton $framesop.chksop -text "Write SOP marker" \
			   -variable VMEncoder::var(sop) -onvalue 1 -offvalue 0 ]
		set frameeph [frame $subcod.frameeph -borderwidth 1]
		set chkeph [checkbutton $frameeph.chkeph -text "Write EPH marker" \
			   -variable VMEncoder::var(eph) -onvalue 1 -offvalue 0 ]
	
		set framepoc [frame $subcod.framepoc -borderwidth 1]
		set labpoc [label $framepoc.labpoc -text "Progression order: " ]
		set progorder [ComboBox $framepoc.progorder \
			   -text {Choose a progression order} \
			   -width 10 \
			   -textvariable VMEncoder::var(progorder) \
			   -values {"LRCP" "RLCP" "RPCL" "PCRL" "CPRL"} \
			   -helptext "Progression order"]
		set VMEncoder::var(progorder) "LRCP"

		pack $labrate -side left -padx 2 -anchor n
		pack $labcblk -side left -padx 2 -anchor n
		pack $labpoc $progorder -side left -padx 2 -anchor w
		#pack $labtile -side left -padx 2 -anchor n
		pack $chksop -side left -padx 2 -anchor w
		pack $chkeph -side left -padx 2 -anchor w
	########### ENTROPY CODING  #############
	set labent [LabelFrame $parent.labent -text "Entropy Coding" -side top -anchor w -relief sunken -borderwidth 1]
	set subent  [$labent getframe]
		foreach entval {2EB 3EB} entropy {2D_EBCOT 3D_EBCOT} {
			set rad [radiobutton $subent.$entval \
				-text $entropy \
				-variable VMEncoder::var(encoding) \
				-command "disableGR $entval $labcblk $progorder $labrate $chksop $chkeph" \
				-value $entval ]
			pack $rad -anchor w
		}
		$subent.2EB select 

	pack $subent -padx 2 -anchor n

	pack $framerate $framecblk $framepoc $framesop $frameeph -side top -anchor w
	pack $subcod -anchor n

	pack $labent $labcod -side left -fill both -padx 4 -expand yes


}

proc VMEncoder::_transformE { parent } {

	variable var

	########### TRANSFORM  #############
	set labtrf [LabelFrame $parent.labtrf -text "Transform" -side top -anchor w -relief sunken -borderwidth 1]
	set subtrf  [$labtrf getframe]
	set labres [LabelFrame $parent.labres -side top -anchor w -relief sunken -borderwidth 1]
	set subres [$labres getframe]
		
		########### ATK #############
		set frameatk [frame $subres.frameatk -borderwidth 1]
		set labatk [label $frameatk.labatk -text "Wavelet kernel:  " -anchor w]
		set atk [ComboBox $frameatk.atk \
				-textvariable VMEncoder::var(atk) \
				-width 20 \
				-text {Choose a wavelet kernel} \
				-editable false \
				-values {"R5.3" "I9.7"} ]
		set VMEncoder::var(atk) "R5.3"
		pack $labatk $atk -side left -anchor w
		########### RESOLUTIONS #############
		set frameres1 [frame $subres.frameres1 -borderwidth 1]
		set labresolution [label $frameres1.labresol -text "Resolutions: " -anchor w ]
		set frameres2 [frame $subres.frameres2 -borderwidth 1]
		set labresX [label $frameres2.labresX -text "  X" -anchor w ]
		set labresY [label $frameres2.labresY -text "  Y" -anchor w ]
		set labresZ [label $frameres2.labresZ -text "  Z" -anchor w ]
		

		set resX [SpinBox $frameres2.spinresX \
				-range {1 6 1} -textvariable VMEncoder::var(resX) \
				-helptext "Number of resolutions in X" \
				-width 3 \
				-editable false ]
 		set resY [SpinBox $frameres2.spinresY \
				-range {1 6 1} -textvariable VMEncoder::var(resY) \
				-helptext "Number of resolutions in Y" \
				-width 3 \
				-editable false ]
		set resZ [SpinBox $frameres2.spinresZ \
				-range {1 6 1} -textvariable VMEncoder::var(resZ) \
				-helptext "Number of resolutions in Z" \
				-width 3 \
				-editable false \
				-state disabled ]
		set VMEncoder::var(resX) 3
		set VMEncoder::var(resY) 3
		set VMEncoder::var(resZ) 3

		########### TRF  #############
		foreach trfval {2DWT 3DWT} trf {2D-DWT 3D-DWT} {
			set rad [radiobutton $subtrf.$trfval -text $trf \
					-variable VMEncoder::var(transform) \
					-command "disable3RLS $trfval $atk $resX $resY $resZ"\
					-value $trfval ]
			pack $rad -anchor w
		}
		$subtrf.2DWT select
		
	pack $subtrf -side left -padx 2 -pady 4
	
		pack $labresolution -padx 2 -side left -anchor w
		pack $labresX $resX -padx 2 -side left -anchor w
		pack $labresY $resY -padx 2 -side left -anchor w
		pack $labresZ $resZ -padx 2 -side left -anchor w

		pack $frameres1 -side top -fill x
		pack $frameres2 $frameatk -side top -padx 2 -pady 4 -anchor n

	pack $subres -side left -padx 2 -pady 4
	pack $labtrf $labres -side left -fill both -padx 4 -expand yes
}


proc VMEncoder::_encode { framesrc framedst } {

	variable var

	set source [$framesrc.labsrc.f.entrysrc get ]
	set destination [$framedst.labdst.f.entrydst get ]
	set cond1 [string match *.pgx [string tolower $source]]
	set cond2 [string match *-*.pgx [string tolower $source]]
	set cond3 [string match *.bin [string tolower $source]]

	set img ".img"
	set pattern [string range $source 0 [expr [string length $source]-5]]
	set pattern $pattern$img
	set exist [file exists $pattern]
	
	#comprobamos datos son correctos
	if {($cond1 == 1) && ($cond2 == 0)} {
	  MessageDlg .msgdlg -parent . -message "Info : Really want to encode an slice instead of a volume?.\n For a group of .pgx slices, name must contain a - denoting a sequential index!" -type ok -icon info
	} 
	
	if {$source == ""} {
	  MessageDlg .msgdlg -parent . -message "Error : Source file is not defined !" -type ok -icon error 
	} elseif {$destination == ""} {
	  MessageDlg .msgdlg -parent . -message "Error : Destination file is not defined !" -type ok -icon error 
	} elseif { ($VMEncoder::var(transform) != "3RLS") && ($VMEncoder::var(atk) == "Choose a wavelet transformation kernel") } {
	  MessageDlg .msgdlg -parent . -title "Info" -message "Please choose a wavelet transformation kernel"\
			-type ok -icon warning
	} elseif {($exist == 0) && ($cond1 == 0) && ($cond3 == 1)} {
		  MessageDlg .msgdlg -parent . -message "Error : IMG file associated to BIN volume file not found in same directory !" -type ok -icon info 
	} else {

		#creamos datain a partir de los parametros de entrada
#		set dirJP3Dencoder [mk_relativepath $VMEncoder::JP3Dencoder]
		set dirJP3Dencoder $VMEncoder::JP3Dencoder
		set datain [concat " $dirJP3Dencoder -i [mk_relativepath $source] "]
		if {$cond3 == 1} {
		   set datain [concat " $datain -m [mk_relativepath $pattern] "]
		}
		set datain [concat " $datain -o [mk_relativepath $destination] "]
		if {$VMEncoder::var(encoding) != "2EB"} {
			set datain [concat " $datain -C $VMEncoder::var(encoding) "]
		}
		if {$VMEncoder::var(transform) == "2DWT"} {
			set datain [concat " $datain -n $VMEncoder::var(resX),$VMEncoder::var(resY) "]
		} elseif {$VMEncoder::var(transform) == "3DWT"} {
			set datain [concat " $datain -n $VMEncoder::var(resX),$VMEncoder::var(resY),$VMEncoder::var(resZ) "]
		}
		
		set datain [concat " $datain -r $VMEncoder::var(rate) "]
		
		if {$VMEncoder::var(atk) == "I9.7"} {
			set datain [concat " $datain -I "]
		} 
		if {$VMEncoder::var(sop) == 1} {
			set datain [concat " $datain -SOP "]
		}
		if {$VMEncoder::var(eph) == 1} {
			set datain [concat " $datain -EPH "]
		}
		if {$VMEncoder::var(progorder) != "LRCP"} {
			set datain [concat " $datain -p $VMEncoder::var(progorder) "]
		}
		if {$VMEncoder::var(cblksize) != "64,64,64"} {
			set datain [concat " $datain -b $VMEncoder::var(cblksize) "]
		}

		
		#Making this work would be great !!! 
		set VMEncoder::var(progval) 10
		ProgressDlg .progress -parent . -title "Wait..." \
			-type         infinite \
			-width        20 \
			-textvariable "Compute in progress..."\
			-variable     VMEncoder::progval \
			-stop         "Stop" \
			-command      {destroy .progress}
		after 200 set VMEncoder::var(progval) 2
		set fp [open "| $datain " r+] 
		fconfigure $fp -buffering line 
		set jp3dVM::dataout [concat "EXECUTED PROGRAM:\n\t$datain"]
		while {-1 != [gets $fp tmp]} {
			set jp3dVM::dataout [concat "$jp3dVM::dataout\n$tmp"]
		}
		destroy .progress
		set cond [string first "ERROR" $jp3dVM::dataout]
		set cond2 [string first "RESULT" $jp3dVM::dataout]
		if {$cond != -1} {
		   MessageDlg .msgdlg -parent . -message [string range $jp3dVM::dataout [expr $cond-1] end] -type ok -icon error
		} elseif {$cond2 != -1} {
		   MessageDlg .msgdlg -parent . -message [string range $jp3dVM::dataout [expr $cond2+7] end] -type ok -icon info
		   close $fp
		} else {
		   #Must do something with this !!! [pid $fp]
  		   close $fp
		}
	}
}

proc VMEncoder::_reset { framesrc framedst frametrf framecod} {

	variable var

	#Restore defaults values
	set VMEncoder::var(transform) 2DWT
	set VMEncoder::var(encoding) 2EB
	set VMEncoder::var(atk) "R5.3"
	set VMEncoder::var(progorder) "LRCP"
	set atk $frametrf.labres.f.frameatk.atk
	set resX $frametrf.labres.f.frameres2.spinresX
	set resY $frametrf.labres.f.frameres2.spinresY
	set resZ $frametrf.labres.f.frameres2.spinresZ
	disable3RLS 2DWT $atk $resX $resY $resZ 
	set labcblk $framecod.labcod.f.framecblk.labcblk
	set progorder $framecod.labcod.f.framepoc.progorder
	set labrate $framecod.labcod.f.framerate.labrate
	set chksop $framecod.labcod.f.framesop.chksop
	set chkeph $framecod.labcod.f.frameeph.chkeph
	disableGR 3EB $labcblk $progorder $labrate $chksop $chkeph

	$framesrc.labsrc.f.entrysrc delete 0 end
	$framedst.labdst.f.entrydst delete 0 end
}

proc fileDialogE {w ent operation} {

	variable file
	variable i j

	if {$operation == "open"} {
		set types {
			{"Source Image Files"	{.pgx .bin}	}
			{"All files"		*}
		}
		set file [tk_getOpenFile -filetypes $types -parent $w]
		if {[string compare $file ""]} {
			$ent delete 0 end
			$ent insert end $file
			$ent xview moveto 1
		}
	} else {
		set types {
			{"JP3D Files"		   {.jp3d}	}
			{"JPEG2000 Files"	   {.j2k}	}
			{"All files"		*}
		}
		set file [tk_getSaveFile -filetypes $types -parent $w \
			-initialfile Untitled -defaultextension .jp3d]
		if {[string compare $file ""]} {
			$ent delete 0 end
			$ent insert end $file
			$ent xview moveto 1
		}
	}
}

proc mk_relativepath {abspath} {

	set mydir [split [string trimleft [pwd] {/}] {/}]
	set abspathcomps [split [string trimleft $abspath {/}] {/}]

	set i 0
	while {$i<[llength $mydir]} {
		if {![string compare [lindex $abspathcomps $i] [lindex $mydir $i]]} {
			incr i
		} else {
			break
		}
	}
	set h [expr [llength $mydir]-$i]
	set j [expr [llength $abspathcomps]-$i]

	if {!$h} {
		set relpath "./"
	} else {
		set relpath ""
		while { $h > 0 } {
			set relpath "../$relpath"
			incr h -1
		}
	}

	set h [llength $abspathcomps]
	while { $h > $i } {
		set relpath [concat $relpath[lindex $abspathcomps [expr [llength $abspathcomps]-$j]]/]
		incr h -1
		incr j -1
	}
	return [string trim $relpath {/}]
}

proc disable3RLS {flag atk resX resY resZ}  {

	if {$flag == "3RLS"} {
		$atk configure -state disabled
		$resX configure -state disabled
		$resY configure -state disabled
		$resZ configure -state disabled
	} elseif {$flag == "2DWT"} {
		$atk configure -state normal
		$resX configure -state normal
		$resY configure -state normal
		$resZ configure -state disabled
	} elseif {$flag == "3DWT"} {
		$atk configure -state normal
		$resX configure -state normal
		$resY configure -state normal
		$resZ configure -state normal
	}
}

proc disableGR {flag labcblk progorder labrate chksop chkeph} {
	
	if {$flag == "2EB"} {
		$labcblk configure -state normal
		$progorder configure -state normal
		$labrate configure -state normal
		$chksop configure -state normal
		$chkeph configure -state normal
		set VMEncoder::var(cblksize) "64,64,64"
		set VMEncoder::var(tilesize) "512,512,512"
	} elseif {$flag == "3EB"} {
		$labcblk configure -state normal
		$progorder configure -state normal
		$labrate configure -state normal
		$chksop configure -state normal
		$chkeph configure -state normal
		set VMEncoder::var(cblksize) "64,64,64"
		set VMEncoder::var(tilesize) "512,512,512"
	} else {
		$labcblk configure -state disabled
		$progorder configure -state disabled
		$labrate configure -state disabled
		$chksop configure -state disabled
		$chkeph configure -state disabled
	}
}