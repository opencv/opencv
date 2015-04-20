#!/bin/sh
# The next line is executed by /bin/sh, but not tcl \
exec wish "$0" ${1+"$@"}

namespace eval jp3dVM {

    variable _progress 0
    variable _afterid  ""
    variable _status "Compute in progress..."
    variable notebook
    variable mainframe
    variable dataout "Process execution information"
    variable status
    variable prgtext
    variable prgindic

    set pwd [pwd]
    cd [file dirname [info script]]
    variable VMDIR [pwd]
    cd $pwd

    foreach script {encoder.tcl decoder.tcl} {
	namespace inscope :: source $VMDIR/$script
    }
}


proc jp3dVM::create { } {
    variable notebook
    variable mainframe
    variable dataout

    bind all <F12> { catch {console show} }

    # Menu description
    set descmenu {
        "&File" {} {} 0 {
            {command "E&xit" {} "Exit BWidget jp3dVM" {} -command exit}
        }
        "&Options" {} {} 0 {
            {command "&Encode" {} "Show encoder" {}
                -command  {$jp3dVM::notebook raise [$jp3dVM::notebook page 0]}
            }
            {command "&Decode" {} "Show decoder" {}
                -command  {$jp3dVM::notebook raise [$jp3dVM::notebook page 1]}
            }
        }
	"&Help" {} {} 0 {
            {command "&About authors..." {} "Show info about authors" {} 
		-command {MessageDlg .msgdlg -parent . -title "About authors" -message " Copyright @ LPI-UVA 2006 " -type ok -icon info}}
        }
    }

    set mainframe [MainFrame .mainframe \
                       -menu         $descmenu \
                       -textvariable jp3dVM::status \
                       -progressvar  jp3dVM::prgindic]

    $mainframe addindicator -text "JP3D Verification Model 1.0.0"

    # NoteBook creation
    set frame    [$mainframe getframe]
    set notebook [NoteBook $frame.nb]

    set logo [frame $frame.logo]
    #creo imagen logo
    image create photo LPIimg -file logoLPI.gif
    set logoimg [Label $logo.logoimg -image LPIimg]
    
    set f0  [VMEncoder::create $notebook]
    set f1  [VMDecoder::create $notebook]

	set tfinfo [TitleFrame $frame.codinfo -text "Program Execution"]
	set codinfo [$tfinfo getframe]
	set sw [ScrolledWindow $codinfo.sw -relief sunken -borderwidth 2 -scrollbar both]
	set sf [ScrollableFrame $codinfo.sf ]
	$sw setwidget $sf
	set subf [$sf getframe]
	set labinfo [label $subf.labinfo -textvariable jp3dVM::dataout -justify left]

	pack $labinfo -side left 
	pack $sw 

    $notebook compute_size
    $notebook raise [$notebook page 0]

    pack $logoimg -side left -fill x -expand yes 
    pack $notebook -expand yes 
    pack $logo $tfinfo -side left -expand yes 
    pack $mainframe -fill both -expand yes
    update idletasks
}


proc jp3dVM::main {} {
    variable VMDIR

    lappend ::auto_path [file dirname $VMDIR]
    namespace inscope :: package require BWidget

    option add *TitleFrame.l.font {helvetica 11 bold italic}

    wm withdraw .
    wm title . "JP3D Verification Model @ LPI"

    jp3dVM::create
    BWidget::place . 0 0 center
    wm deiconify .
    raise .
    focus -force .
}

jp3dVM::main
wm geom . [wm geom .]
