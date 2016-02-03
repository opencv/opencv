# Microsoft Developer Studio Project File - Name="OPJViewer" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=OPJVIEWER - WIN32 RELEASE
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "OPJViewer.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "OPJViewer.mak" CFG="OPJVIEWER - WIN32 RELEASE"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "OPJViewer - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "OPJViewer - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "OPJViewer - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /G6 /MD /W3 /GX /O2 /I "$(WXWIN28)\lib\vc_lib\msw" /I "$(WXWIN28)\include" /I ".." /I "..\libopenjpeg" /I "$(MXFLIB)" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "_WINDOWS" /D WINVER=0x400 /D "_MT" /D wxUSE_GUI=1 /D "wxUSE_LIBOPENJPEG" /D "OPJ_STATIC" /D "USE_JPWL" /D "USE_JPSEC" /D "OPJ_HTMLABOUT" /D "OPJ_MANYFORMATS" /D "OPJ_INICONFIG" /FR /FD /Zm200 /c
# ADD BASE RSC /l 0x410 /d "NDEBUG"
# ADD RSC /l 0x409 /i "$(WXWIN28)\include" /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib comctl32.lib rpcrt4.lib wsock32.lib wxzlib.lib wxregex.lib wxpng.lib wxjpeg.lib wxbase28.lib wxmsw28_core.lib wxmsw28_html.lib wxmsw28_adv.lib wxmsw28_core.lib wxbase28.lib wxtiff.lib wxjpeg.lib wxpng.lib wxzlib.lib wxregex.lib wxexpat.lib LibOpenJPEG_JPWL.lib mxflib.lib /nologo /subsystem:windows /machine:I386 /nodefaultlib:"libcmt.lib" /libpath:"$(WXWIN28)\lib\vc_lib" /libpath:"..\jpwl\Release" /libpath:"$(MXFLIB)\build\msvc\Release" /IGNORE:4089
# SUBTRACT LINK32 /pdb:none /nodefaultlib
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Update build number
PostBuild_Cmds=buildupdate.bat
# End Special Build Tool

!ELSEIF  "$(CFG)" == "OPJViewer - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "OPJViewer___Win32_Debug"
# PROP BASE Intermediate_Dir "OPJViewer___Win32_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GX /Zi /Od /I "$(WXWIN28)\INCLUDE" /I "$(WXWIN28)\lib\vc_lib\msw" /I "$(WXWIN28)\include" /I ".." /I "..\libopenjpeg" /I "$(MXFLIB)" /D "_DEBUG" /D "__WXDEBUG__" /D WXDEBUG=1 /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "_WINDOWS" /D WINVER=0x400 /D "_MT" /D wxUSE_GUI=1 /D "wxUSE_LIBOPENJPEG" /D "OPJ_STATIC" /D "USE_JPWL" /D "OPJ_HTMLABOUT" /D "OPJ_INICONFIG" /D "OPJ_MANYFORMATS" /D "USE_JPSEC" /FR /FD /GZ /Zm200 /c
# ADD BASE RSC /l 0x410 /d "_DEBUG"
# ADD RSC /l 0x410 /i "$(WXWIN28)\include" /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib comctl32.lib rpcrt4.lib wsock32.lib wxzlibd.lib wxregexd.lib wxpngd.lib wxjpegd.lib wxtiffd.lib wxbase28d.lib wxmsw28d_core.lib wxmsw28d_html.lib wxmsw28d_adv.lib LibOpenJPEG_JPWLd.lib mxflib.lib /nologo /subsystem:windows /debug /machine:I386 /nodefaultlib:"libcmtd.lib" /pdbtype:sept /libpath:"$(WXWIN28)\lib\vc_lib" /libpath:"..\jpwl\Debug" /libpath:"$(MXFLIB)\build\msvc\Debug"
# SUBTRACT LINK32 /pdb:none

!ENDIF 

# Begin Target

# Name "OPJViewer - Win32 Release"
# Name "OPJViewer - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\source\imagjpeg2000.cpp
# End Source File
# Begin Source File

SOURCE=.\source\imagmxf.cpp
# End Source File
# Begin Source File

SOURCE=..\codec\index.c
# End Source File
# Begin Source File

SOURCE=.\source\OPJAbout.cpp
# End Source File
# Begin Source File

SOURCE=.\source\OPJDialogs.cpp
# End Source File
# Begin Source File

SOURCE=.\source\OPJThreads.cpp
# End Source File
# Begin Source File

SOURCE=.\source\OPJViewer.cpp
# End Source File
# Begin Source File

SOURCE=.\source\wxj2kparser.cpp
# End Source File
# Begin Source File

SOURCE=.\source\wxjp2parser.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\source\about_htm.h
# End Source File
# Begin Source File

SOURCE=.\source\build.h
# PROP Exclude_From_Build 1
# End Source File
# Begin Source File

SOURCE=.\source\imagjpeg2000.h
# End Source File
# Begin Source File

SOURCE=.\source\imagmxf.h
# End Source File
# Begin Source File

SOURCE=..\codec\index.h
# End Source File
# Begin Source File

SOURCE=.\source\OPJViewer.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\source\wx\msw\blank.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\bullseye.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\cdrom.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\computer.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\cross.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\drive.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\file1.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\floppy.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\folder1.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\folder2.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\hand.cur
# End Source File
# Begin Source File

SOURCE=.\source\icon1.xpm
# End Source File
# Begin Source File

SOURCE=.\source\icon2.xpm
# End Source File
# Begin Source File

SOURCE=.\source\icon3.xpm
# End Source File
# Begin Source File

SOURCE=.\source\icon4.xpm
# End Source File
# Begin Source File

SOURCE=.\source\icon5.xpm
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\magnif1.cur
# End Source File
# Begin Source File

SOURCE=.\source\opj_logo.xpm
# End Source File
# Begin Source File

SOURCE=.\source\OPJChild.ico
# End Source File
# Begin Source File

SOURCE=.\source\OPJChild16.xpm
# End Source File
# Begin Source File

SOURCE=.\source\OPJViewer.ico
# End Source File
# Begin Source File

SOURCE=.\source\OPJViewer.rc
# End Source File
# Begin Source File

SOURCE=.\source\OPJViewer16.xpm
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\pbrush.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\pencil.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\pntleft.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\pntright.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\removble.ico
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\rightarr.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\roller.cur
# End Source File
# Begin Source File

SOURCE=.\source\wx\msw\std.ico
# End Source File
# End Group
# End Target
# End Project
