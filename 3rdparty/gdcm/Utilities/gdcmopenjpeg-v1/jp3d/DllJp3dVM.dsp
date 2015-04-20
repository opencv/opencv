# Microsoft Developer Studio Project File - Name="DllOpenJPEG" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=DllOpenJPEG - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "DllOpenJPEG.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "DllOpenJPEG.mak" CFG="DllOpenJPEG - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "DllOpenJPEG - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "DllOpenJPEG - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "DllOpenJPEG - Win32 Release"

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
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "DLLOPENJPEG_EXPORTS" /Yu"stdafx.h" /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "OPJ_EXPORTS" /FD /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x40c /d "NDEBUG"
# ADD RSC /l 0x40c /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386 /out:"Release/OpenJPEG.dll"
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=if not exist dist mkdir dist	copy libopenjpeg\openjpeg3d.h dist	copy Release\OpenJPEG.dll dist	copy Release\OpenJPEG.lib dist
# End Special Build Tool

!ELSEIF  "$(CFG)" == "DllOpenJPEG - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "DLLOPENJPEG_EXPORTS" /Yu"stdafx.h" /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "OPJ_EXPORTS" /FD /GZ /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x40c /d "_DEBUG"
# ADD RSC /l 0x40c /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /out:"Debug/OpenJPEGd.dll" /pdbtype:sept
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=if not exist dist mkdir dist	copy libopenjpeg\openjpeg3d.h dist	copy Debug\OpenJPEGd.dll dist	copy Debug\OpenJPEGd.lib dist
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "DllOpenJPEG - Win32 Release"
# Name "DllOpenJPEG - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\libopenjpeg\bio.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\cio.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\dwt.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\event.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\image.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\j2k.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\j2k_lib.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\jp2.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\jpt.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\mct.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\mqc.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\openjpeg.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\pi.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\raw.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\t1.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\t2.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\tcd.c
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\tgt.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\libopenjpeg\bio.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\cio.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\dwt.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\event.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\fix.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\image.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\int.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\j2k.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\j2k_lib.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\jp2.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\jpt.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\mct.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\mqc.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\openjpeg3d.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\opj_includes.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\pi.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\raw.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\t1.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\t2.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\tcd.h
# End Source File
# Begin Source File

SOURCE=.\libopenjpeg\tgt.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\OpenJPEG.rc
# End Source File
# End Group
# End Target
# End Project
