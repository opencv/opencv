#!/bin/sh

rm -rf $HOME/.wine

SETUPVS=no

if [ "$SETUPVS" = "yes"]; then
# install gecko stuff:
winetricks apps list
# install VS 2005
# see http://bugs.winehq.org/show_bug.cgi?id=31052 for reason option no-isolate:
winetricks --no-isolate vc2005express
#export WINEPREFIX=$HOME/.local/share/wineprefixes/vc2005express
# required see: http://bugs.winehq.org/show_bug.cgi?id=20110#c8
#winetricks vcrun2005

mkdir -p $HOME/.cache/winetricks/junk
cd $HOME/.cache/winetricks/junk

# install VS 2005/SP1
# see why: https://code.google.com/p/winetricks/issues/detail?id=18
wget -c http://download.microsoft.com/download/7/7/3/7737290f-98e8-45bf-9075-85cc6ae34bf1/VS80sp1-KB926748-X86-INTL.exe
# now instal SP1
#wine VS80sp1-KB926748-X86-INTL.exe
fi

SETUP3RD=no
if [ "$SETUP3RD" = "yes"]; then
wget -c http://msysgit.googlecode.com/files/Git-1.7.11-preview20120620.exe
wget -c http://www.cmake.org/files/v2.8/cmake-2.8.8-win32-x86.exe
wget -c http://prdownloads.sourceforge.net/swig/swigwin-2.0.7.zip
wget -c http://slproweb.com/download/Win32OpenSSL-1_0_1c.exe
#wget "http://download.oracle.com/otn-pub/java/jdk/6u25-b06/jdk-6u25-windows-i586.exe?AuthParam=1340870089_b98e26f4e28100ecbb9c7ca9d3c3353f"
# have to manually download it at:
# http://www.oracle.com/technetwork/java/javase/downloads/jdk6-downloads-1637591.html
wget -c http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi

# install !

wine cmake-2.8.8-win32-x86.exe
wine Git-1.7.11-preview20120620.exe
msiexec /i python-2.7.3.msi
# dont ask:
#winetricks vcrun2005
# call twice in case of failure:
#wine VS80sp1-KB926748-X86-INTL.exe
wine Win32OpenSSL-1_0_1c.exe
wine jdk-6u33-windows-i586.exe

mkdir "$HOME/.wine/drive_c/Program Files/Swig"
unzip -d "$HOME/.wine/drive_c/Program Files/Swig" swigwin-2.0.7.zip
fi

# You will need wine 1.5.7 otherwise you get:

# Unhandled exception: unimplemented function msvcp90.dll.??0?$basic_ifstream@DU?$char_traits@D@std@@@std@@QAE@PBDHH@Z called in 32-bit code (0x7b83bbb2).
# -> http://bugs.winehq.org/show_bug.cgi?id=28228#c15

# you should then be stuck on:
# Unhandled exception: unimplemented function msvcp90.dll.??0?$basic_ostringstream@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QAE@H@Z called in 32-bit code (0x7b83bcb2).
# See: http://bugs.winehq.org/show_bug.cgi?id=26832

#winetricks vcrun2008
#winetricks --no-isolate psdk2003

# impossible to install SDK Win7:
#winetricks --no-isolate psdkwin7

# local vs2008express installation:
cd $HOME/.cache/winetricks/vc2008express
wget -c http://download.microsoft.com/download/e/8/e/e8eeb394-7f42-4963-a2d8-29559b738298/VS2008ExpressWithSP1ENUX1504728.iso
7z x VS2008ExpressWithSP1ENUX1504728.iso
# http://appdb.winehq.org/objectManager.php?sClass=version&iId=11210
winetricks dotnet35
wine VCExpress/autorun.exe
