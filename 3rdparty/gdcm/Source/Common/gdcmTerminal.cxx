/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTerminal.h"

#include <iostream>
#include <iostream>
#include <fstream>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h> /* SetConsoleTextAttribute */
#endif

// FIXME on ming32 a couple of stuff are not defined:
#ifndef COMMON_LVB_REVERSE_VIDEO
#define COMMON_LVB_REVERSE_VIDEO 0x4000
#endif
#ifndef COMMON_LVB_UNDERSCORE
#define COMMON_LVB_UNDERSCORE 0x8000
#endif

namespace gdcm
{

namespace terminal
{

class ConsoleImp
{
private:
// console implementation details:
#ifdef WIN32
  HANDLE hConsoleHandle;
  CONSOLE_SCREEN_BUFFER_INFO ConsoleInfo;
  WORD wNormalAttributes;
#endif
// vt100 implementation details:
  int attribute;
  int fgcolor;
  int bgcolor;
public:
  ConsoleImp()
  {
#ifdef WIN32
    hConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    GetConsoleScreenBufferInfo(hConsoleHandle, &ConsoleInfo);
    wNormalAttributes = ConsoleInfo.wAttributes;
#endif
    attribute = fgcolor = bgcolor = 9;
  }
  ~ConsoleImp()
  {
#ifdef WIN32
    SetConsoleTextAttribute(hConsoleHandle, wNormalAttributes);
  }
  WORD get_attributes() {
    GetConsoleScreenBufferInfo(hConsoleHandle, &ConsoleInfo);
    return ConsoleInfo.wAttributes;
#endif
  }

  void setattribute(int att) { attribute = att; }
  void setfgcolor(int col) { fgcolor = col; }
  void setbgcolor(int col) { bgcolor = col; }
  //std::string resettextcolor() const {
  //  char command[13];
  //  sprintf(command, "%c[%d;%d;%dm", 0x1B, 0, 0, 0);
  //  return command;
  //}
  std::string textcolor() const {
    char command[16];
    int n = sprintf(command, "%c[%d;%d;%dm", 0x1B, attribute, fgcolor + 30, bgcolor + 40);
    assert( n < 16 ); (void)n;
    return command;
  }
  void set_attributes(int color) {
#ifdef WIN32
  static const int colors[8] = { 0, 4, 2, 6, 1, 5, 3, 7 };
  WORD wAttributes;

  wAttributes = get_attributes();
  // http://swapoff.org/browser/todo/trunk/util/Terminal.cc
  // http://www.koders.com/cpp/fid5D5965EDC640274BE13A63CFEC649FA76F65A59D.aspx
  // http://cvs.4suite.org/viewcvs/4Suite/Ft/Lib/Terminal.py?rev=1.1&content-type=text/vnd.viewcvs-markup
  // http://linuxgazette.net/issue65/padala.html
  // https://svn.linux.ncsu.edu/svn/cls/branches/ncsu-gdm/pre-gdm-2.14/Xdefaults.old
  // http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/475116
  // http://techpubs.sgi.com/library/tpl/cgi-bin/getdoc.cgi?coll=linux&db=man&fname=/usr/share/catman/man4/console_codes.4.html
  // http://www.columbia.edu/kermit/ftp/k95/terminal.txt
  // http://www.codeproject.com/KB/cpp/Colored_Conslole_Messages.aspx
  // http://www.betarun.com/Pages/ConsoleColor/
  // http://support.microsoft.com/?scid=kb%3Ben-us%3B319883&x=20&y=8
  //http://www.dreamincode.net/code/snippet921.htm
  //http://www.codeproject.com/KB/cpp/Colored_Conslole_Messages.aspx
  //http://fabrizio.net/ccode/Old/20070427/Console.cpp
  //http://www.opensource.apple.com/darwinsource/10.4.8.x86/tcsh-46/tcsh/win32/console.c
  //http://msdn2.microsoft.com/en-us/library/ms682088(VS.85).aspx
  //
    {
    int n = color;

    if      (n == 0)        // Normal (default)
      wAttributes = wNormalAttributes;
    else if (n == 1)        // Bold
      wAttributes |= FOREGROUND_INTENSITY;
    else if (n == 4)        // Underlined
      wAttributes |= COMMON_LVB_UNDERSCORE;
    else if (n == 5)        // Blink (appears as BACKGROUND_INTENSITY)
      wAttributes |= BACKGROUND_INTENSITY;
    else if (n == 7)        // Inverse
      wAttributes |= COMMON_LVB_REVERSE_VIDEO;
    else if (n == 21)        // Not bold
      wAttributes &= ~FOREGROUND_INTENSITY;
    else if (n == 24)        // Not underlined
      wAttributes &= ~COMMON_LVB_UNDERSCORE;
    else if (n == 25)        // Steady (not blinking)
      wAttributes &= ~BACKGROUND_INTENSITY;
    else if (n == 27)        // Positive (not inverse)
      wAttributes &= ~COMMON_LVB_REVERSE_VIDEO;
    else if (30 <= n && n <= 37)  // Set foreground color
      wAttributes = (wAttributes & ~0x0007) | colors[n - 30];
    else if (n == 39)        // Set foreground color to default
      wAttributes = (wAttributes & ~0x0007) | (wNormalAttributes & 0x0007);
    else if (40 <= n && n <= 47)  // Set background color
      wAttributes = (wAttributes & ~0x0070) | (colors[n - 40] << 4);
    else if (n == 49)        // Set background color to default
      wAttributes = (wAttributes & ~0x0070) | (wNormalAttributes & 0x0070);
    else if (90 <= n && n <= 97)  // Set foreground color (bright)
      wAttributes = (wAttributes & ~0x0007) | colors[n - 90]
        | FOREGROUND_INTENSITY;
    else if (100 <= n && n <= 107)  // Set background color (bright)
      wAttributes = (wAttributes & ~0x0070) | (colors[n - 100] << 4)
        | BACKGROUND_INTENSITY;
    else              // (default)
      wAttributes = wNormalAttributes;
    }

  // Though Windows' console supports COMMON_LVB_REVERSE_VIDEO,
  // it seems to be buggy.  So we must simulate it.
  if (wAttributes & COMMON_LVB_REVERSE_VIDEO)
    wAttributes = (wAttributes & COMMON_LVB_UNDERSCORE)
      | ((wAttributes & 0x00f0) >> 4) | ((wAttributes & 0x000f) << 4);
  SetConsoleTextAttribute(hConsoleHandle, wAttributes);
#else
  (void)color;
#endif //WIN32
}

};
// http://linuxgazette.net/issue65/padala.html
//  The Color Code:     <ESC>[{attr};{fg};{bg}m

static ConsoleImp cimp;
static Mode mode;

void setmode( Mode m)
{
  mode = m;
}
std::string setfgcolor( Color c)
{
  if( mode == VT100 )
    {
    cimp.setfgcolor(c);
    return cimp.textcolor();
    }
  else if( mode == CONSOLE )
    cimp.set_attributes(30+c);
  return "";
}
std::string setbgcolor( Color c )
{
  if( mode == VT100 )
    {
    cimp.setbgcolor(c);
    return cimp.textcolor();
    }
  else if( mode == CONSOLE )
    cimp.set_attributes(40+c);
  return "";
}

std::string setattribute( Attribute att )
{
  if( mode == VT100 )
    {
    cimp.setattribute(att);
    return cimp.textcolor();
    }
  else if( mode == CONSOLE )
    cimp.set_attributes(att);
  return "";
}

}


} // end namespace gdcm
