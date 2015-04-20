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

namespace term = gdcm::terminal;

void TestAll()
{
  std::cout << term::setattribute( term::bright ) << "bright" << std::endl;
  std::cout << term::setattribute( term::dim ) << "dim" << std::endl;
  std::cout << term::setattribute( term::underline ) << "underline" << std::endl;
  std::cout << term::setattribute( term::blink ) << "blink" << std::endl;
  std::cout << term::setattribute( term::reverse ) << "reverse" << std::endl;
  std::cout << term::setattribute( term::reset ) << "reset" << std::endl;
  std::cout << term::setfgcolor( term::black ) << "fg:black" << std::endl;
  std::cout << term::setfgcolor( term::red ) << "fg:red " << std::endl;
  std::cout << term::setfgcolor( term::green ) << "fg:green" << std::endl;
  std::cout << term::setfgcolor( term::yellow ) << "fg:yellow" << std::endl;
  std::cout << term::setfgcolor( term::blue ) << "fg:blue" << std::endl;
  std::cout << term::setfgcolor( term::magenta ) << "fg:magenta" << std::endl;
  std::cout << term::setfgcolor( term::cyan ) << "fg:cyan" << std::endl;
  std::cout << term::setfgcolor( term::white ) << "fg:white" << std::endl;
  std::cout << term::setattribute( term::reverse ) << term::setfgcolor( term::white ) << "fg:white" << std::endl;
  std::cout << term::setbgcolor( term::black ) << "bg:black" << std::endl;
  std::cout << term::setbgcolor( term::red ) << "bg:red " << std::endl;
  std::cout << term::setbgcolor( term::green ) << "bg:green" << std::endl;
  std::cout << term::setbgcolor( term::yellow ) << "bg:yellow" << std::endl;
  std::cout << term::setbgcolor( term::blue ) << "bg:blue" << std::endl;
  std::cout << term::setbgcolor( term::magenta ) << "bg:magenta" << std::endl;
  std::cout << term::setbgcolor( term::cyan ) << "bg:cyan" << std::endl;
  std::cout << term::setbgcolor( term::white ) << "bg:white" << std::endl;
  std::cout << term::setattribute( term::reset ) << "reset" << std::endl;
  //std::cerr << term::setbgcolor( term::blue ) << "cerr:bg:blue" << std::endl;
}

int TestTerminal(int , char *[])
{
  // Typically for WIN32
  term::setmode( term::CONSOLE );
  TestAll();
  // For all *NIX
  // rxvt is WIN32 app, but is a VT100 compatible TERM
  term::setmode( term::VT100 );
  TestAll();

  return 0;
}
