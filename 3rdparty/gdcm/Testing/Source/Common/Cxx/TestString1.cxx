/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmString.h"
#include <iostream>

typedef gdcm::String<'\\',64> CSComp;

static void Print(CSComp v)
{
  std::cout << v << std::endl;
}

int TestString1(int , char *[])
{
{
  gdcm::String<'\\'> s = "coucou";
  std::cout << s << std::endl;

  const char str[] = "WINDOW1\\WINDOW2\\WINDOW3";
  //const size_t lenstr = strlen(str);

  gdcm::String<'\\'> ms1, ms2, ms3;
  std::stringstream ss;
  ss << str;
  ss >> ms1;
  ss.get(); // discard backslash
  std::cout << ms1 << std::endl;
  if( ms1 != "WINDOW1" ) return 1;
  ss >> ms2;
  ss.get();
  std::cout << ms2 << std::endl;
  if( ms2 != "WINDOW2" ) return 1;
  ss >> ms3;
  ss.get();
  std::cout << ms3 << std::endl;
  if( ms3 != "WINDOW3" ) return 1;

  // we are at the end:
  if( !!ss )
    {
    std::cerr << "not at the end" << std::endl;
    return 1;
    }
}
{
  gdcm::String<'^'> s = "coucou";
  std::cout << s << std::endl;

  const char str[] = "WINDOW1^WINDOW2^WINDOW3";
  //const size_t lenstr = strlen(str);

  gdcm::String<'^'> ms1, ms2, ms3;
  std::stringstream ss;
  ss << str;
  ss >> ms1;
  ss.get(); // discard backslash
  std::cout << ms1 << std::endl;
  if( ms1 != "WINDOW1" ) return 1;
  ss >> ms2;
  ss.get();
  std::cout << ms2 << std::endl;
  if( ms2 != "WINDOW2" ) return 1;
  ss >> ms3;
  ss.get();
  std::cout << ms3 << std::endl;
  if( ms3 != "WINDOW3" ) return 1;

  // we are at the end:
  if( !!ss ) return 1;
}
{
  gdcm::String<> s = "coucou";
  std::cout << s << std::endl;

  const char str[] = "WINDOW1^WINDOW2^WINDOW3";
  //const size_t lenstr = strlen(str);

  gdcm::String<> ms1;
  std::stringstream ss;
  ss << str;
  ss >> ms1;
  std::cout << ms1 << std::endl;
  ss.get(); // discard \n
  if ( ms1 != str ) return 1;

  // we are at the end:
  if( !!ss ) return 1;
}

  std::string privatecreator = " CREATOR  SMS-AX  ";
  std::cout << "[" << privatecreator << "]" << std::endl;
  privatecreator.erase(privatecreator.find_last_not_of(' ') + 1);
  std::cout << "[" << privatecreator << "]" << std::endl;

  static const CSComp values[] = {"DERIVED","SECONDARY"};
  std::cout << values[0] << std::endl;
  Print( values[0] );

  const char trim[] = "8 ";
  gdcm::String<> strim( trim );
  std::cout << "|" << strim.Trim() << "|" << std::endl;

  return 0;
}
