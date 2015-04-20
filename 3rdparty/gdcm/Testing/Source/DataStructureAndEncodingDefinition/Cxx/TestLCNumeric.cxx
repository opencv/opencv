/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAttribute.h"

#include <stdlib.h> /* setenv */
#include <locale.h>


int TestLCNumeric(int , char *[])
{
  //setenv("LC_NUMERIC", "fr_FR", 1);
  //const char ss[] = "LC_NUMERIC=fr_FR";
  //setlocale(LC_NUMERIC,"C");

  // Seems to only affect the local sscanf
  char *l = setlocale(LC_NUMERIC,"fr_FR.UTF-8");
  if( !l )
    {
    std::cerr << "Could not set LC_NUMERIC" << std::endl;
    return 1;
    }

  float a = 1. / 3;

  printf("Float: %f\n", a );

  // The following affect all ostringstream
  try
    {
    //std::locale b = std::locale( "fr_FR" );
    std::locale::global( std::locale( "fr_FR.UTF-8" ) ) ;

    //char *copy = strdup(ss);
    //putenv(copy);
    //free(copy);
    std::ostringstream os;
    os.imbue(std::locale::classic());
    double d = 1.2;
    os << d;
    std::string s = os.str();
    std::cout << "s:" << s << std::endl;
    std::string::size_type pos_comma = s.find( "," );
    if( pos_comma != std::string::npos )
      {
      std::cerr << "We found the comma symbol" << std::endl;
      return 1;
      }
    std::string::size_type pos_dot = s.find( "." );
    if( pos_dot == std::string::npos )
      {
      std::cerr << "We did not found the dot symbol" << std::endl;
      return 1;
      }
    }
  catch(std::exception &ex)
    {
    // ok something went wrong when setting up fr_FR locale.
    // just ignore for now
    std::cerr << "What: " << ex.what() << std::endl;
    }

  return 0;
}
