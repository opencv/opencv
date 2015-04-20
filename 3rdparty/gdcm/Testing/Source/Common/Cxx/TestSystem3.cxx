/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSystem.h"
#include "gdcmTesting.h"

#include <vector>
#include <string.h> // strdup

int TestSystem3(int, char *[])
{
  const char isostr[] = "\\ISO 2022 IR 13\\ISO 2022 IR 87";

  const char delim[] = "\\";
  char *token;
{
  char *query = strdup( isostr );
  char *str1;
  char *saveptr1;

  std::vector< std::string > v;
  for (str1 = query; ; str1 = NULL)
    {
    token = gdcm::System::StrTokR(str1, delim, &saveptr1);
    if (token == NULL)
      break;
    //std::cout << "[" << token << "]" << std::endl;
    v.push_back( token );
    }
  free( query );

  if( v.size() != 2 ) return 1;
  if( v[0] != "ISO 2022 IR 13" ) return 1;
  if( v[1] != "ISO 2022 IR 87" ) return 1;
}

{
  std::vector< std::string > v;
  char *string = strdup( isostr );
  if(!string) return 1;
  char *copy = string;
  while ((token = gdcm::System::StrSep(&string, delim)) != NULL)
    {
    //printf("token=%s\n", token);
    v.push_back( token );
    }
  free( copy );
  if( v.size() != 3 ) return 1;
  if( v[0] != "" ) return 1;
  if( v[1] != "ISO 2022 IR 13" ) return 1;
  if( v[2] != "ISO 2022 IR 87" ) return 1;
}

  return 0;
}
