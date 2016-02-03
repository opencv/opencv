/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmReader.h"
#include "gdcmXMLPrinter.h"
#include "gdcmFilename.h"
#include "gdcmTesting.h"

int TestXMLPrint(const char *filename, bool verbose= false)
{
  gdcm::Reader r;
  r.SetFileName( filename );
  if( !r.Read() )
    {
    std::cerr << "Could not read: " << filename << std::endl;
    return 1;
    }

  gdcm::XMLPrinter print;
  //print.SetStyle( gdcm::XMLPrinter::LOADBULKDATA );
  print.SetFile( r.GetFile() );
  std::ostringstream out;
  if( verbose )
    print.Print( std::cout );
  print.Print( out );

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();

  std::string buf = out.str();
  if( buf.find( "GDCM:UNKNOWN" ) != std::string::npos )
    {
    std::cerr << "UNKNOWN Attribute with : " << name << std::endl;
    return 1;
    }
  return 0;
}


int TestXMLPrinter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestXMLPrint(filename, true);
    }

  // else
  int r = 0, i = 0;
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestXMLPrint( filename );
    ++i;
    }

  return r;
}
