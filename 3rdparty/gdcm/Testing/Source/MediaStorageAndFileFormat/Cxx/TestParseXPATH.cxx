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
#include "gdcmDataSet.h"
#include "gdcmDict.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmStringFilter.h"
#include "gdcmTesting.h"
#include "gdcmFilename.h"

static bool CheckResult( std::string const & filename, std::string const & value )
{
  if(
       filename == "D_CLUNIE_MR3_JPLY.dcm"
    || filename == "D_CLUNIE_RG3_JPLY.dcm"
    || filename == "D_CLUNIE_NM1_JPLY.dcm"
    || filename == "D_CLUNIE_MR4_JPLY.dcm"
    || filename == "D_CLUNIE_CT1_J2KI.dcm"
    || filename == "D_CLUNIE_CT1_JLSN.dcm"
    || filename == "D_CLUNIE_MR1_JPLY.dcm"
    || filename == "D_CLUNIE_SC1_JPLY.dcm"
    || filename == "D_CLUNIE_MR2_JPLY.dcm"
    || filename == "D_CLUNIE_RG2_JPLY.dcm"
    || filename == "D_CLUNIE_XA1_JPLY.dcm" )
    {
    return value == "Lossy Compression ";
    }
  else if ( filename == "JPEG_LossyYBR.dcm"
         || filename == "MEDILABInvalidCP246_EVRLESQasUN.dcm" )
    return value ==  "Full fidelity image, uncompressed or lossless compressed";
  else if ( filename == "NM-PAL-16-PixRep1.dcm" )
    return value == "Full fidelity image ";
  else
    return value == "";
}

int TestParseXPATHFile(const char* filename, bool verbose = false )
{
  (void)verbose;
  //static gdcm::Global &g = gdcm::Global::GetInstance();
  //static const gdcm::Dicts &dicts = g.GetDicts();
  //static const gdcm::Dict &pubdict = dicts.GetPublicDict();

  gdcm::Reader reader;
//  reader.SetFileName( "/home/mathieu/Creatis/gdcmData/D_CLUNIE_CT1_J2KI.dcm" );
  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    return 1;
    }

  gdcm::StringFilter sf;
  sf.SetFile( reader.GetFile() );

  const char query_const[] = "/DicomNativeModel/DicomAttribute[@keyword='DerivationCodeSequence']/Item[@number=1]//DicomAttribute[@keyword='CodeMeaning']/Value[@number=1]";

  std::string value;
  bool ret = sf.ExecuteQuery( query_const, value );

  if( !ret )
    {
    return 1;
    }

  gdcm::Filename fn( filename );

  bool b = CheckResult( fn.GetName(), value );
  if( !b )
    {
    std::cerr << "Problem with: " << filename << " -> " << value << std::endl;
    }

  return !b;
}

int TestParseXPATH(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestParseXPATHFile(filename, true);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestParseXPATHFile(filename);
    ++i;
    }
  return EXIT_SUCCESS;
}
