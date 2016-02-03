/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMPolyDataReader.h"
#include "vtkMedicalImageProperties.h"

#include "gdcmDirectory.h"
#include "gdcmTesting.h"
#include "gdcmTrace.h"

static const char * rtstruct_files[] = {
  "gdcmNonImageData/RTSTRUCT_1.3.6.1.4.1.22213.1.1396.2.dcm",
  "gdcmNonImageData/RT/RTStruct.dcm",
  "gdcmNonImageData/exRT_Structure_Set_Storage.dcm",
  0
};

static const int rtstruct_files_nb[] = {
  9,
  4,
  10
};

int TestvtkGDCMPolyDataRead(const char *filename, int nb, bool verbose)
{
  int ret = 0;
  if( verbose )
    std::cerr << "Reading : " << filename << std::endl;

  vtkGDCMPolyDataReader *reader = vtkGDCMPolyDataReader::New();
  reader->SetFileName( filename );

  //int canread = reader->CanReadFile( filename );
  reader->Update();

  if( verbose )
    {
    reader->GetOutput()->Print( cout );
    reader->GetMedicalImageProperties()->Print( cout );
    }

  //std::cout << reader->GetNumberOfOutputPorts() << std::endl;
  if( nb != -1 && reader->GetNumberOfOutputPorts() != nb )
    {
    ret = 1;
    }

  reader->Delete();
  return ret;
}

int TestvtkGDCMPolyDataReader(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMPolyDataRead(filename, -1, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  //const char * const *filenames = gdcm::Testing::GetFileNames();
  //gdcmDataExtra
  const char *extradataroot = gdcm::Testing::GetDataExtraRoot();
  while( (filename = rtstruct_files[i]) )
    {
    std::string fullpath = extradataroot;
    fullpath += "/";
    fullpath += filename;
    r += TestvtkGDCMPolyDataRead( fullpath.c_str(), rtstruct_files_nb[i], true);
    ++i;
    }

  return r;
}
