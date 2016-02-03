/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMImageReader.h"

#include "vtkImageData.h"
#include "vtkMultiThreader.h"
#include "vtkMedicalImageProperties.h"

#include "gdcmTesting.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmImageReader.h"

#ifndef vtkFloatingPointType
#define vtkFloatingPointType float
#endif

int TestvtkGDCMImageReaderIsLossyFunc(const char *filename, bool verbose = false)
{
  gdcm::Filename fn = filename;
  vtkGDCMImageReader *reader = vtkGDCMImageReader::New();
  int canread = reader->CanReadFile( filename );
  int res = 0;
  if( canread )
    {
    reader->SetFileName( filename );
    reader->Update();
    if( verbose )
      {
      reader->GetOutput()->Print( cout );
      reader->GetMedicalImageProperties()->Print( cout );
      }
    int reflossy = gdcm::Testing::GetLossyFlagFromFile( filename );
    if( reader->GetLossyFlag() != reflossy )
      {
      std::cerr << "Mismatch for " << filename << std::endl;
      ++res;
      }
    }
  else
    {
    std::cerr << "Could not read: " << filename << std::endl;
    //++res;
    }
  reader->Delete();

  return res;
}

int TestvtkGDCMImageReaderIsLossy(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMImageReaderIsLossyFunc(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestvtkGDCMImageReaderIsLossyFunc( filename );
    ++i;
    }

  return r;
}
