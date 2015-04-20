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

#include "vtkImageNoiseSource.h"
#include "vtkGDCMImageWriter.h"

#include "gdcmTesting.h"
#include "gdcmSystem.h"

int TestvtkGDCMImageWriterIsLossy(int , char *[])
{
  vtkImageNoiseSource * noise = vtkImageNoiseSource::New();
  noise->SetWholeExtent(1,256,1,256,0,0);
  noise->SetMinimum(0.0);
  noise->SetMaximum(255.0);

  // Create directory first:
  const char subdir[] = "TestvtkGDCMImageWriterIsLossy";
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  const char *filename = "noise.dcm";
  std::string gdcmfile = gdcm::Testing::GetTempFilename( filename, subdir );

  vtkGDCMImageWriter *writer = vtkGDCMImageWriter::New();
#if (VTK_MAJOR_VERSION >= 6)
  writer->SetInputConnection( noise->GetOutputPort() );
#else
  writer->SetInput( noise->GetOutput() );
#endif
  writer->SetShift( 0 );
  writer->SetScale( 1.5 );
  writer->SetLossyFlag( 1 );
  writer->SetFileName( gdcmfile.c_str() );
  writer->Write();

  noise->Delete();
  writer->Delete();

  vtkGDCMImageReader * reader = vtkGDCMImageReader::New();
  reader->SetFileName( gdcmfile.c_str() );
  reader->Update();

  int lossyflag = reader->GetLossyFlag();
  reader->Delete();

  if( lossyflag != 1 ) return 1;

  return 0;
}
