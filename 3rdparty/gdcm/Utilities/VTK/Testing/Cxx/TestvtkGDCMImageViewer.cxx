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

#include "vtkPNGWriter.h"
#include "vtkImageColorViewer.h"
#include "vtkImageData.h"
#include "vtkImageActor.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkInteractorStyleImage.h"
#include <vtksys/SystemTools.hxx>

#include "gdcmTesting.h"

int TestvtkGDCMReadImageViewer(const char *filename)
{
  vtkGDCMImageReader *reader = vtkGDCMImageReader::New();
  //reader->CanReadFile( filename );
  std::cerr << "Reading : " << filename << std::endl;
  reader->SetFileName( filename );
  reader->Update();

  reader->GetOutput()->Print( cout );


  vtkImageColorViewer *viewer = vtkImageColorViewer::New();
  viewer->SetInput( reader->GetOutput() );


  vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();

  viewer->SetupInteractor( iren );
  viewer->Render();

  iren->Initialize();
  iren->Start();

  reader->Delete();
  iren->Delete();

  return 0;
}

int TestvtkGDCMImageViewer(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMReadImageViewer(filename);
    }

  // else
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestvtkGDCMReadImageViewer( filename );
    ++i;
    }

  return r;
}
