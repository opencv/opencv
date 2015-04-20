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
#include "vtkImageData.h"
#include "vtkImageActor.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkInteractorStyleImage.h"
#include <vtksys/SystemTools.hxx>

#include "gdcmTesting.h"

int TestvtkGDCMReadImageActor(const char *filename)
{
  vtkGDCMImageReader *reader = vtkGDCMImageReader::New();
  //reader->CanReadFile( filename );
  std::cerr << "Reading : " << filename << std::endl;
  reader->SetFileName( filename );
  reader->Update();

  //reader->GetOutput()->Print( cout );

  vtkImageActor *ia = vtkImageActor::New();
  ia->SetInput( reader->GetOutput() );

  // Create the RenderWindow, Renderer and both Actors
  vtkRenderer *ren1 = vtkRenderer::New();
  vtkRenderWindow *renWin = vtkRenderWindow::New();
  renWin->AddRenderer (ren1);
  vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
  iren->SetRenderWindow (renWin);

  // Add the actors to the renderer, set the background and size
  ren1->AddActor (ia);

  vtkInteractorStyleImage *style = vtkInteractorStyleImage::New();
  iren->SetInteractorStyle( style );
  style->Delete();
  iren->Initialize();
  iren->Start();

  reader->Delete();

  return 0;
}

int TestvtkGDCMImageActor(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMReadImageActor(filename);
    }

  // else
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestvtkGDCMReadImageActor( filename );
    ++i;
    }

  return r;
}
