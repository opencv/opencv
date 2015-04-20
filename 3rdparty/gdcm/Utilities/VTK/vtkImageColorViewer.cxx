/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*=========================================================================

  Portions of this file are subject to the VTK Toolkit Version 3 copyright.

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageColorViewer.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageColorViewer.h"

#include "vtkCamera.h"
#include "vtkCommand.h"
#include "vtkImageActor.h"
#include "vtkImageData.h"
#if (VTK_MAJOR_VERSION >= 6)
#include "vtkImageMapper3D.h"
#endif
#include "vtkImageData.h"
#include "vtkImageMapToWindowLevelColors2.h"
#include "vtkInteractorStyleImage.h"
#include "vtkObjectFactory.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkPolyDataMapper.h"
#if (VTK_MAJOR_VERSION >= 5)
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#endif

vtkCxxRevisionMacro(vtkImageColorViewer, "$Revision: 1.3 $")
vtkStandardNewMacro(vtkImageColorViewer)

//----------------------------------------------------------------------------
vtkImageColorViewer::vtkImageColorViewer()
{
  this->RenderWindow    = NULL;
  this->Renderer        = NULL;
  this->ImageActor      = vtkImageActor::New();
  this->OverlayImageActor      = vtkImageActor::New();
  this->WindowLevel     = vtkImageMapToWindowLevelColors2::New();
  this->Interactor      = NULL;
  this->InteractorStyle = NULL;

  this->Slice = 0;
  this->FirstRender = 1;
  this->SliceOrientation = vtkImageColorViewer::SLICE_ORIENTATION_XY;

  // Setup the pipeline

  vtkRenderWindow *renwin = vtkRenderWindow::New();
  this->SetRenderWindow(renwin);
  renwin->Delete();

  vtkRenderer *ren = vtkRenderer::New();
  this->SetRenderer(ren);
  ren->Delete();

  this->InstallPipeline();
}

//----------------------------------------------------------------------------
vtkImageColorViewer::~vtkImageColorViewer()
{
  if (this->WindowLevel)
    {
    this->WindowLevel->Delete();
    this->WindowLevel = NULL;
    }

  if (this->ImageActor)
    {
    this->ImageActor->Delete();
    this->ImageActor = NULL;
    }

  if (this->OverlayImageActor)
    {
    this->OverlayImageActor->Delete();
    this->OverlayImageActor = NULL;
    }

  if (this->Renderer)
    {
    this->Renderer->Delete();
    this->Renderer = NULL;
    }

  if (this->RenderWindow)
    {
    this->RenderWindow->Delete();
    this->RenderWindow = NULL;
    }

  if (this->Interactor)
    {
    this->Interactor->Delete();
    this->Interactor = NULL;
    }

  if (this->InteractorStyle)
    {
    this->InteractorStyle->Delete();
    this->InteractorStyle = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetupInteractor(vtkRenderWindowInteractor *arg)
{
  if (this->Interactor == arg)
    {
    return;
    }

  this->UnInstallPipeline();

  if (this->Interactor)
    {
    this->Interactor->UnRegister(this);
    }

  this->Interactor = arg;

  if (this->Interactor)
    {
    this->Interactor->Register(this);
    }

  this->InstallPipeline();

  if (this->Renderer)
    {
    this->Renderer->GetActiveCamera()->ParallelProjectionOn();
    }
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetRenderWindow(vtkRenderWindow *arg)
{
  if (this->RenderWindow == arg)
    {
    return;
    }

  this->UnInstallPipeline();

  if (this->RenderWindow)
    {
    this->RenderWindow->UnRegister(this);
    }

  this->RenderWindow = arg;

  if (this->RenderWindow)
    {
    this->RenderWindow->Register(this);
    }

  this->InstallPipeline();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetRenderer(vtkRenderer *arg)
{
  if (this->Renderer == arg)
    {
    return;
    }

  this->UnInstallPipeline();

  if (this->Renderer)
    {
    this->Renderer->UnRegister(this);
    }

  this->Renderer = arg;

  if (this->Renderer)
    {
    this->Renderer->Register(this);
    }

  this->InstallPipeline();
  this->UpdateOrientation();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetSize(int a,int b)
{
  this->RenderWindow->SetSize(a, b);
}

//----------------------------------------------------------------------------
int* vtkImageColorViewer::GetSize()
{
  return this->RenderWindow->GetSize();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::GetSliceRange(int &min, int &max)
{
#if (VTK_MAJOR_VERSION >= 6)
  vtkAlgorithm *input = this->GetInputAlgorithm();
#else
  vtkImageData *input = this->GetInput();
#endif
  if (input)
    {
    input->UpdateInformation();
#if (VTK_MAJOR_VERSION >= 6)
    int *w_ext = input->GetOutputInformation(0)->Get(
      vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT());
#else
    int *w_ext = input->GetWholeExtent();
#endif
    min = w_ext[this->SliceOrientation * 2];
    max = w_ext[this->SliceOrientation * 2 + 1];
    }
}

//----------------------------------------------------------------------------
int* vtkImageColorViewer::GetSliceRange()
{
#if (VTK_MAJOR_VERSION >= 6)
  vtkAlgorithm *input = this->GetInputAlgorithm();
#else
  vtkImageData *input = this->GetInput();
#endif
  if (input)
    {
    input->UpdateInformation();
#if (VTK_MAJOR_VERSION >= 6)
    return input->GetOutputInformation(0)->Get(
      vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()) +
      this->SliceOrientation * 2;
#else
    return input->GetWholeExtent() + this->SliceOrientation * 2;
#endif
    }
  return NULL;
}

//----------------------------------------------------------------------------
int vtkImageColorViewer::GetSliceMin()
{
  int *range = this->GetSliceRange();
  if (range)
    {
    return range[0];
    }
  return 0;
}

//----------------------------------------------------------------------------
int vtkImageColorViewer::GetSliceMax()
{
  int *range = this->GetSliceRange();
  if (range)
    {
    return range[1];
    }
  return 0;
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetSlice(int slice)
{
  int *range = this->GetSliceRange();
  if (range)
    {
    if (slice < range[0])
      {
      slice = range[0];
      }
    else if (slice > range[1])
      {
      slice = range[1];
      }
    }

  if (this->Slice == slice)
    {
    return;
    }

  this->Slice = slice;
  this->Modified();

  this->UpdateDisplayExtent();
  this->Render();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetSliceOrientation(int orientation)
{
  if (orientation < vtkImageColorViewer::SLICE_ORIENTATION_YZ ||
      orientation > vtkImageColorViewer::SLICE_ORIENTATION_XY)
    {
    vtkErrorMacro("Error - invalid slice orientation " << orientation);
    return;
    }

  if (this->SliceOrientation == orientation)
    {
    return;
    }

  this->SliceOrientation = orientation;

  // Update the viewer

  int *range = this->GetSliceRange();
  if (range)
    {
    this->Slice = static_cast<int>((range[0] + range[1]) * 0.5);
    }

  this->UpdateOrientation();
  this->UpdateDisplayExtent();

  if (this->Renderer && this->GetInput())
    {
    double scale = this->Renderer->GetActiveCamera()->GetParallelScale();
    this->Renderer->ResetCamera();
    this->Renderer->GetActiveCamera()->SetParallelScale(scale);
    }

  this->Render();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::UpdateOrientation()
{
  // Set the camera position

  vtkCamera *cam = this->Renderer ? this->Renderer->GetActiveCamera() : NULL;
  if (cam)
    {
    switch (this->SliceOrientation)
      {
      case vtkImageColorViewer::SLICE_ORIENTATION_XY:
        cam->SetFocalPoint(0,0,0);
        cam->SetPosition(0,0,1); // -1 if medical ?
        cam->SetViewUp(0,1,0);
        break;

      case vtkImageColorViewer::SLICE_ORIENTATION_XZ:
        cam->SetFocalPoint(0,0,0);
        cam->SetPosition(0,-1,0); // 1 if medical ?
        cam->SetViewUp(0,0,1);
        break;

      case vtkImageColorViewer::SLICE_ORIENTATION_YZ:
        cam->SetFocalPoint(0,0,0);
        cam->SetPosition(1,0,0); // -1 if medical ?
        cam->SetViewUp(0,0,1);
        break;
      }
    }
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::UpdateDisplayExtent()
{
#if (VTK_MAJOR_VERSION >= 6)
  vtkAlgorithm *input = this->GetInputAlgorithm();
#else
  vtkImageData *input = this->GetInput();
#endif
  if (!input || !this->ImageActor)
    {
    return;
    }

  input->UpdateInformation();
#if (VTK_MAJOR_VERSION >= 6)
  vtkInformation* outInfo = input->GetOutputInformation(0);
  int *w_ext = outInfo->Get(
    vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT());
#else
  int *w_ext = input->GetWholeExtent();
#endif

  // Is the slice in range ? If not, fix it

  int slice_min = w_ext[this->SliceOrientation * 2];
  int slice_max = w_ext[this->SliceOrientation * 2 + 1];
  if (this->Slice < slice_min || this->Slice > slice_max)
    {
    this->Slice = static_cast<int>((slice_min + slice_max) * 0.5);
    }

  // Set the image actor

  switch (this->SliceOrientation)
    {
    case vtkImageColorViewer::SLICE_ORIENTATION_XY:
      this->ImageActor->SetDisplayExtent(
        w_ext[0], w_ext[1], w_ext[2], w_ext[3], this->Slice, this->Slice);
      break;

    case vtkImageColorViewer::SLICE_ORIENTATION_XZ:
      this->ImageActor->SetDisplayExtent(
        w_ext[0], w_ext[1], this->Slice, this->Slice, w_ext[4], w_ext[5]);
      break;

    case vtkImageColorViewer::SLICE_ORIENTATION_YZ:
      this->ImageActor->SetDisplayExtent(
        this->Slice, this->Slice, w_ext[2], w_ext[3], w_ext[4], w_ext[5]);
      break;
    }

  // Figure out the correct clipping range

  if (this->Renderer)
    {
    if (this->InteractorStyle &&
        this->InteractorStyle->GetAutoAdjustCameraClippingRange())
      {
      this->Renderer->ResetCameraClippingRange();
      }
    else
      {
      vtkCamera *cam = this->Renderer->GetActiveCamera();
      if (cam)
        {
        double bounds[6];
        this->ImageActor->GetBounds(bounds);
        double spos = (double)bounds[this->SliceOrientation * 2];
        double cpos = (double)cam->GetPosition()[this->SliceOrientation];
        double range = fabs(spos - cpos);
#if (VTK_MAJOR_VERSION >= 6)
        double *spacing = outInfo->Get(vtkDataObject::SPACING());
#else
        double *spacing = input->GetSpacing();
#endif
        double avg_spacing =
          ((double)spacing[0] + (double)spacing[1] + (double)spacing[2]) / 3.0;
        cam->SetClippingRange(
          range - avg_spacing * 3.0, range + avg_spacing * 3.0);
        }
      }
    }
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetPosition(int a,int b)
{
  this->RenderWindow->SetPosition(a, b);
}

//----------------------------------------------------------------------------
int* vtkImageColorViewer::GetPosition()
{
  return this->RenderWindow->GetPosition();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetDisplayId(void *a)
{
  this->RenderWindow->SetDisplayId(a);
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetWindowId(void *a)
{
  this->RenderWindow->SetWindowId(a);
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetParentId(void *a)
{
  this->RenderWindow->SetParentId(a);
}

//----------------------------------------------------------------------------
double vtkImageColorViewer::GetColorWindow()
{
  return this->WindowLevel->GetWindow();
}

//----------------------------------------------------------------------------
double vtkImageColorViewer::GetColorLevel()
{
  return this->WindowLevel->GetLevel();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetColorWindow(double s)
{
  this->WindowLevel->SetWindow(s);
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetColorLevel(double s)
{
  this->WindowLevel->SetLevel(s);
}

//----------------------------------------------------------------------------
class vtkImageColorViewerCallback : public vtkCommand
{
public:
  static vtkImageColorViewerCallback *New() { return new vtkImageColorViewerCallback; }

  void Execute(vtkObject *caller,
               unsigned long event,
               void *vtkNotUsed(callData))
    {
      if (this->IV->GetInput() == NULL)
        {
        return;
        }

      // Reset

      if (event == vtkCommand::ResetWindowLevelEvent)
        {
#if (VTK_MAJOR_VERSION >= 6)
        this->IV->GetInputAlgorithm()->UpdateInformation();
        vtkStreamingDemandDrivenPipeline::SetUpdateExtent(
          this->IV->GetInputInformation(),
          vtkStreamingDemandDrivenPipeline::GetWholeExtent(
            this->IV->GetInputInformation()));
        this->IV->GetInputAlgorithm()->Update();
#else
        this->IV->GetInput()->UpdateInformation();
        this->IV->GetInput()->SetUpdateExtent
          (this->IV->GetInput()->GetWholeExtent());
        this->IV->GetInput()->Update();
#endif
        double *range = this->IV->GetInput()->GetScalarRange();
        this->IV->SetColorWindow(range[1] - range[0]);
        this->IV->SetColorLevel(0.5 * (range[1] + range[0]));
        this->IV->Render();
        return;
        }

      // Start

      if (event == vtkCommand::StartWindowLevelEvent)
        {
        this->InitialWindow = this->IV->GetColorWindow();
        this->InitialLevel = this->IV->GetColorLevel();
        return;
        }

      // Adjust the window level here

      vtkInteractorStyleImage *isi =
        static_cast<vtkInteractorStyleImage *>(caller);

      int *size = this->IV->GetRenderWindow()->GetSize();
      double window = this->InitialWindow;
      double level = this->InitialLevel;

      // Compute normalized delta

      double dx = 4.0 *
        (isi->GetWindowLevelCurrentPosition()[0] -
         isi->GetWindowLevelStartPosition()[0]) / size[0];
      double dy = 4.0 *
        (isi->GetWindowLevelStartPosition()[1] -
         isi->GetWindowLevelCurrentPosition()[1]) / size[1];

      // Scale by current values

      if (fabs(window) > 0.01)
        {
        dx = dx * window;
        }
      else
        {
        dx = dx * (window < 0 ? -0.01 : 0.01);
        }
      if (fabs(level) > 0.01)
        {
        dy = dy * level;
        }
      else
        {
        dy = dy * (level < 0 ? -0.01 : 0.01);
        }

      // Abs so that direction does not flip

      if (window < 0.0)
        {
        dx = -1*dx;
        }
      if (level < 0.0)
        {
        dy = -1*dy;
        }

      // Compute new window level

      double newWindow = dx + window;
      double newLevel;
      newLevel = level - dy;

      // Stay away from zero and really

      if (fabs(newWindow) < 0.01)
        {
        newWindow = 0.01*(newWindow < 0 ? -1 : 1);
        }
      if (fabs(newLevel) < 0.01)
        {
        newLevel = 0.01*(newLevel < 0 ? -1 : 1);
        }

      this->IV->SetColorWindow(newWindow);
      this->IV->SetColorLevel(newLevel);
      this->IV->Render();
    }

  vtkImageColorViewer *IV;
  double InitialWindow;
  double InitialLevel;
};

//----------------------------------------------------------------------------
void vtkImageColorViewer::InstallPipeline()
{
  if (this->RenderWindow && this->Renderer)
    {
    this->RenderWindow->AddRenderer(this->Renderer);
    }

  if (this->Interactor)
    {
    if (!this->InteractorStyle)
      {
      this->InteractorStyle = vtkInteractorStyleImage::New();
      vtkImageColorViewerCallback *cbk = vtkImageColorViewerCallback::New();
      cbk->IV = this;
      this->InteractorStyle->AddObserver(
        vtkCommand::WindowLevelEvent, cbk);
      this->InteractorStyle->AddObserver(
        vtkCommand::StartWindowLevelEvent, cbk);
      this->InteractorStyle->AddObserver(
        vtkCommand::ResetWindowLevelEvent, cbk);
      cbk->Delete();
      }

    this->Interactor->SetInteractorStyle(this->InteractorStyle);
    this->Interactor->SetRenderWindow(this->RenderWindow);
    }

  if (this->Renderer && this->ImageActor)
    {
    this->Renderer->AddViewProp(this->ImageActor);
    }

  if (this->ImageActor && this->WindowLevel)
    {
#if (VTK_MAJOR_VERSION >= 6)
    this->ImageActor->GetMapper()->SetInputConnection(
      this->WindowLevel->GetOutputPort());
#else
    this->ImageActor->SetInput(this->WindowLevel->GetOutput());
#endif
    }
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::UnInstallPipeline()
{
  if (this->ImageActor)
    {
#if (VTK_MAJOR_VERSION >= 6)
    this->ImageActor->GetMapper()->SetInputConnection(NULL);
#else
    this->ImageActor->SetInput(NULL);
#endif
    }

  if (this->Renderer && this->ImageActor)
    {
    this->Renderer->RemoveViewProp(this->ImageActor);
    }

  if (this->RenderWindow && this->Renderer)
    {
    this->RenderWindow->RemoveRenderer(this->Renderer);
    }

  if (this->Interactor)
    {
    this->Interactor->SetInteractorStyle(NULL);
    this->Interactor->SetRenderWindow(NULL);
    }
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::Render()
{
  if (this->FirstRender)
    {
    // Initialize the size if not set yet

#if (VTK_MAJOR_VERSION >= 6)
    vtkAlgorithm *input = this->GetInputAlgorithm();
#else
    vtkImageData *input = this->GetInput();
#endif
    if (input)
      {
      input->UpdateInformation();
#if (VTK_MAJOR_VERSION >= 6)
      int *w_ext = this->GetInputInformation()->Get(
        vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT());
#else
      int *w_ext = input->GetWholeExtent();
#endif
      int xs = 0, ys = 0;

      switch (this->SliceOrientation)
        {
        case vtkImageColorViewer::SLICE_ORIENTATION_XY:
        default:
          xs = w_ext[1] - w_ext[0] + 1;
          ys = w_ext[3] - w_ext[2] + 1;
          break;

        case vtkImageColorViewer::SLICE_ORIENTATION_XZ:
          xs = w_ext[1] - w_ext[0] + 1;
          ys = w_ext[5] - w_ext[4] + 1;
          break;

        case vtkImageColorViewer::SLICE_ORIENTATION_YZ:
          xs = w_ext[3] - w_ext[2] + 1;
          ys = w_ext[5] - w_ext[4] + 1;
          break;
        }

      // if it would be smaller than 150 by 100 then limit to 150 by 100
      if (this->RenderWindow->GetSize()[0] == 0)
        {
        this->RenderWindow->SetSize(
          xs < 150 ? 150 : xs, ys < 100 ? 100 : ys);
        }

      if (this->Renderer)
        {
        this->Renderer->ResetCamera();
        this->Renderer->GetActiveCamera()->SetParallelScale(
          xs < 150 ? 75 : (xs - 1 ) / 2.0);
        }
      this->FirstRender = 0;
      }
    }
  if (this->GetInput())
    {
    this->RenderWindow->Render();
    }
}

//----------------------------------------------------------------------------
const char* vtkImageColorViewer::GetWindowName()
{
  return this->RenderWindow->GetWindowName();
}

//----------------------------------------------------------------------------
void vtkImageColorViewer::SetOffScreenRendering(int i)
{
  this->RenderWindow->SetOffScreenRendering(i);
}

//----------------------------------------------------------------------------
int vtkImageColorViewer::GetOffScreenRendering()
{
  return this->RenderWindow->GetOffScreenRendering();
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 6)
void vtkImageColorViewer::SetInputData(vtkImageData *in)
{
  this->WindowLevel->SetInputData(in);
  this->UpdateDisplayExtent();
}
#else
void vtkImageColorViewer::SetInput(vtkImageData *in)
{
  this->WindowLevel->SetInput(in);
  this->UpdateDisplayExtent();
}
#endif
//----------------------------------------------------------------------------
vtkImageData* vtkImageColorViewer::GetInput()
{
  return vtkImageData::SafeDownCast(this->WindowLevel->GetInput());
}
//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 6)
vtkInformation* vtkImageColorViewer::GetInputInformation()
{
  return this->WindowLevel->GetInputInformation();
}
//----------------------------------------------------------------------------
vtkAlgorithm* vtkImageColorViewer::GetInputAlgorithm()
{
  return this->WindowLevel->GetInputAlgorithm();
}
#endif
//----------------------------------------------------------------------------
void vtkImageColorViewer::SetInputConnection(vtkAlgorithmOutput* input)
{
  this->WindowLevel->SetInputConnection(input);
  this->UpdateDisplayExtent();
}

//----------------------------------------------------------------------------
/*
void vtkImageColorViewer::AddInput(vtkPolyData * input)
{
  vtkRenderWindow *renwin = this->GetRenderWindow ();
  vtkRenderer *Renderer     = vtkRenderer::New();
  vtkPolyDataMapper * mapper = vtkPolyDataMapper::New();
  mapper->SetInput( input );
  vtkActor * actor = vtkActor::New();
  actor->SetMapper( mapper );
  Renderer->AddViewProp(actor);

  renwin->AddRenderer(Renderer);
  Renderer->Delete();
  mapper->Delete();
  actor->Delete();
}
*/

void vtkImageColorViewer::AddInput(vtkImageData * input)
{
  vtkRenderWindow *renwin = this->GetRenderWindow ();
  renwin->SetNumberOfLayers(2);
  vtkRenderer *renderer     = vtkRenderer::New();
  renderer->SetLayer(1);
  OverlayImageActor->SetOpacity(0.5);
  vtkImageMapToWindowLevelColors2 *windowLevel     = vtkImageMapToWindowLevelColors2::New();
#if (VTK_MAJOR_VERSION >= 6)
  windowLevel->SetInputData(input);
  OverlayImageActor->SetInputData(windowLevel->GetOutput());
#else
  windowLevel->SetInput(input);
  OverlayImageActor->SetInput(windowLevel->GetOutput());
#endif
  renderer->AddViewProp(OverlayImageActor);
  OverlayImageActor->SetVisibility(1);

  renwin->AddRenderer(renderer);
  renderer->Delete();
  windowLevel->Delete();
}

void vtkImageColorViewer::AddInputConnection(vtkAlgorithmOutput* input)
{
  vtkRenderWindow *renwin = this->GetRenderWindow ();
  renwin->SetNumberOfLayers(2);
  vtkRenderer *renderer     = vtkRenderer::New();
  renderer->SetLayer(1);
  OverlayImageActor->SetOpacity(0.5);
  vtkImageMapToWindowLevelColors2 *windowLevel     = vtkImageMapToWindowLevelColors2::New();
  windowLevel->SetInputConnection(input);
#if (VTK_MAJOR_VERSION >= 6)
  OverlayImageActor->SetInputData(windowLevel->GetOutput());
#else
  OverlayImageActor->SetInput(windowLevel->GetOutput());
#endif
  renderer->AddViewProp(OverlayImageActor);
  OverlayImageActor->SetVisibility(1);

  renwin->AddRenderer(renderer);
  renderer->Delete();
  windowLevel->Delete();
}

double vtkImageColorViewer::GetOverlayVisibility()
{
  return this->OverlayImageActor->GetVisibility();
}

void vtkImageColorViewer::SetOverlayVisibility(double vis)
{
  this->OverlayImageActor->SetVisibility((int)vis);
}

//----------------------------------------------------------------------------
#ifndef VTK_LEGACY_REMOVE
int vtkImageColorViewer::GetWholeZMin()
{
  VTK_LEGACY_REPLACED_BODY(vtkImageColorViewer::GetWholeZMin, "VTK 5.0",
                           vtkImageColorViewer::GetSliceMin);
  return this->GetSliceMin();
}
int vtkImageColorViewer::GetWholeZMax()
{
  VTK_LEGACY_REPLACED_BODY(vtkImageColorViewer::GetWholeZMax, "VTK 5.0",
                           vtkImageColorViewer::GetSliceMax);
  return this->GetSliceMax();
}
int vtkImageColorViewer::GetZSlice()
{
  VTK_LEGACY_REPLACED_BODY(vtkImageColorViewer::GetZSlice, "VTK 5.0",
                           vtkImageColorViewer::GetSlice);
  return this->GetSlice();
}
void vtkImageColorViewer::SetZSlice(int s)
{
  VTK_LEGACY_REPLACED_BODY(vtkImageColorViewer::SetZSlice, "VTK 5.0",
                           vtkImageColorViewer::SetSlice);
  this->SetSlice(s);
}
#endif

//----------------------------------------------------------------------------
void vtkImageColorViewer::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "RenderWindow:\n";
  this->RenderWindow->PrintSelf(os,indent.GetNextIndent());
  os << indent << "Renderer:\n";
  this->Renderer->PrintSelf(os,indent.GetNextIndent());
  os << indent << "ImageActor:\n";
  this->ImageActor->PrintSelf(os,indent.GetNextIndent());
  os << indent << "WindowLevel:\n" << endl;
  this->WindowLevel->PrintSelf(os,indent.GetNextIndent());
  os << indent << "Slice: " << this->Slice << endl;
  os << indent << "SliceOrientation: " << this->SliceOrientation << endl;
  os << indent << "InteractorStyle: " << endl;
  if (this->InteractorStyle)
    {
    os << "\n";
    this->InteractorStyle->PrintSelf(os,indent.GetNextIndent());
    }
  else
    {
    os << "None";
    }
}
