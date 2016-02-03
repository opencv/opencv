/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkGDCMThreadedImageReader2 - read DICOM files with multiple threads
// .SECTION Description
// vtkGDCMThreadedImageReader2 is a source object that reads some DICOM files
// This reader is threaded. Meaning that on a multiple core CPU with N cpu, it will
// read approx N times faster than when reading in a single thread assuming the IO is
// not a bottleneck operation.
// If looking for a single threaded class see: vtkGDCMImageReader
//
// .SECTION Warning: Advanced users only. Do not use this class in the general case,
// you have to understand how physicaly medium works first (sequential reading for
// instance) before playing with this class
//
// .SECTION Implementation note: when FileLowerLeft is set to on the image is not flipped
// upside down as VTK would expect, use this option only if you know what you are doing
//
// .SECTION FIXME: need to implement the other mode where FileLowerLeft is set to OFF
//
// .SECTION FIXME: need to implement reading of series of 3D files
//
// .SECTION Implementation note: this class is meant to superseed vtkGDCMThreadedImageReader
// because it had support for ProgressEvent support even from python layer. There is a
// subtle trick down in the threading mechanism in VTK were the main thread (talking to the
// python interpreter) is also part of the execution process (and the N-1 other thread
// are just there to execute the remaining of ThreadedRequestData), this separation into
// two types of thread is necessary to acheive a working implementation of UpdateProgress

// .SECTION See Also
// vtkMedicalImageReader2 vtkMedicalImageProperties vtkGDCMImageReader

#ifndef VTKGDCMTHREADEDIMAGEREADER2_H
#define VTKGDCMTHREADEDIMAGEREADER2_H

#include "vtkThreadedImageAlgorithm.h"

class vtkStringArray;
class VTK_EXPORT vtkGDCMThreadedImageReader2 : public vtkThreadedImageAlgorithm
{
public:
  static vtkGDCMThreadedImageReader2 *New();
  vtkTypeRevisionMacro(vtkGDCMThreadedImageReader2,vtkThreadedImageAlgorithm);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  vtkGetMacro(FileLowerLeft,int);
  vtkSetMacro(FileLowerLeft,int);
  vtkBooleanMacro(FileLowerLeft,int);

  vtkGetMacro(NumberOfOverlays,int);

  vtkSetMacro(DataScalarType,int);
  vtkGetMacro(DataScalarType,int);

  vtkSetMacro(NumberOfScalarComponents,int);
  vtkGetMacro(NumberOfScalarComponents,int);

  vtkGetMacro(LoadOverlays,int);
  vtkSetMacro(LoadOverlays,int);
  vtkBooleanMacro(LoadOverlays,int);

  vtkSetVector6Macro(DataExtent,int);
  vtkGetVector6Macro(DataExtent,int);

  vtkSetVector3Macro(DataOrigin,double);
  vtkGetVector3Macro(DataOrigin,double);

  vtkSetVector3Macro(DataSpacing,double);
  vtkGetVector3Macro(DataSpacing,double);

  //vtkGetStringMacro(FileName);
  //vtkSetStringMacro(FileName);
  virtual const char *GetFileName(int i = 0);
  virtual void SetFileName(const char *filename);

  virtual void SetFileNames(vtkStringArray*);
  vtkGetObjectMacro(FileNames, vtkStringArray);

  int SplitExtent(int splitExt[6], int startExt[6],
                  int num, int total);

  // Description:
  // Explicitely set the Rescale Intercept (0028,1052)
  vtkSetMacro(Shift,double);
  vtkGetMacro(Shift,double);

  // Description:
  // Explicitely get/set the Rescale Slope (0028,1053)
  vtkSetMacro(Scale,double);
  vtkGetMacro(Scale,double);

  // Description:
  // Determine whether or not reader should use value from Shift/Scale
  // Default is 1
  vtkSetMacro(UseShiftScale,int);
  vtkGetMacro(UseShiftScale,int);
  vtkBooleanMacro(UseShiftScale,int);

protected:
  vtkGDCMThreadedImageReader2();
  ~vtkGDCMThreadedImageReader2();

  int RequestInformation(vtkInformation *request,
                         vtkInformationVector **inputVector,
                         vtkInformationVector *outputVector);

protected:
  void ThreadedRequestData (
    vtkInformation * request,
    vtkInformationVector** inputVector,
    vtkInformationVector * outputVector,
    vtkImageData ***inData,
    vtkImageData **outData,
    int outExt[6], int id);

private:
  int FileLowerLeft;
  char *FileName;
  vtkStringArray *FileNames;
  int LoadIconImage;
  int DataExtent[6];
  int LoadOverlays;
  int NumberOfOverlays;
  int DataScalarType;

  int NumberOfScalarComponents;
  double DataSpacing[3];
  double DataOrigin[3];
  int IconImageDataExtent[6];

  double Shift;
  double Scale;
  int UseShiftScale;

private:
  vtkGDCMThreadedImageReader2(const vtkGDCMThreadedImageReader2&);  // Not implemented.
  void operator=(const vtkGDCMThreadedImageReader2&);  // Not implemented.
};

#endif
