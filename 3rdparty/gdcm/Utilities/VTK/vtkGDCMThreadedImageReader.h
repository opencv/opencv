/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkGDCMThreadedImageReader - read DICOM files with multiple threads
// .SECTION Description
// vtkGDCMThreadedImageReader is a source object that reads some DICOM files
// This reader is threaded. Meaning that on a multiple core CPU with N cpu, it will
// read approx N times faster than when reading in a single thread.
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
// .SECTION FIXME: you need to call SetFileName when reading a volume file (multiple slices DICOM)
// since SetFileNames expect each single file to be single slice (see parent class)
//
// .SECTION BUG: you should really consider using vtkGDCMThreadedImageReader2 instead !
//
// .SECTION See Also
// vtkMedicalImageReader2 vtkMedicalImageProperties vtkGDCMThreadedImageReader2

#ifndef VTKGDCMTHREADEDIMAGEREADER_H
#define VTKGDCMTHREADEDIMAGEREADER_H

#include "vtkGDCMImageReader.h"

class VTK_EXPORT vtkGDCMThreadedImageReader : public vtkGDCMImageReader
{
public:
  static vtkGDCMThreadedImageReader *New();
  vtkTypeRevisionMacro(vtkGDCMThreadedImageReader,vtkGDCMImageReader);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Explicitely set the Rescale Intercept (0028,1052)
  vtkSetMacro(Shift,double);

  // Description:
  // Explicitely get/set the Rescale Slope (0028,1053)
  vtkSetMacro(Scale,double);

  // Description:
  // Determine whether or not reader should use value from Shift/Scale
  // Default is 1
  vtkSetMacro(UseShiftScale,int);
  vtkGetMacro(UseShiftScale,int);
  vtkBooleanMacro(UseShiftScale,int);

  // Within this class this is allowed to set the Number of Overlays from outside
  //vtkSetMacro(NumberOfOverlays,int);

protected:
  vtkGDCMThreadedImageReader();
  ~vtkGDCMThreadedImageReader();

#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
  int RequestInformation(vtkInformation *request,
                         vtkInformationVector **inputVector,
                         vtkInformationVector *outputVector);
  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector);
#else /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/
  void ExecuteInformation();
  void ExecuteData(vtkDataObject *out);
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/

  void ReadFiles(unsigned int nfiles, const char *filenames[]);
  void RequestDataCompat();

private:
  vtkGDCMThreadedImageReader(const vtkGDCMThreadedImageReader&);  // Not implemented.
  void operator=(const vtkGDCMThreadedImageReader&);  // Not implemented.

  int UseShiftScale;
};

#endif
