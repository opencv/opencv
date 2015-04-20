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
  Module:    $RCSfile: vtkImageRGBToYBR.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageRGBToYBR - Converts YBR components to RGB.
// .SECTION Description
// For each pixel with hue, saturation and value components this filter
// outputs the color coded as red, green, blue.  Output type must be the same
// as input type.

// .SECTION See Also
// vtkImageRGBToHSV

#ifndef VTKIMAGERGBTOYBR_H
#define VTKIMAGERGBTOYBR_H

#include "vtkThreadedImageAlgorithm.h"

class VTK_EXPORT vtkImageRGBToYBR : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageRGBToYBR *New();
  vtkTypeRevisionMacro(vtkImageRGBToYBR,vtkThreadedImageAlgorithm);

  void PrintSelf(ostream& os, vtkIndent indent);

protected:
  vtkImageRGBToYBR();
  ~vtkImageRGBToYBR() {};

  void ThreadedExecute (vtkImageData *inData, vtkImageData *outData,
                       int ext[6], int id);
private:
  vtkImageRGBToYBR(const vtkImageRGBToYBR&);  // Not implemented.
  void operator=(const vtkImageRGBToYBR&);  // Not implemented.
};

#endif
