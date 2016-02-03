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
  Module:    $RCSfile: vtkImagePlanarComponentsToComponents.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImagePlanarComponentsToComponents - Converts planar comp to pixel comp
// .SECTION Description

// .SECTION See Also
// TODO: Can I make this filter threaded ?
// TODO: How do I handle the VTK-flipping (FileLowerLeft)?

#ifndef VTKIMAGEPLANARCOMPONENTSTOCOMPONENTS_H
#define VTKIMAGEPLANARCOMPONENTSTOCOMPONENTS_H

#include "vtkImageAlgorithm.h"

// everything is now handled within the vtkGDCMImageReader as Planar Configuration can not
// be externalized (conflict with file lower left)

#error do not use this class

//class VTK_EXPORT vtkImagePlanarComponentsToComponents : public vtkThreadedImageAlgorithm
class VTK_EXPORT vtkImagePlanarComponentsToComponents : public vtkImageAlgorithm
{
public:
  static vtkImagePlanarComponentsToComponents *New();
  //vtkTypeRevisionMacro(vtkImagePlanarComponentsToComponents,vtkThreadedImageAlgorithm);
  vtkTypeRevisionMacro(vtkImagePlanarComponentsToComponents,vtkImageAlgorithm);

  void PrintSelf(ostream& os, vtkIndent indent);

protected:
  vtkImagePlanarComponentsToComponents();
  ~vtkImagePlanarComponentsToComponents() {};

//  void ThreadedExecute (vtkImageData *inData, vtkImageData *outData,
//                       int ext[6], int id);
//  virtual int RequestInformation (vtkInformation *, vtkInformationVector**, vtkInformationVector *);
  virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
  vtkImagePlanarComponentsToComponents(const vtkImagePlanarComponentsToComponents&);  // Not implemented.
  void operator=(const vtkImagePlanarComponentsToComponents&);  // Not implemented.
};

#endif
