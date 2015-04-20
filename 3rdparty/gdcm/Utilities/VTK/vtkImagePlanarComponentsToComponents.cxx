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
  Module:    $RCSfile: vtkImagePlanarComponentsToComponents.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImagePlanarComponentsToComponents.h"

#include "vtkImageData.h"
#include "vtkImageProgressIterator.h"
#include "vtkMath.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "gdcmImageChangePlanarConfiguration.h"

#include <assert.h>

vtkCxxRevisionMacro(vtkImagePlanarComponentsToComponents, "$Revision: 1.31 $")
vtkStandardNewMacro(vtkImagePlanarComponentsToComponents)

//----------------------------------------------------------------------------
vtkImagePlanarComponentsToComponents::vtkImagePlanarComponentsToComponents()
{
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}


//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
template <class T>
void vtkImagePlanarComponentsToComponentsExecute(vtkImagePlanarComponentsToComponents *self,
                             vtkImageData *inData,
                             vtkImageData *outData,
                             int outExt[6], int id, T *)
{
  int maxX, maxY, maxZ;
  // find the region to loop over
  maxX = outExt[1] - outExt[0];
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];
  (void)self;
  (void)id;


  //target = static_cast<unsigned long>((maxZ+1)*(maxY+1)/50.0);
  //target++;

  const T *inPtr = (T*)inData->GetScalarPointer(outExt[0],outExt[2],outExt[4]);
  T *outPtr = static_cast<T*>(outData->GetScalarPointer(outExt[0],outExt[2],outExt[4]));

  // Loop through ouput pixels

  size_t framesize = (maxX+1) * (maxY+1) * 3;
  for(int z = 0; z <= maxZ; ++z)
    {
    const T *frame = inPtr + z * framesize;
    size_t size = framesize / 3;
    const T *r = frame + 0;
    const T *g = frame + size;
    const T *b = frame + size + size;

    T *framecopy = outPtr + z * framesize;
    gdcm::ImageChangePlanarConfiguration::RGBPlanesToRGBPixels(framecopy, r, g, b, size);
    }


}

//----------------------------------------------------------------------------
int vtkImagePlanarComponentsToComponents::RequestData(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  int idxX, idxY, idxZ;
  vtkIdType outIncX, outIncY, outIncZ;
  int *outExt;
  double sum;
  double yContrib, zContrib;
  double temp, temp2;
  unsigned long count = 0;
  unsigned long target;
  //
  // get the input
  vtkInformation* in1Info = inputVector[0]->GetInformationObject(0);
  vtkImageData *inData = vtkImageData::SafeDownCast(
    in1Info->Get(vtkDataObject::DATA_OBJECT()));

  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkImageData *output = vtkImageData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData *data = this->AllocateOutputData(output);

  if (data->GetScalarType() != VTK_UNSIGNED_CHAR
    && data->GetScalarType() != VTK_UNSIGNED_SHORT )
    {
    vtkErrorMacro("Execute: This source only deal with uchar/ushort");
    }

  outExt = data->GetExtent();

  // Get increments to march through data
  data->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);

  switch (inData->GetScalarType())
    {
    vtkTemplateMacro(
      vtkImagePlanarComponentsToComponentsExecute(this, inData,
                              data, outExt, 0, static_cast<VTK_TT *>(0)));
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return 0;
    }

  return 1;
}

//----------------------------------------------------------------------------
void vtkImagePlanarComponentsToComponents::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
