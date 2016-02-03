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
  Module:    $RCSfile: vtkImageMapToColors16.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageMapToColors16.h"

#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkScalarsToColors.h"
#include "vtkPointData.h"

vtkCxxRevisionMacro(vtkImageMapToColors16, "$Revision: 1.30 $")
vtkStandardNewMacro(vtkImageMapToColors16)
vtkCxxSetObjectMacro(vtkImageMapToColors16,LookupTable,vtkScalarsToColors)

//----------------------------------------------------------------------------
// Constructor sets default values
vtkImageMapToColors16::vtkImageMapToColors16()
{
  this->OutputFormat = 4;
  this->ActiveComponent = 0;
  this->PassAlphaToOutput = 0;
  this->LookupTable = NULL;
  this->DataWasPassed = 0;
}

//----------------------------------------------------------------------------
vtkImageMapToColors16::~vtkImageMapToColors16()
{
  if (this->LookupTable != NULL)
    {
    this->LookupTable->UnRegister(this);
    }
}

//----------------------------------------------------------------------------
unsigned long vtkImageMapToColors16::GetMTime()
{
  unsigned long t1, t2;

  t1 = this->Superclass::GetMTime();
  if (this->LookupTable)
    {
    t2 = this->LookupTable->GetMTime();
    if (t2 > t1)
      {
      t1 = t2;
      }
    }
  return t1;
}

//----------------------------------------------------------------------------
// This method checks to see if we can simply reference the input data
int vtkImageMapToColors16::RequestData(vtkInformation *request,
                                      vtkInformationVector **inputVector,
                                      vtkInformationVector *outputVector)
{
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  vtkImageData *outData = vtkImageData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData *inData = vtkImageData::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));

  // If LookupTable is null, just pass the data
  if (this->LookupTable == NULL)
    {
    vtkDebugMacro("RequestData: LookupTable not set, "\
                  "passing input to output.");

    outData->SetExtent(inData->GetExtent());
    outData->GetPointData()->PassData(inData->GetPointData());
    this->DataWasPassed = 1;
    }
  else // normal behaviour
    {
    this->LookupTable->Build(); //make sure table is built

    if (this->DataWasPassed)
      {
      outData->GetPointData()->SetScalars(NULL);
      this->DataWasPassed = 0;
      }

    return this->Superclass::RequestData(request, inputVector, outputVector);
    }

  return 1;
}

//----------------------------------------------------------------------------
int vtkImageMapToColors16::RequestInformation (
  vtkInformation * vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);

  int numComponents = 4;

  switch (this->OutputFormat)
    {
    case VTK_RGBA:
      numComponents = 4;
      break;
    case VTK_RGB:
      numComponents = 3;
      break;
    case VTK_LUMINANCE_ALPHA:
      numComponents = 2;
      break;
    case VTK_LUMINANCE:
      numComponents = 1;
      break;
    default:
      vtkErrorMacro("RequestInformation: Unrecognized color format.");
      break;
    }

  if (this->LookupTable == NULL)
    {
    vtkInformation *scalarInfo = vtkDataObject::GetActiveFieldInformation(inInfo,
      vtkDataObject::FIELD_ASSOCIATION_POINTS, vtkDataSetAttributes::SCALARS);
    if ( scalarInfo->Get(vtkDataObject::FIELD_ARRAY_TYPE()) != VTK_UNSIGNED_CHAR )
      {
      vtkErrorMacro(
        "RequestInformation: No LookupTable was set but input data is not "
        "VTK_UNSIGNED_CHAR, therefore input can't be passed through!");
      return 1;
      }
    else if ( numComponents !=
      scalarInfo->Get(vtkDataObject::FIELD_NUMBER_OF_COMPONENTS()) )
      {
      vtkErrorMacro(
        "RequestInformation: No LookupTable was set but number of components "
        "in input doesn't match OutputFormat, therefore input can't be passed"
        " through!");
      return 1;
      }
    }

//  vtkDataObject::
//    SetPointDataActiveScalarInfo(outInfo, VTK_UNSIGNED_CHAR, numComponents);
  vtkDataObject::
    SetPointDataActiveScalarInfo(outInfo, VTK_UNSIGNED_SHORT, numComponents);
  return 1;
}

//----------------------------------------------------------------------------
// This non-templated function executes the filter for any type of data.

void vtkImageMapToColors16Execute(vtkImageMapToColors16 *self,
                                vtkImageData *inData, void *inPtr,
                                vtkImageData *outData,
                                unsigned short *outPtr,
                                int outExt[6], int id)
{
  int idxY, idxZ;
  int extX, extY, extZ;
  vtkIdType inIncX, inIncY, inIncZ;
  vtkIdType outIncX, outIncY, outIncZ;
  unsigned long count = 0;
  unsigned long target;
  int dataType = inData->GetScalarType();
  int scalarSize = inData->GetScalarSize();
  int numberOfComponents,numberOfOutputComponents,outputFormat;
  int rowLength;
  vtkScalarsToColors *lookupTable = self->GetLookupTable();
  unsigned short *outPtr1;
  void *inPtr1;

  // find the region to loop over
  extX = outExt[1] - outExt[0] + 1;
  extY = outExt[3] - outExt[2] + 1;
  extZ = outExt[5] - outExt[4] + 1;

  target = static_cast<unsigned long>(extZ*extY/50.0);
  target++;

  // Get increments to march through data
  inData->GetContinuousIncrements(outExt, inIncX, inIncY, inIncZ);
  // because we are using void * and char * we must take care
  // of the scalar size in the increments
  inIncY *= scalarSize;
  inIncZ *= scalarSize;
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);
  numberOfComponents = inData->GetNumberOfScalarComponents();
  numberOfOutputComponents = outData->GetNumberOfScalarComponents();
  outputFormat = self->GetOutputFormat();
  rowLength = extX*scalarSize*numberOfComponents;

  // Loop through output pixels
  outPtr1 = outPtr;
  inPtr1 = static_cast<void *>(
    static_cast<char *>(inPtr) + self->GetActiveComponent()*scalarSize);
  for (idxZ = 0; idxZ < extZ; idxZ++)
    {
    for (idxY = 0; !self->AbortExecute && idxY < extY; idxY++)
      {
      if (!id)
        {
        if (!(count%target))
          {
          self->UpdateProgress((double)count/(50.0*(double)target));
          }
        count++;
        }
      lookupTable->MapScalarsThroughTable2(inPtr1,(unsigned char*)outPtr1,
                                           dataType,extX,numberOfComponents,
                                           outputFormat);
      //if (self->GetPassAlphaToOutput() &&
      //    dataType == VTK_UNSIGNED_CHAR && numberOfComponents > 1 &&
      //    (outputFormat == VTK_RGBA || outputFormat == VTK_LUMINANCE_ALPHA))
      //  {
      //  unsigned char *outPtr2 = outPtr1 + numberOfOutputComponents - 1;
      //  unsigned char *inPtr2 = static_cast<unsigned char *>(inPtr1)
      //    - self->GetActiveComponent()*scalarSize + numberOfComponents - 1;
      //  for (int i = 0; i < extX; i++)
      //    {
      //    *outPtr2 = (*outPtr2 * *inPtr2)/255;
      //    outPtr2 += numberOfOutputComponents;
      //    inPtr2 += numberOfComponents;
      //    }
      //  }
      outPtr1 += outIncY + extX*numberOfOutputComponents;
      inPtr1 = static_cast<void *>(
        static_cast<char *>(inPtr1) + inIncY + rowLength);
      }
    outPtr1 += outIncZ;
    inPtr1 = static_cast<void *>(static_cast<char *>(inPtr1) + inIncZ);
    }
}

//----------------------------------------------------------------------------
// This method is passed a input and output data, and executes the filter
// algorithm to fill the output from the input.

void vtkImageMapToColors16::ThreadedRequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *vtkNotUsed(outputVector),
  vtkImageData ***inData,
  vtkImageData **outData,
  int outExt[6], int id)
{
  void *inPtr = inData[0][0]->GetScalarPointerForExtent(outExt);
  void *outPtr = outData[0]->GetScalarPointerForExtent(outExt);

  vtkImageMapToColors16Execute(this, inData[0][0], inPtr,
                             outData[0], static_cast<unsigned short *>(outPtr),
                             outExt, id);
}

//----------------------------------------------------------------------------
void vtkImageMapToColors16::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "OutputFormat: " <<
    (this->OutputFormat == VTK_RGBA ? "RGBA" :
     (this->OutputFormat == VTK_RGB ? "RGB" :
      (this->OutputFormat == VTK_LUMINANCE_ALPHA ? "LuminanceAlpha" :
       (this->OutputFormat == VTK_LUMINANCE ? "Luminance" : "Unknown"))))
    << "\n";
  os << indent << "ActiveComponent: " << this->ActiveComponent << "\n";
  os << indent << "PassAlphaToOutput: " << this->PassAlphaToOutput << "\n";
  os << indent << "LookupTable: ";
  if (this->LookupTable)
    {
    this->LookupTable->PrintSelf(os << endl,indent.GetNextIndent());
    }
  else
    {
    os << "(none)\n";
    }
}
