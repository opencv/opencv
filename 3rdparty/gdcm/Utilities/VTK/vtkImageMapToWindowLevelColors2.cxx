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
  Module:    $RCSfile: vtkImageMapToWindowLevelColors2.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageMapToWindowLevelColors2.h"

#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkScalarsToColors.h"
#include "vtkPointData.h"

vtkCxxRevisionMacro(vtkImageMapToWindowLevelColors2, "$Revision: 1.3 $")
vtkStandardNewMacro(vtkImageMapToWindowLevelColors2)

// Constructor sets default values
vtkImageMapToWindowLevelColors2::vtkImageMapToWindowLevelColors2()
{
  this->Window = 255;
  this->Level  = 127.5;
}

vtkImageMapToWindowLevelColors2::~vtkImageMapToWindowLevelColors2()
{
}

//----------------------------------------------------------------------------
// This method checks to see if we can simply reference the input data
int vtkImageMapToWindowLevelColors2::RequestData(
  vtkInformation *request,
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);

  vtkImageData *outData = vtkImageData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData *inData = vtkImageData::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));

  // If LookupTable is null and window / level produces no change,
  // then just pass the data
  if (this->LookupTable == NULL &&
      (inData->GetScalarType() == VTK_UNSIGNED_CHAR &&
       this->Window == 255 && this->Level == 127.5))
    {
    vtkDebugMacro("ExecuteData: LookupTable not set, "\
                  "Window / Level at default, "\
                  "passing input to output.");

    outData->SetExtent(inData->GetExtent());
    outData->GetPointData()->PassData(inData->GetPointData());
    this->DataWasPassed = 1;
    }
  else
    // normal behaviour - skip up a level since we don't want to
    // call the superclasses ExecuteData - it would pass the data if there
    // is no lookup table even if there is a window / level - wrong
    // behavior.
    {
    if (this->DataWasPassed)
      {
      outData->GetPointData()->SetScalars(NULL);
      this->DataWasPassed = 0;
      }

    return this->vtkThreadedImageAlgorithm::RequestData(request, inputVector,
                                                        outputVector);
    }

  return 1;
}

//----------------------------------------------------------------------------
int vtkImageMapToWindowLevelColors2::RequestInformation (
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  vtkInformation *inScalarInfo =
    vtkDataObject::GetActiveFieldInformation(inInfo,
    vtkDataObject::FIELD_ASSOCIATION_POINTS, vtkDataSetAttributes::SCALARS);
  if (!inScalarInfo)
    {
    vtkErrorMacro("Missing scalar field on input information!");
    return 0;
    }

  // If LookupTable is null and window / level produces no change,
  // then the data will be passed
  if ( this->LookupTable == NULL &&
       (inScalarInfo->Get(vtkDataObject::FIELD_ARRAY_TYPE()) ==
        VTK_UNSIGNED_CHAR &&
        this->Window == 255 && this->Level == 127.5) )
    {
    if (inScalarInfo->Get(vtkDataObject::FIELD_ARRAY_TYPE()) !=
        VTK_UNSIGNED_CHAR)
      {
      vtkErrorMacro("ExecuteInformation: No LookupTable was set and input data is not VTK_UNSIGNED_CHAR!");
      }
    else
      {
      // no lookup table, pass the input if it was UNSIGNED_CHAR
      vtkDataObject::SetPointDataActiveScalarInfo
        (outInfo, VTK_UNSIGNED_CHAR,
         inScalarInfo->Get(vtkDataObject::FIELD_NUMBER_OF_COMPONENTS()));
      }
    }
  else  // the lookup table was set or window / level produces a change
    {
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
        vtkErrorMacro("ExecuteInformation: Unrecognized color format.");
        break;
      }
    vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_UNSIGNED_CHAR, numComponents);
    }

  return 1;
}

/*
 * This templated routine calculates effective lower and upper limits
 * for a window of values of type T, lower and upper.
 */
template <class T>
void vtkImageMapToWindowLevelClamps ( vtkImageData *data, double w,
                                      double l, T& lower, T& upper,
                                      unsigned char &lower_val,
                                      unsigned char &upper_val)
{
  double f_lower, f_upper, f_lower_val, f_upper_val;
  double adjustedLower, adjustedUpper;
  double range[2];

  data->GetPointData()->GetScalars()->GetDataTypeRange( range );

  f_lower = l - fabs(w) / 2.0;
  f_upper = f_lower + fabs(w);

  // Set the correct lower value
  if ( f_lower <= range[1])
    {
    if (f_lower >= range[0])
      {
      lower = (T) f_lower;
      adjustedLower = f_lower;
      }
    else
      {
      lower = (T) range[0];
      adjustedLower = range[0];
      }
    }
  else
    {
    lower = (T) range[1];
    adjustedLower = range[1];
    }

  // Set the correct upper value
  if ( f_upper >= range[0])
    {
    if (f_upper <= range[1])
      {
      upper = (T) f_upper;
      adjustedUpper = f_upper;
      }
    else
      {
      upper = (T) range[1];
      adjustedUpper = range[1];
      }
    }
  else
    {
    upper = (T) range [0];
    adjustedUpper = range [0];
    }

  // now compute the lower and upper values
  if (w >= 0)
    {
    f_lower_val = 255.0*(adjustedLower - f_lower)/w;
    f_upper_val = 255.0*(adjustedUpper - f_lower)/w;
    }
  else
    {
    f_lower_val = 255.0 + 255.0*(adjustedLower - f_lower)/w;
    f_upper_val = 255.0 + 255.0*(adjustedUpper - f_lower)/w;
    }

  if (f_upper_val > 255)
    {
    upper_val = 255;
    }
  else if (f_upper_val < 0)
    {
    upper_val = 0;
    }
  else
    {
    upper_val = (unsigned char)(f_upper_val);
    }

  if (f_lower_val > 255)
    {
    lower_val = 255;
    }
  else if (f_lower_val < 0)
    {
    lower_val = 0;
    }
  else
    {
    lower_val = (unsigned char)(f_lower_val);
    }
}

//----------------------------------------------------------------------------
// Small helper to do the clamp:
template <typename T>
void vtkClampHelper1(T* iptr, unsigned char *optr,
  T lower, T upper,
  unsigned char lower_val, unsigned char upper_val,
  double shift, double scale)
{
  unsigned short ushort_val;
  if (*iptr <= lower)
    {
    ushort_val = lower_val;
    }
  else if (*iptr >= upper)
    {
    ushort_val = upper_val;
    }
  else
    {
    ushort_val = (unsigned char) (((double)*iptr + shift)*scale);
    }
  *optr = (unsigned char)((*optr * ushort_val) >> 8);
}

//----------------------------------------------------------------------------
template <typename T>
void vtkClampHelper2(T* iptr, unsigned char *optr,
  T lower, T upper,
  unsigned char lower_val, unsigned char upper_val,
  double shift, double scale)
{
  unsigned char result_val;
  if (*iptr <= lower)
    {
    result_val = lower_val;
    }
  else if (*iptr >= upper)
    {
    result_val = upper_val;
    }
  else
    {
    result_val = (unsigned char) (((double)*iptr + shift)*scale);
    }
  *optr = result_val;
}

//----------------------------------------------------------------------------
// This non-templated function executes the filter for any type of data.
template <class T>
void vtkImageMapToWindowLevelColors2Execute(
  vtkImageMapToWindowLevelColors2 *self,
  vtkImageData *inData, T *inPtr,
  vtkImageData *outData,
  unsigned char *outPtr,
  int outExt[6], int id)
{
  int idxX, idxY, idxZ;
  int extX, extY, extZ;
  vtkIdType inIncX, inIncY, inIncZ;
  vtkIdType outIncX, outIncY, outIncZ;
  unsigned long count = 0;
  unsigned long target;
  int dataType = inData->GetScalarType();
  int numberOfComponents,numberOfOutputComponents,outputFormat;
  int rowLength;
  vtkScalarsToColors *lookupTable = self->GetLookupTable();
  unsigned char *outPtr1;
  T *inPtr1;
  unsigned char *optr;
  T    *iptr;
  double shift =  self->GetWindow() / 2.0 - self->GetLevel();
  double scale = 255.0 / self->GetWindow();

  T   lower, upper;
  unsigned char lower_val, upper_val;
  vtkImageMapToWindowLevelClamps( inData, self->GetWindow(),
                                  self->GetLevel(),
                                  lower, upper, lower_val, upper_val );

  // find the region to loop over
  extX = outExt[1] - outExt[0] + 1;
  extY = outExt[3] - outExt[2] + 1;
  extZ = outExt[5] - outExt[4] + 1;

  target = (unsigned long)(extZ*extY/50.0);
  target++;

  // Get increments to march through data
  inData->GetContinuousIncrements(outExt, inIncX, inIncY, inIncZ);

  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);
  numberOfComponents = inData->GetNumberOfScalarComponents();
  numberOfOutputComponents = outData->GetNumberOfScalarComponents();
  outputFormat = self->GetOutputFormat();

  rowLength = extX*numberOfComponents;

  // Loop through output pixels
  outPtr1 = outPtr;
  inPtr1 = inPtr;
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

      iptr = inPtr1;
      optr = outPtr1;

      if ( lookupTable )
        {
        lookupTable->MapScalarsThroughTable2(inPtr1,(unsigned char *)outPtr1,
                                             dataType,extX,numberOfComponents,
                                             outputFormat);

        for (idxX = 0; idxX < extX; idxX++)
          {
          vtkClampHelper1<T>(iptr,optr,lower,upper,lower_val,upper_val,shift,scale);
          switch (outputFormat)
            {
            case VTK_RGBA:
              vtkClampHelper1<T>(iptr+(1%numberOfComponents),optr+1,lower,upper,lower_val,upper_val,shift,scale);
              vtkClampHelper1<T>(iptr+(2%numberOfComponents),optr+2,lower,upper,lower_val,upper_val,shift,scale);
              *(optr+3) = 255;
              break;
            case VTK_RGB:
              vtkClampHelper1<T>(iptr+(1%numberOfComponents),optr+1,lower,upper,lower_val,upper_val,shift,scale);
              vtkClampHelper1<T>(iptr+(2%numberOfComponents),optr+2,lower,upper,lower_val,upper_val,shift,scale);
              break;
            case VTK_LUMINANCE_ALPHA:
              *(optr+1) = 255;
              break;
            }
          iptr += numberOfComponents;
          optr += numberOfOutputComponents;
          }
        }
      else
        {
        for (idxX = 0; idxX < extX; idxX++)
          {
          // We want to shift to the right position depending on the numberOfComponents from input
          // if grayscale we should stay at the same position, otherwise need to shift to r,g,b
          // (0%numberOfComponents) == 0 ...
          // (1%numberOfComponents) == 0 or 1
          vtkClampHelper2<T>(iptr,optr,lower,upper,lower_val,upper_val,shift,scale);
          switch (outputFormat)
            {
            case VTK_RGBA:
              vtkClampHelper2<T>(iptr+(1%numberOfComponents),optr+1,lower,upper,lower_val,upper_val,shift,scale);
              vtkClampHelper2<T>(iptr+(2%numberOfComponents),optr+2,lower,upper,lower_val,upper_val,shift,scale);
              *(optr+3) = 255;
              break;
            case VTK_RGB:
              vtkClampHelper2<T>(iptr+(1%numberOfComponents),optr+1,lower,upper,lower_val,upper_val,shift,scale);
              vtkClampHelper2<T>(iptr+(2%numberOfComponents),optr+2,lower,upper,lower_val,upper_val,shift,scale);
              break;
            case VTK_LUMINANCE_ALPHA:
              *(optr+1) = 255;
              break;
            }
          iptr += numberOfComponents;
          optr += numberOfOutputComponents;
          }
        }
      outPtr1 += outIncY + extX*numberOfOutputComponents;
      inPtr1 += inIncY + rowLength;
      }
    outPtr1 += outIncZ;
    inPtr1 += inIncZ;
    }
}

//----------------------------------------------------------------------------
// This method is passed a input and output data, and executes the filter
// algorithm to fill the output from the input.

void vtkImageMapToWindowLevelColors2::ThreadedRequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *vtkNotUsed(outputVector),
  vtkImageData ***inData,
  vtkImageData **outData,
  int outExt[6], int id)
{
  void *inPtr = inData[0][0]->GetScalarPointerForExtent(outExt);
  void *outPtr = outData[0]->GetScalarPointerForExtent(outExt);

  switch (inData[0][0]->GetScalarType())
    {
    vtkTemplateMacro(
      vtkImageMapToWindowLevelColors2Execute( this,
                                             inData[0][0],
                                             (VTK_TT *)(inPtr),
                                             outData[0],
                                             (unsigned char *)(outPtr),
                                             outExt,
                                             id));
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}

//----------------------------------------------------------------------------
void vtkImageMapToWindowLevelColors2::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Window: " << this->Window << endl;
  os << indent << "Level: " << this->Level << endl;
}
