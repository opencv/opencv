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
  Module:    $RCSfile: vtkLookupTable16.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkLookupTable16.h"
#include "vtkObjectFactory.h"

#include <cassert>

vtkCxxRevisionMacro(vtkLookupTable16, "$Revision: 1.107 $")
vtkStandardNewMacro(vtkLookupTable16)

vtkLookupTable16::vtkLookupTable16(int sze, int ext)
  : vtkLookupTable(sze, ext)
{
  this->Table16 = vtkUnsignedShortArray::New();
  this->Table16->Register(this);
  this->Table16->Delete();
  this->Table16->SetNumberOfComponents(4);
  this->Table16->Allocate(4*sze,4*ext);
}

//----------------------------------------------------------------------------
vtkLookupTable16::~vtkLookupTable16()
{
  this->Table16->UnRegister(this);
  this->Table16 = NULL;
}

void vtkLookupTable16::Build()
{
}

void vtkLookupTable16::SetNumberOfTableValues(vtkIdType number)
{
  if (this->NumberOfColors == number)
    {
    return;
    }
  this->Modified();
  this->NumberOfColors = number;
  this->Table16->SetNumberOfTuples(number);
}

//----------------------------------------------------------------------------
// Apply shift/scale to the scalar value v and do table lookup.
inline unsigned short *vtkLinearLookup16(double v,
                                      unsigned short *table,
                                      double maxIndex,
                                      double shift, double scale)
{
  double findx = (v + shift)*scale;
  if (findx < 0)
    {
    findx = 0;
    }
  if (findx > maxIndex)
    {
    findx = maxIndex;
    }
  return &table[4*static_cast<int>(findx)];
  /* round
  return &table[4*(int)(findx + 0.5f)];
  */
}

void vtkLookupTableLogRange16(double [2], double [2])
{
  assert(0);
}

inline double vtkApplyLogScale16(double , double [2],
                               double [2])
{
  assert(0);
  return 0;
}

template<class T>
void vtkLookupTable16MapData(vtkLookupTable16 *self, T *input,
                           unsigned short *output, int length,
                           int inIncr, int outFormat)
{
  int i = length;
  double *range = self->GetTableRange();
  double maxIndex = (double)self->GetNumberOfColors() - 1;
  double shift, scale;
  unsigned short *table = self->GetPointer(0);
  unsigned short *cptr;
  double alpha;

  if ( (alpha=self->GetAlpha()) >= 1.0 ) //no blending required
    {
    if (self->GetScale() == VTK_SCALE_LOG10)
      {
      double val;
      double logRange[2];
      vtkLookupTableLogRange16(range, logRange);
      shift = -logRange[0];
      if (logRange[1] <= logRange[0])
        {
        scale = VTK_DOUBLE_MAX;
        }
      else
        {
        /* while this looks like the wrong scale, it is the correct scale
         * taking into account the truncation to int that happens below. */
        scale = (maxIndex + 1)/(logRange[1] - logRange[0]);
        }
      if (outFormat == VTK_RGBA)
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_RGB)
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_LUMINANCE_ALPHA)
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          *output++ = cptr[3];
          input += inIncr;
          }
        }
      else // outFormat == VTK_LUMINANCE
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          input += inIncr;
          }
        }
      }//if log scale

    else //not log scale
      {
      shift = -range[0];
      if (range[1] <= range[0])
        {
        scale = VTK_DOUBLE_MAX;
        }
      else
        {
        /* while this looks like the wrong scale, it is the correct scale
         * taking into account the truncation to int that happens below. */
        scale = (maxIndex + 1)/(range[1] - range[0]);
        }

      if (outFormat == VTK_RGBA)
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_RGB)
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_LUMINANCE_ALPHA)
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          *output++ = cptr[3];
          input += inIncr;
          }
        }
      else // outFormat == VTK_LUMINANCE
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          input += inIncr;
          }
        }
      }//if not log lookup
    }//if blending not needed

  else //blend with the specified alpha
    {
    if (self->GetScale() == VTK_SCALE_LOG10)
      {
      double val;
      double logRange[2];
      vtkLookupTableLogRange16(range, logRange);
      shift = -logRange[0];
      if (logRange[1] <= logRange[0])
        {
        scale = VTK_DOUBLE_MAX;
        }
      else
        {
        /* while this looks like the wrong scale, it is the correct scale
         * taking into account the truncation to int that happens below. */
        scale = (maxIndex + 1)/(logRange[1] - logRange[0]);
        }
      if (outFormat == VTK_RGBA)
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = static_cast<unsigned short>((*cptr)*alpha); cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_RGB)
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_LUMINANCE_ALPHA)
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          *output++ = static_cast<unsigned short>(alpha*cptr[3]);
          input += inIncr;
          }
        }
      else // outFormat == VTK_LUMINANCE
        {
        while (--i >= 0)
          {
          val = vtkApplyLogScale16((double)*input, range, logRange);
          cptr = vtkLinearLookup16(val, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          input += inIncr;
          }
        }
      }//log scale with blending

    else //no log scale with blending
      {
      shift = -range[0];
      if (range[1] <= range[0])
        {
        scale = VTK_DOUBLE_MAX;
        }
      else
        {
        /* while this looks like the wrong scale, it is the correct scale
         * taking into account the truncation to int that happens below. */
        scale = (maxIndex + 1)/(range[1] - range[0]);
        }

      if (outFormat == VTK_RGBA)
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = static_cast<unsigned short>((*cptr)*alpha); cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_RGB)
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = *cptr++;
          *output++ = *cptr++;
          *output++ = *cptr++;
          input += inIncr;
          }
        }
      else if (outFormat == VTK_LUMINANCE_ALPHA)
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          *output++ = static_cast<unsigned short>(cptr[3]*alpha);
          input += inIncr;
          }
        }
      else // outFormat == VTK_LUMINANCE
        {
        while (--i >= 0)
          {
          cptr = vtkLinearLookup16((double)*input, table, maxIndex, shift, scale);
          *output++ = static_cast<unsigned short>(cptr[0]*0.30 + cptr[1]*0.59 +
                                                 cptr[2]*0.11 + 0.5);
          input += inIncr;
          }
        }
      }//no log scale
    }//alpha blending
}

//----------------------------------------------------------------------------
void vtkLookupTable16::MapScalarsThroughTable2(void *input,
                                             unsigned char *output,
                                             int inputDataType,
                                             int numberOfValues,
                                             int inputIncrement,
                                             int outputFormat)
{
  if (this->UseMagnitude && inputIncrement > 1)
    {
assert(0);
//    switch (inputDataType)
//      {
//      vtkTemplateMacro(
//        vtkLookupTableMapMag(this,static_cast<VTK_TT*>(input),(unsigned short*)output,
//                             numberOfValues,inputIncrement,outputFormat);
//        return
//        );
//      case VTK_BIT:
//        vtkErrorMacro("Cannot comput magnitude of bit array.");
//        break;
//      default:
//        vtkErrorMacro(<< "MapImageThroughTable: Unknown input ScalarType");
//      }
    }

  switch (inputDataType)
    {
    case VTK_BIT:
      {
assert(0);
      //vtkIdType i, id;
      //vtkBitArray *bitArray = vtkBitArray::New();
      //bitArray->SetVoidArray(input,numberOfValues,1);
      //vtkUnsignedCharArray *newInput = vtkUnsignedCharArray::New();
      //newInput->SetNumberOfValues(numberOfValues);
      //for (id=i=0; i<numberOfValues; i++, id+=inputIncrement)
      //  {
      //  newInput->SetValue(i, bitArray->GetValue(id));
      //  }
      //vtkLookupTableMapData(this,
      //                      static_cast<unsigned char*>(newInput->GetPointer(0)),
      //                      output,numberOfValues,
      //                      inputIncrement,outputFormat);
      //newInput->Delete();
      //bitArray->Delete();
      }
      break;

    vtkTemplateMacro(
      vtkLookupTable16MapData(this,static_cast<VTK_TT*>(input),(unsigned short*)output,
                            numberOfValues,inputIncrement,outputFormat)
      );
    default:
      vtkErrorMacro(<< "MapImageThroughTable: Unknown input ScalarType");
      return;
    }
}

void vtkLookupTable16::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

}
