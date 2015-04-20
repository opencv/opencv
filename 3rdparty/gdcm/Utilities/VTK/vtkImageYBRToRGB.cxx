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
  Module:    $RCSfile: vtkImageYBRToRGB.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageYBRToRGB.h"

#include "vtkImageData.h"
#include "vtkImageProgressIterator.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(vtkImageYBRToRGB, "$Revision: 1.31 $")
vtkStandardNewMacro(vtkImageYBRToRGB)

//----------------------------------------------------------------------------
vtkImageYBRToRGB::vtkImageYBRToRGB()
{
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
template <class T>
void vtkImageYBRToRGBExecute(vtkImageYBRToRGB *self,
                             vtkImageData *inData,
                             vtkImageData *outData,
                             int outExt[6], int id, T *)
{
  vtkImageIterator<T> inIt(inData, outExt);
  vtkImageProgressIterator<T> outIt(outData, outExt, self, id);
  int idxC;

  // find the region to loop over
  int maxC = inData->GetNumberOfScalarComponents()-1;

  int R, G, B;
  // Loop through ouput pixels
  while (!outIt.IsAtEnd())
    {
    T* inSI = inIt.BeginSpan();
    T* outSI = outIt.BeginSpan();
    T* outSIEnd = outIt.EndSpan();
    while (outSI != outSIEnd)
      {
      // Pixel operation
#if 1
#if 1
      unsigned char a = (unsigned char)(*inSI); ++inSI;
      unsigned char b = (unsigned char)(*inSI); ++inSI;
      unsigned char c = (unsigned char)(*inSI); ++inSI;

      R = 38142 *(a-16) + 52298 *(c -128);
      G = 38142 *(a-16) - 26640 *(c -128) - 12845 *(b -128);
      B = 38142 *(a-16) + 66093 *(b -128);

      R = (R+16384)>>15;
      G = (G+16384)>>15;
      B = (B+16384)>>15;

#else
      int /*unsigned char*/ y = (unsigned char)(*inSI); ++inSI;
      y -= 16;
      unsigned char u = (unsigned char)(*inSI); ++inSI;
      int Cb = (int)u - 128;
      unsigned char v = (unsigned char)(*inSI); ++inSI;
      int Cr = (int)v - 128;

 //     R = y + (1.4075 * (v - 128));
 //     G = y - (0.3455 * (u - 128) - (0.7169 * (v - 128)));
 //     B = y + (1.7790 * (u - 128));
      R = y                + 1.40200 * Cr + 0.5;
      G = y - 0.34414 * Cb - 0.71414 * Cr + 0.5;
      B = y + 1.77200 * Cb + 0.5;


      //int a = (int)y - 16;
      //int b = (int)u - 128;
      //int c = (int)v - 128;

      //R = ( 1.164 * a + 0.    * b + 1.596 * c );
      //G = ( 1.164 * a + 0.391 * b + 0.813 * c );
      //B = ( 1.164 * a + 2.018 * b + 0.    * c );
#endif
      if (R < 0)   R = 0;
      if (G < 0)   G = 0;
      if (B < 0)   B = 0;
      if (R > 255) R = 255;
      if (G > 255) G = 255;
      if (B > 255) B = 255;
#endif

/*
      double y = *inSI; ++inSI;
      double u = *inSI; ++inSI;
      double v = *inSI; ++inSI;
      unsigned char R,G,B;
      double maxval = 255.;

  double dr = y + 1.4020 * v - 0.7010 * maxval;
  double dg = y - 0.3441 * u - 0.7141 * v + 0.5291 * maxval;
  double db = y + 1.7720 * u - 0.8859 * maxval;
  R = (dr < 0.0) ? 0 : ((dr+0.5) > maxval) ? maxval : (unsigned char)(dr+0.5);
  G = (dg < 0.0) ? 0 : ((dg+0.5) > maxval) ? maxval : (unsigned char)(dg+0.5);
  B = (db < 0.0) ? 0 : ((db+0.5) > maxval) ? maxval : (unsigned char)(db+0.5);
*/

      // assign output.
      *outSI = (T)(R); ++outSI;
      *outSI = (T)(G); ++outSI;
      *outSI = (T)(B); ++outSI;

      for (idxC = 3; idxC <= maxC; idxC++)
        {
        *outSI++ = *inSI++;
        }
      }
    inIt.NextSpan();
    outIt.NextSpan();
    }
}

//----------------------------------------------------------------------------
void vtkImageYBRToRGB::ThreadedExecute (vtkImageData *inData,
                                       vtkImageData *outData,
                                       int outExt[6], int id)
{
  vtkDebugMacro(<< "Execute: inData = " << inData
    << ", outData = " << outData);

  // this filter expects that input is the same type as output.
  if (inData->GetScalarType() != outData->GetScalarType())
    {
    vtkErrorMacro(<< "Execute: input ScalarType, " << inData->GetScalarType()
    << ", must match out ScalarType " << outData->GetScalarType());
    return;
    }
  if (inData->GetScalarType() != VTK_UNSIGNED_CHAR )
    {
    return;
    }

  // need three components for input and output
  if (inData->GetNumberOfScalarComponents() < 3)
    {
    vtkErrorMacro("Input has too few components");
    return;
    }
  if (outData->GetNumberOfScalarComponents() < 3)
    {
    vtkErrorMacro("Output has too few components");
    return;
    }

  switch (inData->GetScalarType())
    {
    vtkTemplateMacro(
      vtkImageYBRToRGBExecute(this, inData,
                              outData, outExt, id, static_cast<VTK_TT *>(0)));
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}

//----------------------------------------------------------------------------
void vtkImageYBRToRGB::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
