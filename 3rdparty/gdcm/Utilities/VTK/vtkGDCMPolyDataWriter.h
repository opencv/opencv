/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkGDCMPolyDataWriter - writer DICOM PolyData files (Contour Data...)
// .SECTION Description
// For now only support RTSTRUCT (RT Structure Set Storage)
// .SECTION TODO
// Need to do the same job for DVH Sequence/DVH Data...
// .SECTION Warning
//
// .SECTION See Also
// vtkGDCMImageReader vtkGDCMPolyDataReader vtkRTStructSetProperties


#ifndef VTKGDCMPOLYDATAWRITER_H
#define VTKGDCMPOLYDATAWRITER_H

#include "vtkPolyDataWriter.h"
#include "vtkStringArray.h"
#include "vtkStdString.h"


class vtkMedicalImageProperties;
class vtkRTStructSetProperties;
//BTX
namespace gdcm { class File; }
//ETX
class VTK_EXPORT vtkGDCMPolyDataWriter : public vtkPolyDataWriter
{
public:
  static vtkGDCMPolyDataWriter *New();
  vtkTypeRevisionMacro(vtkGDCMPolyDataWriter,vtkPolyDataWriter);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/Get the filename of the file to be read
//  vtkSetStringMacro(FileName);
//  vtkGetStringMacro(FileName);

  // Description:
  // Get the medical image properties object
//  vtkGetObjectMacro(MedicalImageProperties, vtkMedicalImageProperties);
  virtual void SetMedicalImageProperties(vtkMedicalImageProperties *pd);

  virtual void SetRTStructSetProperties(vtkRTStructSetProperties *pd);


  //this function will initialize the contained rtstructset with
  //the inputs of the writer and the various extra information
  //necessary for writing a complete rtstructset.
  //NOTE: inputs must be set BEFORE calling this function!
  //NOTE: the number of outputs for the appendpolydata MUST MATCH the ROI vectors!
  void InitializeRTStructSet(vtkStdString inDirectory,
     vtkStdString inStructLabel, vtkStdString inStructName,
     vtkStringArray* inROINames,
     vtkStringArray* inROIAlgorithmName,
     vtkStringArray* inROIType);

  // make parent class public...
  void SetNumberOfInputPorts(int n);

protected:
  vtkGDCMPolyDataWriter();
  ~vtkGDCMPolyDataWriter();

  vtkMedicalImageProperties *MedicalImageProperties;
  vtkRTStructSetProperties *RTStructSetProperties;

  void WriteData();
//BTX
  void WriteRTSTRUCTInfo(gdcm::File &file);
  void WriteRTSTRUCTData(gdcm::File &file, int num);
//ETX

private:
  vtkGDCMPolyDataWriter(const vtkGDCMPolyDataWriter&);  // Not implemented.
  void operator=(const vtkGDCMPolyDataWriter&);  // Not implemented.
};

#endif
