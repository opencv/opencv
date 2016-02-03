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
  Module:    $RCSfile: vtkLookupTable16.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkLookupTable16 -
// .SECTION Description
//
// .SECTION Caveats
//
// .SECTION See Also
// vtkLookupTable

#ifndef VTKLOOKUPTABLE16_H
#define VTKLOOKUPTABLE16_H

#include "vtkLookupTable.h"
#include "vtkUnsignedShortArray.h"

class VTK_EXPORT vtkLookupTable16 : public vtkLookupTable
{
public:
  static vtkLookupTable16 *New();

  vtkTypeRevisionMacro(vtkLookupTable16,vtkLookupTable);
  void PrintSelf(ostream& os, vtkIndent indent);

  void Build();

  void SetNumberOfTableValues(vtkIdType number);

  unsigned char *WritePointer(const vtkIdType id, const int number);

  unsigned short *GetPointer(const vtkIdType id) {
    return this->Table16->GetPointer(4*id); };

protected:
  vtkLookupTable16(int sze=256, int ext=256);
  ~vtkLookupTable16();

  vtkUnsignedShortArray *Table16;

void MapScalarsThroughTable2(void *input,
                                             unsigned char *output,
                                             int inputDataType,
                                             int numberOfValues,
                                             int inputIncrement,
                                             int outputFormat);

private:
  vtkLookupTable16(const vtkLookupTable16&);  // Not implemented.
  void operator=(const vtkLookupTable16&);  // Not implemented.
};

//----------------------------------------------------------------------------
inline unsigned char *vtkLookupTable16::WritePointer(const vtkIdType id,
                                                   const int number)
{
  //this->InsertTime.Modified();
  return (unsigned char*)this->Table16->WritePointer(4*id,4*number);
}

#endif
