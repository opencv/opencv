/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkStringArray -
// .SECTION Description
//
// .SECTION
//
// .SECTION See Also

#ifndef VTKSTRINGARRAY_H
#define VTKSTRINGARRAY_H

#ifdef __vtkStringArray_h
#error Something went terribly wrong
#endif

#include "vtkObject.h"

#include <string>

class vtkStringArrayInternals;
class VTK_EXPORT vtkStringArray : public vtkObject
{
public:
  static vtkStringArray *New();
  vtkTypeRevisionMacro(vtkStringArray,vtkObject);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

//BTX
  //std::string &GetValue(unsigned int i);
  vtkIdType InsertNextValue(std::string const & f);
//ETX
  const char *GetValue(unsigned int i);
  int GetNumberOfValues();
  vtkIdType InsertNextValue(const char *f);

  vtkIdType GetSize();

protected:
  vtkStringArray();
  ~vtkStringArray();

private:
  vtkStringArray(const vtkStringArray&);  // Not implemented.
  void operator=(const vtkStringArray&);  // Not implemented.

  vtkStringArrayInternals *Internal;
};

#endif
