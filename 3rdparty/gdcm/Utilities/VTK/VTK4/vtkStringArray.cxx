/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkStringArray.h"

#include "vtkObjectFactory.h"

#include <vector>
#include <string>

vtkCxxRevisionMacro(vtkStringArray, "$Revision: 1.1 $")
vtkStandardNewMacro(vtkStringArray)

struct vtkStringArrayInternals
{
  std::vector< std::string > Internal;
};

vtkStringArray::vtkStringArray()
{
  Internal = new vtkStringArrayInternals;
}

vtkStringArray::~vtkStringArray()
{
  delete Internal;
}

//std::string &vtkStringArray::GetValue(unsigned int i)
const char *vtkStringArray::GetValue(unsigned int i)
{
  return Internal->Internal[i].c_str();
}

int vtkStringArray::GetNumberOfValues()
{
  return Internal->Internal.size();
}

vtkIdType vtkStringArray::InsertNextValue(const char *f)
{
  Internal->Internal.push_back( f );
  return Internal->Internal.size();
}

vtkIdType vtkStringArray::InsertNextValue(std::string const & f)
{
  Internal->Internal.push_back( f );
  return Internal->Internal.size();
}

vtkIdType vtkStringArray::GetSize()
{
  return Internal->Internal.size();
}

//----------------------------------------------------------------------------
void vtkStringArray::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
