/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCommandDataSet.h"

#include "gdcmVR.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmTag.h"

namespace gdcm
{

std::istream &CommandDataSet::Read(std::istream &is)
{
  this->DataSet::Read<ImplicitDataElement,SwapperNoOp>(is);
  return is;
}

std::ostream &CommandDataSet::Write(std::ostream &os) const
{
  this->DataSet::Write<ImplicitDataElement,SwapperNoOp>(os);
  return os;
}

} // end namespace gdcm
