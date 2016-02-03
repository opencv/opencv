/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFragment.h"

namespace gdcm
{

VL Fragment::GetLength() const
{
  assert( !ValueLengthField.IsUndefined() );
  assert( !ValueField || ValueField->GetLength() == ValueLengthField );
  return TagField.GetLength() + ValueLengthField.GetLength()
    + ValueLengthField;
}

VL Fragment::ComputeLength() const
{
  const ByteValue *bv = GetByteValue();
  assert( bv );
  assert( !ValueLengthField.IsUndefined() );
  //assert( !ValueField || ValueField->GetLength() == ValueLengthField );
  return TagField.GetLength() + ValueLengthField.GetLength()
    + bv->ComputeLength() /*ValueLengthField*/;
}

} // end namespace gdcm
