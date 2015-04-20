/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmValidate.h"

namespace gdcm
{
//-----------------------------------------------------------------------------
Validate::Validate():F(0)
{
}
//-----------------------------------------------------------------------------
Validate::~Validate()
{
}

//-----------------------------------------------------------------------------
void Validate::Validation()
{
  if(!F) return;
  V.GetHeader().SetPreamble( F->GetHeader().GetPreamble() );
  //FileMetaInformation &fmi = F.GetHeader();
  FileMetaInformation fmi( F->GetHeader() );
  fmi.FillFromDataSet( F->GetDataSet() );
  std::cout << "Validation" << std::endl;
  //std::cout << fmi << std::endl;
  //std::cout << fmi.GetDataElement( Tag(0x0002, 0x0002) ) << std::endl;
  V.SetHeader( fmi );
  V.SetDataSet( F->GetDataSet() );
  //std::cout << V.GetHeader() << std::endl;
}

}
