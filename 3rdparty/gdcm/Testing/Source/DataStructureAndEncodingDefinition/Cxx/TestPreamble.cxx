/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPreamble.h"
#include "gdcmFileMetaInformation.h"

int TestPreamble(int, char *[])
{
{
  gdcm::Preamble p;
}
  gdcm::FileMetaInformation * m_pFileMetaInformation = new gdcm::FileMetaInformation();
  gdcm::Preamble *p = new gdcm::Preamble();

  m_pFileMetaInformation->SetPreamble( *p );
  m_pFileMetaInformation->Clear();
  delete m_pFileMetaInformation;
  m_pFileMetaInformation = NULL ;
    delete p;
  return 0;
}
