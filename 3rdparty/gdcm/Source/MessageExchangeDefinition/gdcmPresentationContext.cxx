/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPresentationContext.h"

#include "gdcmUIDs.h"
#include "gdcmAttribute.h"
#include "gdcmMediaStorage.h"
#include "gdcmTransferSyntax.h"

#include <limits>

namespace gdcm
{

PresentationContext::PresentationContext( )
{
  ID = 0x01;
}

PresentationContext::PresentationContext( UIDs::TSName asname, UIDs::TSName tsname )
{
  ID = 0x01;
  const char *asnamestr = UIDs::GetUIDString( asname );
  AbstractSyntax = asnamestr;

  const char *tsnamestr = UIDs::GetUIDString( tsname );
  AddTransferSyntax( tsnamestr );
}

void PresentationContext::AddTransferSyntax( const char *tsstr )
{
  TransferSyntaxes.push_back( tsstr );
}

void PresentationContext::SetPresentationContextID( uint8_t id )
{
  assert( id );
  ID = id;
}

uint8_t PresentationContext::GetPresentationContextID() const
{
  return ID;
}

void PresentationContext::Print(std::ostream &os) const
{
  os << "AbstractSyntax:" << AbstractSyntax << std::endl;
  std::vector<std::string>::const_iterator it = TransferSyntaxes.begin();
  for( ; it != TransferSyntaxes.end(); ++it )
    {
    os << *it << std::endl;
    }
}

} // end namespace gdcm
