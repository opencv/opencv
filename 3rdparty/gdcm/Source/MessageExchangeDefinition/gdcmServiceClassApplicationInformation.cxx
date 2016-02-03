/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmServiceClassApplicationInformation.h"

namespace gdcm
{
namespace network
{

ServiceClassApplicationInformation::ServiceClassApplicationInformation()
{
  InternalArray[0] = 3; // Level of Support
  InternalArray[1] = 0; // Reserved
  InternalArray[2] = 0; // Level of Digital Signature support
  InternalArray[3] = 0; // Reserved
  InternalArray[4] = 2; // Element coercion
  InternalArray[5] = 0; // Reserved
}

std::istream &ServiceClassApplicationInformation::Read(std::istream &is)
{
  is.read( (char*)InternalArray, sizeof(InternalArray) );
  return is;
}

const std::ostream &ServiceClassApplicationInformation::Write(std::ostream &os) const
{
  assert( InternalArray[0] < 4 );
  assert( InternalArray[1] == 0 );
  assert( InternalArray[2] < 4 );
  assert( InternalArray[3] == 0 );
  assert( InternalArray[4] < 3 );
  assert( InternalArray[5] == 0 );
  os.write( (char*)InternalArray, sizeof(InternalArray) );
  return os;
}

size_t ServiceClassApplicationInformation::Size() const
{
  assert( sizeof(InternalArray) == 6 );
  return 6;
}

void ServiceClassApplicationInformation::Print(std::ostream &os) const
{
  os << "ServiceClassApplicationInformation: " << std::endl;
  os << " Level of Support: " << (int)InternalArray[0] << std::endl;
  os << " Level of Digital Signature support: " << (int)InternalArray[2] << std::endl;
  os << " Element coercion: " << (int)InternalArray[4] << std::endl;
}

/*
Level of support This byte field defines the supported storage level of the
                 Association-acceptor. It shall be encoded as an
                 unsigned binary integer and shall use one of the
                 following values:
                  0 - level 0 SCP
                  1 - level 1 SCP
                  2 - level 2 SCP
                  3 - N/A - Association-acceptor is SCU only
                 If extended negotiation is not supported, no
                 assumptions shall be made by the Association-
                 requester about the capabilities of the Association-
                 acceptor based upon this extended negotiation.

 Level of Digital A Level 2 SCP may further define its behavior in this
Signature support byte field.
                   0 – The signature level is unspecified, the AE is an
                  SCU only, or the AE is not a level 2 SCP
                   1 – signature level 1
                   2 – signature level 2
                   3 – signature level 3
                  If extended negotiation is not supported, no
                  assumptions shall be made by the Association-
                  requester about the capabilities of the Association-
                  acceptor based upon this extended negotiation.

Element Coercion This byte field defines whether the Association-acceptor
                 may coerce Data Elements. It shall be encoded as an
                 unsigned binary integer and shall use one of the
                 following values:
                  0 - does not coerce any Data Element
                  1 - may coerce Data Elements
                  2 - N/A - Association-acceptor is SCU only
                 If extended negotiation is not supported, no
                 assumptions shall be made by the Association-
                 requester about the capabilities of the Association-
                 acceptor based upon this extended negotiation.
*/

void ServiceClassApplicationInformation::SetTuple(uint8_t levelofsupport, uint8_t levelofdigitalsig, uint8_t elementcoercion)
{
  if( levelofsupport < 4 )
    InternalArray[0] = levelofsupport;
  if( levelofdigitalsig < 4 )
    InternalArray[2] = levelofdigitalsig;
  if( elementcoercion < 3 )
    InternalArray[4] = elementcoercion;
}

} // end namespace network
} // end namespace gdcm
