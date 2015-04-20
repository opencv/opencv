/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSERVICECLASSAPPLICATIONINFORMATION_H
#define GDCMSERVICECLASSAPPLICATIONINFORMATION_H

#include "gdcmTypes.h"

namespace gdcm
{

namespace network
{

/**
 * PS 3.4
 * Table B.3-1
 * SERVICE-CLASS-APPLICATION-INFORMATION (A-ASSOCIATE-RQ)
 */
class ServiceClassApplicationInformation
{
public:
  ServiceClassApplicationInformation();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  size_t Size() const;
  void SetTuple(uint8_t levelofsupport, uint8_t levelofdigitalsig,
    uint8_t elementcoercion);

  void Print(std::ostream &os) const;
private:
  uint8_t InternalArray[6];
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMSERVICECLASSAPPLICATIONINFORMATION_H
