/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSOPCLASSEXTENDEDNEGOCIATIONSUB_H
#define GDCMSOPCLASSEXTENDEDNEGOCIATIONSUB_H

#include "gdcmServiceClassApplicationInformation.h"

namespace gdcm
{
namespace network
{

/**
 * \brief SOPClassExtendedNegociationSub
 * PS 3.7
 * Table D.3-11
 * SOP CLASS EXTENDED NEGOTIATION SUB-ITEM FIELDS
 * (A-ASSOCIATE-RQ and A-ASSOCIATE-AC)
 */
class SOPClassExtendedNegociationSub
{
public:
  SOPClassExtendedNegociationSub();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  size_t Size() const;
  void Print(std::ostream &os) const;

  void SetTuple(const char *uid, uint8_t levelofsupport = 3,
    uint8_t levelofdigitalsig = 0,
    uint8_t elementcoercion = 2);

private:
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength;
  uint16_t UIDLength;
  std::string /*SOP-class-uid*/ Name; // UID
  ServiceClassApplicationInformation SCAI;
};

} // end namespace network

} // end namespace gdcm

#endif // GDCMSOPCLASSEXTENDEDNEGOCIATIONSUB_H
