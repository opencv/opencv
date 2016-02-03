/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMAASSOCIATERJPDU_H
#define GDCMAASSOCIATERJPDU_H

#include "gdcmTypes.h"
#include "gdcmBasePDU.h"

namespace gdcm
{

namespace network
{

/**
 * \brief AAssociateRJPDU
 * Table 9-21
 * ASSOCIATE-RJ PDU FIELDS
 */
class AAssociateRJPDU : public BasePDU
{
public:
  AAssociateRJPDU();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;
  void Print(std::ostream &os) const;
  size_t Size() const;
  bool IsLastFragment() const { return true; }
private:
  static const uint8_t ItemType; // PDUType ?
  static const uint8_t Reserved2;
  uint32_t ItemLength; // PDU Length ?
  static const uint8_t Reserved8;
  uint8_t Result;
  uint8_t Source;
  uint8_t Reason; // diag ?
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMAASSOCIATERJPDU_H
