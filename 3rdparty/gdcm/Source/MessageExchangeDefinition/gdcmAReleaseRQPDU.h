/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMARELEASERQPDU_H
#define GDCMARELEASERQPDU_H

#include "gdcmTypes.h"
#include "gdcmBasePDU.h"

namespace gdcm
{

namespace network
{

/**
 * \brief AReleaseRQPDU
 * Table 9-24
 * A-RELEASE-RQ PDU FIELDS
 */
class AReleaseRQPDU : public BasePDU
{
public:
  AReleaseRQPDU();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;
  size_t Size() const;
  void Print(std::ostream &os) const;
  bool IsLastFragment() const { return true; }
private:
  static const uint8_t ItemType; // PDUType ?
  static const uint8_t Reserved2;
  uint32_t ItemLength; // PDU Length
  static const uint32_t Reserved7_10;
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMARELEASERQPDU_H
