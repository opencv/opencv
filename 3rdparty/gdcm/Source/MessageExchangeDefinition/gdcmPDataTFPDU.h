/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPDATATFPDU_H
#define GDCMPDATATFPDU_H

#include "gdcmTypes.h"
#include "gdcmPresentationDataValue.h"
#include "gdcmBasePDU.h"
#include <limits>

namespace gdcm
{

namespace network
{

/**
 * \brief PDataTFPDU
 * Table 9-22
 * P-DATA-TF PDU FIELDS
 */
class GDCM_EXPORT PDataTFPDU : public BasePDU
{
public:
  PDataTFPDU();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  /// \internal Compute Size
  size_t Size() const;

  void AddPresentationDataValue( PresentationDataValue const &pdv ) {
    V.push_back( pdv );
    assert(Size() < std::numeric_limits<uint32_t>::max());
    ItemLength = (uint32_t)Size() - 6;
    }

  typedef std::vector<PresentationDataValue>::size_type SizeType;
  PresentationDataValue const &GetPresentationDataValue(SizeType i) const {
    assert( !V.empty() && i < V.size() );
    return V[i];
  }
  SizeType GetNumberOfPresentationDataValues() const {
    return V.size();
    }

  void Print(std::ostream &os) const;
  bool IsLastFragment() const;

protected:
  std::istream &ReadInto(std::istream &is, std::ostream &os);
private:
  static const uint8_t ItemType; // PDUType ?
  static const uint8_t Reserved2;
  uint32_t ItemLength; // PDU Length ?
  std::vector<PresentationDataValue> V;
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMPDATATFPDU_H
