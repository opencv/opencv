/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMUNEXPLICITIMPLICITDATAELEMENT_H
#define GDCMUNEXPLICITIMPLICITDATAELEMENT_H

#include "gdcmDataElement.h"

namespace gdcm
{
// Data Element (ExplicitImplicit)
/**
 * \brief Class to read/write a DataElement as ExplicitImplicit Data Element
 * This class gather two known bugs:
 * 1. GDCM 1.2.0 would rewrite VR=UN Value Length on 2 bytes instead of 4 bytes
 * 2. GDCM 1.2.0 would also rewrite DataElement as Implicit when the VR would not be known
 *    this would only happen in some very rare cases.
 * gdcm 2.X design could handle bug #1 or #2 exclusively, this class can now handle
 * file which have both issues.
 * See: gdcmData/TheralysGDCM120Bug.dcm
 */
class GDCM_EXPORT UNExplicitImplicitDataElement : public DataElement
{
public:
  VL GetLength() const;

  template <typename TSwap>
  std::istream &Read(std::istream &is);

  template <typename TSwap>
  std::istream &ReadPreValue(std::istream &is);

  template <typename TSwap>
  std::istream &ReadValue(std::istream &is);

  // PURPOSELY do not provide an implementation for writing !
  //template <typename TSwap>
  //const std::ostream &Write(std::ostream &os) const;
};

} // end namespace gdcm

#include "gdcmUNExplicitImplicitDataElement.txx"

#endif //GDCMUNEXPLICITIMPLICITDATAELEMENT_H
