/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMEXPLICITIMPLICITDATAELEMENT_H
#define GDCMEXPLICITIMPLICITDATAELEMENT_H

#include "gdcmDataElement.h"

namespace gdcm
{
// Data Element (ExplicitImplicit)
/**
 * \brief Class to read/write a DataElement as ExplicitImplicit Data Element
 * \note This only happen for some Philips images
 * Should I derive from ExplicitDataElement instead ?
 * This is the class that is the closest the GDCM1.x parser. At each element we try first
 * to read it as explicit, if this fails, then we try again as an implicit element.
 */
class GDCM_EXPORT ExplicitImplicitDataElement : public DataElement
{
public:
  VL GetLength() const;

  template <typename TSwap>
  std::istream &Read(std::istream &is);

  template <typename TSwap>
  std::istream &ReadPreValue(std::istream &is);

  template <typename TSwap>
  std::istream &ReadValue(std::istream &is, bool readvalues = true);

  template <typename TSwap>
  std::istream &ReadWithLength(std::istream &is, VL & length)
    {
    return Read<TSwap>(is); (void)length;
    }

  // PURPOSELY do not provide an implementation for writing !
  //template <typename TSwap>
  //const std::ostream &Write(std::ostream &os) const;
};

} // end namespace gdcm

#include "gdcmExplicitImplicitDataElement.txx"

#endif //GDCMEXPLICITIMPLICITDATAELEMENT_H
