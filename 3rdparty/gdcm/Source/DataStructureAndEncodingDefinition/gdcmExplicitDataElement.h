/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMEXPLICITDATAELEMENT_H
#define GDCMEXPLICITDATAELEMENT_H

#include "gdcmDataElement.h"

namespace gdcm
{
/**
 * \brief Class to read/write a DataElement as Explicit Data Element
 * \note bla
 */
class GDCM_EXPORT ExplicitDataElement : public DataElement
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
  std::istream &ReadWithLength(std::istream &is, VL & length);

  template <typename TSwap>
  const std::ostream &Write(std::ostream &os) const;
};

} // end namespace gdcm

#include "gdcmExplicitDataElement.txx"

#endif //GDCMEXPLICITDATAELEMENT_H
