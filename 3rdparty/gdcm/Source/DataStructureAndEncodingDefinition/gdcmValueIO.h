/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMVALUEIO_H
#define GDCMVALUEIO_H

#include "gdcmTypes.h"

namespace gdcm
{
/**
 * \brief Class to dispatch template calls
 */
template <typename TDE, typename TSwap, typename TType=uint8_t>
class /*GDCM_EXPORT*/ ValueIO
{
public:
  static std::istream &Read(std::istream &is, Value& v, bool readvalues);

  static const std::ostream &Write(std::ostream &os, const Value& v);
};

} // end namespace gdcm

#include "gdcmValueIO.txx"

#endif //GDCMVALUEIO_H
