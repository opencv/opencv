/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCODER_H
#define GDCMCODER_H

#include "gdcmTypes.h"
#include "gdcmDataElement.h" // FIXME

namespace gdcm
{

class TransferSyntax;
class DataElement;
/**
 * \brief Coder
 */
class GDCM_EXPORT Coder
{
public:
  virtual ~Coder() {}

  /// Return whether this coder support this transfer syntax (can code it)
  virtual bool CanCode(TransferSyntax const &) const = 0;

  // Note: in / out are reserved keyword in C#. Change to in_ / out_

  /// Code
  virtual bool Code(DataElement const &in_, DataElement &out_) { (void)in_; (void)out_; return false; }
protected:
  virtual bool InternalCode(const char *bv, unsigned long len, std::ostream &os) { (void)bv;(void)os;(void)len;return false; }
};

} // end namespace gdcm

#endif //GDCMCODER_H
