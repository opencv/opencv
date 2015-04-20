/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMENUMERATEDVALUES_H
#define GDCMENUMERATEDVALUES_H

#include "gdcmTypes.h"

namespace gdcm
{
/**
 * \brief
 * Element. A Data Element with Enumerated Values that does not have a Value equivalent to one of the
 * Values specified in this standard has an invalid value within the scope of a specific Information
 * Object/SOP Class definition.
 * Note:
 *   1. Patient Sex (0010, 0040) is an example of a Data Element having Enumerated Values. It is defined to
 *   have a Value that is either "M", "F", or "O" (see PS 3.3). No other Value shall be given to this Data
 *   Element.
 *   2. Future modifications of this standard may add to the set of allowed values for Data Elements with
 *   Enumerated Values. Such additions by themselves may or may not require a change in SOP Class
 *   UIDs, depending on the semantics of the Data Element.
 */
class GDCM_EXPORT EnumeratedValues
{
public:
  EnumeratedValues() {
  }
private:
};

} // end namespace gdcm

#endif //GDCMENUMERATEDVALUES_H
