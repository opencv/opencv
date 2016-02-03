/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMSEGMENTEDPALETTECOLORLOOKUPTABLE_H
#define GDCMSEGMENTEDPALETTECOLORLOOKUPTABLE_H

#include "gdcmLookupTable.h"

namespace gdcm
{

/**
 * \brief SegmentedPaletteColorLookupTable class
 */
class GDCM_EXPORT SegmentedPaletteColorLookupTable : public LookupTable
{
public:
  SegmentedPaletteColorLookupTable();
  ~SegmentedPaletteColorLookupTable();
  void Print(std::ostream &) const {}

  /// Initialize a SegmentedPaletteColorLookupTable
  void SetLUT(LookupTableType type, const unsigned char *array,
    unsigned int length);

};

} // end namespace gdcm

#endif //GDCMSEGMENTEDPALETTECOLORLOOKUPTABLE_H
