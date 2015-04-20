/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPixmap.h"

namespace gdcm
{
/*
 * PICKER-16-MONO2-Nested_icon.dcm:
(0088,0200) SQ (Sequence with undefined length #=1)     # u/l, 1 PixmapSequence
  (fffe,e000) na (Item with undefined length #=10)        # u/l, 1 Item
    (0028,0002) US 1                                        #   2, 1 SamplesPerPixel
    (0028,0004) CS [MONOCHROME2]                            #  12, 1 PhotometricInterpretation
    (0028,0010) US 64                                       #   2, 1 Rows
    (0028,0011) US 64                                       #   2, 1 Columns
    (0028,0034) IS [1\1]                                    #   4, 2 PixelAspectRatio
    (0028,0100) US 8                                        #   2, 1 BitsAllocated
    (0028,0101) US 8                                        #   2, 1 BitsStored
    (0028,0102) US 7                                        #   2, 1 HighBit
    (0028,0103) US 0                                        #   2, 1 PixelRepresentation
    (7fe0,0010) OW 0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000... # 4096, 1 PixelData
  (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
(fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/

Pixmap::Pixmap():Overlays(),Curves(),Icon(new IconImage) {}

Pixmap::~Pixmap() {}

bool Pixmap::AreOverlaysInPixelData() const
{
  int total = 0;
  std::vector<Overlay>::const_iterator it = Overlays.begin();
  for(; it != Overlays.end(); ++it)
    {
    total += (int)it->IsInPixelData();
    }
  assert( total == (int)GetNumberOfOverlays() || !total );
  return total != 0;
}

void Pixmap::Print(std::ostream &os) const
{
  Bitmap::Print(os);
  for( std::vector<Overlay>::const_iterator it = Overlays.begin();
    it != Overlays.end(); ++it)
    {
    it->Print( os );
    }
  for( std::vector<Curve>::const_iterator it = Curves.begin();
    it != Curves.end(); ++it)
    {
    it->Print( os );
    }
}

}
