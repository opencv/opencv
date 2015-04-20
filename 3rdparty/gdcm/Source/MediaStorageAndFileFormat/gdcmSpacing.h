/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSPACING_H
#define GDCMSPACING_H

#include "gdcmTypes.h"
#include "gdcmAttribute.h"

namespace gdcm
{
/**
  It all began with a mail to WG6:

Subject: 	Imager Pixel Spacing vs Pixel Spacing
Body: 	[Apologies for the duplicate post, namely to David Clunie & OFFIS team]

I have been trying to understand CP-586 in the following two cases:

On the one hand:
- DISCIMG/IMAGES/CRIMAGE taken from
http://dclunie.com/images/pixelspacingtestimages.zip

And on the other hand:
- http://gdcm.sourceforge.net/thingies/cr_pixelspacing.dcm

If I understand correctly the CP, one is required to use Pixel Spacing
for measurement ('true size' print) instead of Imager Pixel Spacing,
since the two attributes are present and Pixel Spacing is different from
Imager Pixel Spacing.

If this is correct, then the test data DISCIMG/IMAGES/CRIMAGE is
incorrect. If this is incorrect (ie. I need to use Imager Pixel
Spacing), then the display of cr_pixelspacing.dcm for measurement will
be incorrect.

Could someone please let me know what am I missing here? I could not
find any information in any header that would allow me to differentiate
those.

Thank you for your time,

Ref:
  http://lists.nema.org/scripts/lyris.pl?sub=488573&id=400720477
*/

/**
 * \brief Class for Spacing
 *
 * See PS 3.3-2008, Table C.7-11b IMAGE PIXEL MACRO ATTRIBUTES

Ratio of the vertical size and horizontal size
of the pixels in the image specified by a
pair of integer values where the first value
is the vertical pixel size, and the second
value is the horizontal pixel size. Required
if the aspect ratio values do not have a
ratio of 1:1 and the physical pixel spacing is
not specified by Pixel Spacing (0028,0030),
or Imager Pixel Spacing (0018,1164) or
Nominal Scanned Pixel Spacing
(0018,2010), either for the entire Image or
per-frame in a Functional Group Macro.
See C.7.6.3.1.7.


PS 3.3-2008
10.7.1.3 Pixel Spacing Value Order and Valid Values
All pixel spacing related attributes shall have non-zero values, except when there is only a single row or
column or pixel of data present, in which case the corresponding value may be zero.

Ref:
http://gdcm.sourceforge.net/wiki/index.php/Imager_Pixel_Spacing
 */
class GDCM_EXPORT Spacing
{
public :
  Spacing();
  ~Spacing();

  // Here are the list of spacing we support:
  // (0018,0088) DS [1.500000]                                         # 8,1 Spacing Between Slices
  // (0018,1164) DS [0.5\0.5 ]                                         # 8,2 Imager Pixel Spacing
  // (0018,2010) DS [0.664062\0.664062 ]                               # 18,2 Nominal Scanned Pixel Spacing
  // (0018,7022) DS [0.125\0.125 ]                                     # 12,2 Detector Element Spacing
  // (0028,0030) DS [0.25\0.25 ]                                       # 10,2 Pixel Spacing
  // > (0028,0a02) CS [FIDUCIAL]                                         # 8,1 Pixel Spacing Calibration Type
  // > (0028,0a04) LO [Used fiducial ]                                   # 14,1 Pixel Spacing Calibration Description
  // (0028,0034) IS [4\3 ]                                             # 4,2 Pixel Aspect Ratio
  // (3002,0011) DS [0.8\0.8 ]                                         # 8,2 Image Plane Pixel Spacing

  // Here is the list of Spacing we do not support:
  // <entry group="0018" element="7041" vr="LT" vm="1" name="Grid Spacing Material"/>
  // <entry group="0018" element="9030" vr="FD" vm="1" name="Tag Spacing First Dimension"/>
  // <entry group="0018" element="9218" vr="FD" vm="1" name="Tag Spacing Second Dimension"/>
  // <entry group="0018" element="9322" vr="FD" vm="2" name="Reconstruction Pixel Spacing"/>
  // <entry group="0018" element="9404" vr="FL" vm="2" name="Object Pixel Spacing in Center of Beam"/>
  // <entry group="0040" element="08d8" vr="SQ" vm="1" name="Pixel Spacing Sequence"/>
  // <entry group="0070" element="0101" vr="DS" vm="2" name="Presentation Pixel Spacing"/>
  // <entry group="2010" element="0376" vr="DS" vm="2" name="Printer Pixel Spacing"/>
  // <entry group="300a" element="00e9" vr="DS" vm="2" name="Compensator Pixel Spacing"/>

  typedef enum {
    DETECTOR = 0, // (0018,1164) Imager Pixel Spacing
    MAGNIFIED,    // (0018,1114) (IHE Mammo)
    CALIBRATED,   // (0028,0030) Pixel Spacing -> (0028,0a04) Pixel Spacing Calibration Description
    UNKNOWN
  } SpacingType;

  static Attribute<0x28,0x34> ComputePixelAspectRatioFromPixelSpacing(const Attribute<0x28,0x30>& pixelspacing);
};
} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMSPACING_H
