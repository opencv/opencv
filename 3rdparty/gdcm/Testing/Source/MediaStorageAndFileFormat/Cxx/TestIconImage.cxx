/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIconImage.h"

// FIXME:
// gdcmData/US-GE-4AICL142.dcm has a private data element that is an Icon:
/*
(6003,0010) LO [GEMS_Ultrasound_ImageGroup_001]         #  30, 1 PrivateCreator
(6003,1010) SQ (Sequence with explicit length #=1)      # 12522, 1 Unknown Tag & Data
  (fffe,e000) na (Item with explicit length #=15)         # 12514, 1 Item
    (0002,0010) UI =LittleEndianExplicit                    #  20, 1 TransferSyntaxUID
    (0008,0008) CS [DERIVED\SECONDARY]                      #  18, 2 ImageType
    (0008,2111) ST [SmallPreview]                           #  12, 1 DerivationDescription
    (0028,0002) US 3                                        #   2, 1 SamplesPerPixel
    (0028,0004) CS [RGB]                                    #   4, 1 PhotometricInterpretation
    (0028,0006) US 0                                        #   2, 1 PlanarConfiguration
    (0028,0010) US 64                                       #   2, 1 Rows
    (0028,0011) US 64                                       #   2, 1 Columns
    (0028,0014) US 1                                        #   2, 1 UltrasoundColorDataPresent
    (0028,0100) US 8                                        #   2, 1 BitsAllocated
    (0028,0101) US 8                                        #   2, 1 BitsStored
    (0028,0102) US 7                                        #   2, 1 HighBit
    (0028,0103) US 0                                        #   2, 1 PixelRepresentation
    (6003,0010) LO [GEMS_Ultrasound_ImageGroup_001]         #  30, 1 PrivateCreator
    (6003,1011) OB 00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00... # 12288, 1 Unknown Tag & Data
  (fffe,e00d) na (ItemDelimitationItem for re-encoding)   #   0, 0 ItemDelimitationItem
*/
int TestIconImage(int, char *[])
{
  gdcm::IconImage icon;
  return 0;
}
