############################################################################
#
#  Program: GDCM (Grassroots DICOM). A DICOM library
#
#  Copyright (c) 2006-2011 Mathieu Malaterre
#  All rights reserved.
#  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.
#
#     This software is distributed WITHOUT ANY WARRANTY; without even
#     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#     PURPOSE.  See the above copyright notice for more information.
#
############################################################################

import gdcm
import os,sys

if __name__ == "__main__":
  o1 = gdcm.Tag(0x0000,0x0000)
  o2 = gdcm.Tag(0x0010,0x0000)
  o3 = gdcm.Tag(0x0000,0x0010)
  o4 = gdcm.Tag(0x0010,0x0010)

  if o1 == o2:
    sys.exit(1)
  if o1 == o3:
    sys.exit(1)
  if o1 == o4:
    sys.exit(1)

  if o1 != o1:
    sys.exit(1)

  if o1 > o2:
    print("Fail o1 > o2")
    sys.exit(1)

  sys.exit(0)
