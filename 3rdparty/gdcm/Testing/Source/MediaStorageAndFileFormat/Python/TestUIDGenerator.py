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
  uid = gdcm.UIDGenerator()
  for i in range(0,100):
    print(uid.Generate())

  MY_ROOT = "1.2.3.4"

  # static function:
  gdcm.UIDGenerator_SetRoot( MY_ROOT )
  for i in range(0,100):
    print(uid.Generate())


  # Test succeed ?
  sys.exit(0)
