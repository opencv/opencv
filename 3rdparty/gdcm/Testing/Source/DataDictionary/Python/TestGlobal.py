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

  #gi = gdcm.GlobalInstance
  #print gi
  singleton = gdcm.Global.GetInstance()
  print(singleton)
  d = singleton.GetDicts()
  print(d)
  t = gdcm.Tag(0x0010,0x0010)
  entry = d.GetDictEntry( t )
  print(entry)
  print(entry.GetName())
  #print entry.GetVM()
  print(entry.GetVR())

  # Test succeed ?
  sys.exit(0)
