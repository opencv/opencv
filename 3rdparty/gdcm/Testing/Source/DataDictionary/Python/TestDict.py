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

  singleton = gdcm.Global.GetInstance()
  dicts = singleton.GetDicts()
  d = dicts.GetPublicDict()
  t = gdcm.Tag(0x0010,0x0010)
  keyword = d.GetKeywordFromTag( t )
  if keyword != 'PatientName':
    sys.exit(1)
  print(keyword)

  p = d.GetDictEntryByKeyword( keyword )
  if p[0].GetKeyword() != keyword:
    sys.exit(1)
  print(p[0].GetKeyword())

  p = d.GetDictEntryByKeyword( 'foobar' )
  undef = gdcm.Tag(0xffff,0xffff)
  if p[1] != undef:
    sys.exit(1)
  print(p[1])

  # Test succeed ?
  sys.exit(0)
