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

t = gdcm.Testing()
nfiles = t.GetNumberOfFileNames()
print(nfiles)
for i in range(0,nfiles):
  print(t.GetFileName(i))

print(t.GetFileName(10000))

print(t.GetDataRoot())

# Test succeed ?
#sys.exit(sucess != 1)
