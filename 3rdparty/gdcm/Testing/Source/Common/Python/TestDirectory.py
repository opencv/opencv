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

dir = gdcm.Directory()
t = gdcm.Testing()
dataroot = t.GetDataRoot()

system = gdcm.System()
if not system.FileIsDirectory(dataroot):
  sys.exit(1)

nfiles = dir.Load(dataroot)

if nfiles == 0:
  sys.exit(1)

#print dir.GetFilenames()

for file in dir.GetFilenames():
  print(file)

# Test succeed ?
#sys.exit(sucess != 1)
