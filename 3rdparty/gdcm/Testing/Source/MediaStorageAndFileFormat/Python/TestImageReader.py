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

def TestImageRead(filename, verbose = False):
  r = gdcm.ImageReader()
  r.SetFileName( filename )
  success = r.Read()
  if verbose: print(r.GetImage())
  return success

if __name__ == "__main__":
  sucess = 0
  try:
    filename = os.sys.argv[1]
    sucess += TestImageRead( filename, True )
  except:
    # loop over all files:
    gdcm.Trace.DebugOff()
    gdcm.Trace.WarningOff()
    t = gdcm.Testing()
    nfiles = t.GetNumberOfFileNames()
    for i in range(0,nfiles):
      #print t.GetFileName(i)
      filename = t.GetFileName(i)
      sucess += TestImageRead( filename )

  # Test succeed ?
  sys.exit(sucess == 0)
