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

"""
Usage:

 python MergeFile.py input1.dcm input2.dcm

  It will produce a 'merge.dcm' output file, which contains all meta information from input1.dcm
  and copy the Stored Pixel values from input2.dcm
  This script even works when input2.dcm is a Secondary Capture and does not contains information
  such as IOP and IPP...
"""

import sys
import gdcm

if __name__ == "__main__":

  file1 = sys.argv[1]
  file2 = sys.argv[2]

  r1 = gdcm.ImageReader()
  r1.SetFileName( file1 )
  if not r1.Read():
    sys.exit(1)

  r2 = gdcm.ImageReader()
  r2.SetFileName( file2 )
  if not r2.Read():
    sys.exit(1)

  # Image from r2 could be Secondary Capture and thus would not contains neither IPP nor IOP
  # Instead always prefer to only copy the Raw Data Element.
  # Warning ! Image need to be identical ! Only the value of Stored Pixel can be different.
  r1.GetImage().SetDataElement( r2.GetImage().GetDataElement() )

  w = gdcm.ImageWriter()
  w.SetFile( r1.GetFile() )
  #w.SetImage( r2.GetImage() )  # See comment above
  w.SetImage( r1.GetImage() )

  w.SetFileName( "merge.dcm" )
  if not w.Write():
    sys.exit(1)

  sys.exit(0)
