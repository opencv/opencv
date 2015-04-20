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
Hello World !
"""

import gdcm
import sys

if __name__ == "__main__":

  # verbosity:
  #gdcm.Trace.DebugOn()
  #gdcm.Trace.WarningOn()
  #gdcm.Trace.ErrorOn()

  # Get the filename from the command line
  filename = sys.argv[1]

  # Instanciate a gdcm.Reader
  # This is the main class to handle any type of DICOM object
  # You should check for gdcm.ImageReader for reading specifically DICOM Image file
  r = gdcm.Reader()
  r.SetFileName( filename )
  # If the reader fails to read the file, we should stop !
  if not r.Read():
    print "Not a valid DICOM file"
    sys.exit(1)

  # Get the DICOM File structure
  file = r.GetFile()

  # Get the DataSet part of the file
  dataset = file.GetDataSet()

  # Ok let's print it !
  print dataset

  # Use StringFilter to print a particular Tag:
  sf = gdcm.StringFilter()
  sf.SetFile(r.GetFile())

  # Check if Attribute exist
  print dataset.FindDataElement( gdcm.Tag(0x0028,0x0010))

  # Let's print it as string pair:
  print sf.ToStringPair(gdcm.Tag(0x0028,0x0010))
