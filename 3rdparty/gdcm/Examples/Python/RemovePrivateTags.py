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

 python RemovePrivateTags.py input.dcm output.dcm
"""

import sys
import gdcm


if __name__ == "__main__":

  file1 = sys.argv[1]
  file2 = sys.argv[2]

  # Instanciate the reader.
  r = gdcm.Reader()
  r.SetFileName( file1 )
  if not r.Read():
    sys.exit(1)

  # Remove private tags
  ano = gdcm.Anonymizer()
  ano.SetFile( r.GetFile() )
  if not ano.RemovePrivateTags():
    sys.exit(1)

  # Write DICOM file
  w = gdcm.Writer()
  w.SetFile( ano.GetFile() )
  #w.CheckFileMetaInformationOff() # Do not attempt to check meta header
  w.SetFileName( file2 )
  if not w.Write():
    sys.exit(1)

  # It is usually a good idea to exit the script with an error, as gdcm does not remove partial (incorrect) DICOM file
  # (application level)
