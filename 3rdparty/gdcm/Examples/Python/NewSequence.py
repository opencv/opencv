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

  python NewSequence.py input.dcm output.dcm


Thanks to Robert Irie for code
"""

import sys
import gdcm

if __name__ == "__main__":

  file1 = sys.argv[1]
  file2 = sys.argv[2]

  r = gdcm.Reader()
  r.SetFileName( file1 )
  if not r.Read():
    sys.exit(1)

  f = r.GetFile()
  ds = f.GetDataSet()
  #tsis = gdcm.Tag(0x0008,0x2112) # SourceImageSequence

  # Create a dataelement
  de = gdcm.DataElement(gdcm.Tag(0x0010, 0x2180))
  de.SetByteValue("Occupation", gdcm.VL(len("Occupation")))
  de.SetVR(gdcm.VR(gdcm.VR.SH))

  # Create an item
  it=gdcm.Item()
  it.SetVLToUndefined()      # Needed to not popup error message
  #it.InsertDataElement(de)
  nds=it.GetNestedDataSet()
  nds.Insert(de)

  # Create a Sequence
  sq=gdcm.SequenceOfItems().New()
  sq.SetLengthToUndefined()
  sq.AddItem(it)

  # Insert sequence into data set
  des=gdcm.DataElement(gdcm.Tag(0x0400,0x0550))
  des.SetVR(gdcm.VR(gdcm.VR.SQ))
  des.SetValue(sq.__ref__())
  des.SetVLToUndefined()

  ds.Insert(des)

  w = gdcm.Writer()
  w.SetFile( f )
  w.SetFileName( file2 )
  if not w.Write():
    sys.exit(1)
