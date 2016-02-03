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
    <uid value="1.2.840.10008.5.1.4.1.1.66" name="Raw Data Storage" type="SOP Class" part="PS 3.4" retired="false"/>
"""

import gdcm
import sys,os

if __name__ == "__main__":
  r = gdcm.Reader()
  # Will require Testing...
  dataroot = gdcm.Testing.GetDataRoot()
  filename = os.path.join( dataroot, '012345.002.050.dcm' )
  r.SetFileName( filename )
  r.Read()
  f = r.GetFile()
  ds = f.GetDataSet()

  uid = "1.2.840.10008.5.1.4.1.1.66"
#  f = gdcm.File()
#  ds = f.GetDataSet()
  de = gdcm.DataElement( gdcm.Tag(0x0008,0x0016) )
  de.SetByteValue( uid, gdcm.VL(len(uid)) )
  vr = gdcm.VR( gdcm.VR.UI )
  de.SetVR( vr )
  ds.Replace( de )

  ano = gdcm.Anonymizer()
  ano.SetFile( r.GetFile() )
  ano.RemovePrivateTags()
  ano.RemoveGroupLength()
  taglist = [
  gdcm.Tag(0x0008,0x0008),
  gdcm.Tag(0x0008,0x0022),
  gdcm.Tag(0x0008,0x0032),
  gdcm.Tag(0x0008,0x2111),
  gdcm.Tag(0x0008,0x1150),
  gdcm.Tag(0x0008,0x1155),
  gdcm.Tag(0x0008,0x0100),
  gdcm.Tag(0x0008,0x0102),
  gdcm.Tag(0x0008,0x0104),
  gdcm.Tag(0x0040,0xa170),
  gdcm.Tag(0x0008,0x2112),
  gdcm.Tag(0x0008,0x0100),
  gdcm.Tag(0x0008,0x0102),
  gdcm.Tag(0x0008,0x0104),
  gdcm.Tag(0x0008,0x9215),
  gdcm.Tag(0x0018,0x0010),
  gdcm.Tag(0x0018,0x0022),
  gdcm.Tag(0x0018,0x0050),
  gdcm.Tag(0x0018,0x0060),
  gdcm.Tag(0x0018,0x0088),
  gdcm.Tag(0x0018,0x0090),
  gdcm.Tag(0x0018,0x1040),
  gdcm.Tag(0x0018,0x1100),
  gdcm.Tag(0x0018,0x1110),
  gdcm.Tag(0x0018,0x1111),
  gdcm.Tag(0x0018,0x1120),
  gdcm.Tag(0x0018,0x1130),
  gdcm.Tag(0x0018,0x1150),
  gdcm.Tag(0x0018,0x1151),
  gdcm.Tag(0x0018,0x1152),
  gdcm.Tag(0x0018,0x1160),
  gdcm.Tag(0x0018,0x1190),
  gdcm.Tag(0x0018,0x1210),
  gdcm.Tag(0x0020,0x0012),
  gdcm.Tag(0x0020,0x0032),
  gdcm.Tag(0x0020,0x0037),
  gdcm.Tag(0x0020,0x1041),
  gdcm.Tag(0x0020,0x4000),
  gdcm.Tag(0x0028,0x0002),
  gdcm.Tag(0x0028,0x0004),
  gdcm.Tag(0x0028,0x0010),
  gdcm.Tag(0x0028,0x0011),
  gdcm.Tag(0x0028,0x0030),
  gdcm.Tag(0x0028,0x0100),
  gdcm.Tag(0x0028,0x0101),
  gdcm.Tag(0x0028,0x0102),
  gdcm.Tag(0x0028,0x0103),
  gdcm.Tag(0x0028,0x1052),
  gdcm.Tag(0x0028,0x1053),
  gdcm.Tag(0x0028,0x2110),
  gdcm.Tag(0x0028,0x2112),
  gdcm.Tag(0x7fe0,0x0010),
  gdcm.Tag(0x0018,0x0020),
  gdcm.Tag(0x0018,0x0021),
  gdcm.Tag(0x0018,0x0023),
  gdcm.Tag(0x0018,0x0025),
  gdcm.Tag(0x0018,0x0080),
  gdcm.Tag(0x0018,0x0081),
  gdcm.Tag(0x0018,0x0083),
  gdcm.Tag(0x0018,0x0084),
  gdcm.Tag(0x0018,0x0085),
  gdcm.Tag(0x0018,0x0086),
  gdcm.Tag(0x0018,0x0087),
  gdcm.Tag(0x0018,0x0091),
  gdcm.Tag(0x0018,0x0093),
  gdcm.Tag(0x0018,0x0094),
  gdcm.Tag(0x0018,0x0095),
  gdcm.Tag(0x0018,0x1088),
  gdcm.Tag(0x0018,0x1090),
  gdcm.Tag(0x0018,0x1094),
  gdcm.Tag(0x0018,0x1250),
  gdcm.Tag(0x0018,0x1251),
  gdcm.Tag(0x0018,0x1310),
  gdcm.Tag(0x0018,0x1312),
  gdcm.Tag(0x0018,0x1314),
  gdcm.Tag(0x0018,0x1315),
  gdcm.Tag(0x0018,0x1316),
  gdcm.Tag(0x0020,0x0110),
  gdcm.Tag(0x0028,0x0120),
  gdcm.Tag(0x0028,0x1050),
  gdcm.Tag(0x0028,0x1051)
  ]
  for tag in taglist:
    #print tag
    ano.Remove( tag )

  # special handling
  gen = gdcm.UIDGenerator()
  ano.Replace( gdcm.Tag(0x0008,0x9123), gen.Generate() )
  #ano.Empty( gdcm.Tag(0x0040,0x0555) )


#
#  uid = gen.Generate()
#  de.SetTag( gdcm.Tag(0x0008,0x0018) )
#  de.SetByteValue( uid, gdcm.VL(len(uid)) )
#  ds.Insert( de )

  # init FMI now:
  #fmi = f.GetHeader()
  #ts = gdcm.TransferSyntax()
  #print ts
  #fmi.SetDataSetTransferSyntax( ts ) # default
  #print fmi.GetDataSetTransferSyntax()
  #de.SetTag( gdcm.Tag(0x0002,0x0010) )
  #uid = "1.2.840.10008.1.2"
  #de.SetByteValue( uid, gdcm.VL(len(uid)) )
  #fmi.Insert( de )
#  f.SetHeader( r.GetFile().GetHeader() )

  writer = gdcm.Writer()
  writer.SetFile( ano.GetFile() )
  writer.SetFileName( "rawstorage.dcm" );
  writer.Write()
