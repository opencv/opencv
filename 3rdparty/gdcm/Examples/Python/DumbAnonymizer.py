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
This example shows how one can use the gdcm.Anonymizer in 'dumb' mode.
This class becomes really handy when one knows which particular tag to fill in.

Usage:

 python DumbAnonymizer.py gdcmData/012345.002.050.dcm out.dcm

"""

import gdcm

# http://www.oid-info.com/get/1.3.6.1.4.17434
THERALYS_ORG_ROOT = "1.3.6.1.4.17434"

tag_rules={
  # Value
  (0x0012,0x0010):("Value","MySponsorName"),
  (0x0012,0x0020):("Value","MyProtocolID"),
  (0x0012,0x0021):("Value","MyProtocolName"),
  (0x0012,0x0062):("Value","YES"),
  (0x0012,0x0063):("Value","MyDeidentificationMethod"),

  # Method
  #(0x0002,0x0003):("Method","GenerateMSOPId"),
  #(0x0008,0x1155):("Method","GenerateMSOPId"),
  (0x0008,0x0018):("Method","GenerateMSOPId"),
  (0x0010,0x0010):("Method","GetSponsorInitials"),
  (0x0010,0x0020):("Method","GetSponsorId"),
  (0x0012,0x0030):("Method","GetSiteId"),
  (0x0012,0x0031):("Method","GetSiteName"),
  (0x0012,0x0040):("Method","GetSponsorId"),
  (0x0012,0x0050):("Method","GetTPId"),
  (0x0018,0x0022):("Method","KeepIfExist"),
  (0x0018,0x1315):("Method","KeepIfExist"),
  (0x0020,0x000d):("Method","GenerateStudyId"),
  (0x0020,0x000e):("Method","GenerateSeriesId"),
  (0x0020,0x1002):("Method","GetNumberOfFrames"),
  (0x0020,0x0020):("Method","GetPatientOrientation"),
  # Other:
  (0x0012,0x0051):("Patient Field","Type Examen"),
  (0x0018,0x1250):("Sequence Field","Receive Coil"),
  (0x0018,0x0088):("Sequence Field","Spacing Between Slice"),
  (0x0018,0x0095):("Sequence Field","Pixel Bandwidth"),
  (0x0018,0x0082):("Sequence Field","Invertion Time"),
}

class MyAnon:
  def __init__(self):
    self.studyuid = None
    self.seriesuid = None
    generator = gdcm.UIDGenerator()
    if not self.studyuid:
      self.studyuid = generator.Generate()
    if not self.seriesuid:
      self.seriesuid = generator.Generate()
  def GetSponsorInitials(self):
    return "dummy^foobar"
  def GenerateStudyId(self):
    return self.studyuid
  def GenerateSeriesId(self):
    return self.seriesuid
  #def GenerateMSOPId(self):
  def GenerateMSOPId(self):
    generator = gdcm.UIDGenerator()
    return generator.Generate()
  def GetSiteId(self):
    return "MySiteId"
  def GetSiteName(self):
    return "MySiteName"
  def GetSponsorId(self):
    return "MySponsorId"
  def GetTPId(self):
    return "MyTP"

if __name__ == "__main__":
  import sys
  gdcm.FileMetaInformation.SetSourceApplicationEntityTitle( "DumbAnonymizer" )
  gdcm.UIDGenerator.SetRoot( THERALYS_ORG_ROOT )

  r = gdcm.Reader()
  filename = sys.argv[1]
  r.SetFileName( filename )
  if not r.Read(): sys.exit(1)

  obj = MyAnon()

  w = gdcm.Writer()
  ano = gdcm.Anonymizer()
  ano.SetFile( r.GetFile() )
  ano.RemoveGroupLength()
  for tag,rule in tag_rules.items():
    if rule[0] == 'Value':
      print tag,rule
      ano.Replace( gdcm.Tag( tag[0], tag[1] ), rule[1] )
    elif rule[0] == 'Method':
      print tag,rule
      # result = locals()[rule[1]]()
      methodname = rule[1]
      if hasattr(obj, methodname):
        _member = getattr(obj, methodname)
        result = _member()
        ano.Replace( gdcm.Tag( tag[0], tag[1] ), result )
      else:
        print "Problem with: ", methodname

  outfilename = sys.argv[2]
  w.SetFileName( outfilename )
  w.SetFile( ano.GetFile() )
  if not w.Write(): sys.exit(1)
