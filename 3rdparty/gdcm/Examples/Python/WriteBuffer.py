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

http://chuckhahm.com/Ischem/Zurich/XX_0134

(2005,1132) SQ (Sequence with undefined length #=8)     # u/l, 1 Unknown Tag & Data
  (fffe,e000) na (Item with undefined length #=9)         # u/l, 1 Item
    (2005,0011) LO [Philips MR Imaging DD 002]              #  26, 1 PrivateCreator
    (2005,1137) PN [PDF_CONTROL_GEN_PARS]                   #  20, 1 Unknown Tag & Data
    (2005,1138) PN (no value available)                     #   0, 0 Unknown Tag & Data
    (2005,1139) PN [IEEE_PDF]                               #   8, 1 Unknown Tag & Data
    (2005,1140) PN (no value available)                     #   0, 0 Unknown Tag & Data
    (2005,1141) PN (no value available)                     #   0, 0 Unknown Tag & Data
    (2005,1143) SL 3103                                     #   4, 1 Unknown Tag & Data
    (2005,1144) OW 0566\0000\013b\0000\0a4a\0000\000e\0000\0a7a\0000\0195\0000\0008... # 3104, 1 Unknown Tag & Data
    (2005,1147) CS [Y]                                      #   2, 1 Unknown Tag & Data
  (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
  (fffe,e000) na (Item with undefined length #=9)         # u/l, 1 Item
    (2005,0011) LO [Philips MR Imaging DD 002]              #  26, 1 PrivateCreator
    (2005,1137) PN [PDF_CONTROL_PREP_PARS]                  #  22, 1 Unknown Tag & Data
    (2005,1138) PN (no value available)                     #   0, 0 Unknown Tag & Data
    (2005,1139) PN [IEEE_PDF]                               #   8, 1 Unknown Tag & Data
    (2005,1140) PN (no value available)                     #   0, 0 Unknown Tag & Data
    (2005,1141) PN (no value available)                     #   0, 0 Unknown Tag & Data
    (2005,1143) SL 7934                                     #   4, 1 Unknown Tag & Data
    (2005,1144) OW 19b6\0000\005f\0000\1b2a\0000\00f3\0000\1eee\0000\0000\0000\0008... # 7934, 1 Unknown Tag & Data
    (2005,1147) CS [Y]                                      #   2, 1 Unknown Tag & Data
  (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
...
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

  fg = gdcm.FilenameGenerator()
  f = r.GetFile()
  ds = f.GetDataSet()
  tsis = gdcm.Tag(0x2005,0x1132) #
  if ds.FindDataElement( tsis ):
    sis = ds.GetDataElement( tsis )
    #sqsis = sis.GetSequenceOfItems()
    # GetValueAsSQ handle more cases
    sqsis = sis.GetValueAsSQ()
    if sqsis.GetNumberOfItems():
      nitems = sqsis.GetNumberOfItems();
      fg.SetNumberOfFilenames( nitems )
      fg.SetPrefix( file2 )
      if not fg.Generate():
        print "problem"
        sys.exit(1)
      for i in range(0,nitems):
        item1 = sqsis.GetItem(i+1) # Item start at 1
        nestedds = item1.GetNestedDataSet()
        tprcs = gdcm.Tag(0x2005,0x1144) #
        if nestedds.FindDataElement( tprcs ):
          prcs = nestedds.GetDataElement( tprcs )
          bv = prcs.GetByteValue()
          print bv
          f = open( fg.GetFilename(i) , "w" )
          f.write( bv.WriteBuffer() )
