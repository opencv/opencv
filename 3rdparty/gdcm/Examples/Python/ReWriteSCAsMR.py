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
GDCM 1.x would write out MR Image Storage as Secondary Capture Object while still setting Rescale Slope/Intercept
and saving the Pixel Spacing in (0028,0030)
"""

import gdcm
import sys,os

def CheckSecondaryCaptureObjectIsMRImageStorage(r):
  ds = r.GetFile().GetDataSet()
  # Check Source Image Sequence
  if ds.FindDataElement( gdcm.Tag(0x0008,0x2112) ):
    sis = ds.GetDataElement( gdcm.Tag(0x0008,0x2112) )
    sqsis = sis.GetSequenceOfItems()
    if sqsis.GetNumberOfItems():
      item1 = sqsis.GetItem(1)
      nestedds = item1.GetNestedDataSet()
      if nestedds.FindDataElement( gdcm.Tag(0x0008,0x1150) ):
        ReferencedSOPClassUID = nestedds.GetDataElement( gdcm.Tag(0x0008,0x1150) )
        raw = ReferencedSOPClassUID.GetByteValue().GetPointer()
        uids = gdcm.UIDs()
        # what is the actual object we are looking at ?
        ms = gdcm.MediaStorage()
        ms.SetFromDataSet(ds)
        msuid = ms.GetString()
        uids.SetFromUID( msuid )
        msuidname = uids.GetName() # real Media Storage Name
        uids.SetFromUID( raw )
        sqmsuidname = uids.GetName() # Source Image Sequence Media Storage Name
        # If object is SC and Source derivation is MRImageStorage then we can assume 'Pixel Spacing' is correct
        if( sqmsuidname == 'MR Image Storage' and msuidname == 'Secondary Capture Image Storage' ):
          return True
  # in all other case simply return the currentspacing:
  return False

if __name__ == "__main__":
  r = gdcm.ImageReader()
  filename = sys.argv[1]
  r.SetFileName( filename )
  if not r.Read():
    sys.exit(1)
  f = r.GetFile()

  if( CheckSecondaryCaptureObjectIsMRImageStorage(r) ):
    # Special handling of the spacing:
    # GDCM 1.2.0 would not rewrite correcly DICOM Object and would always set them as 'Secondary Capture Image Storage'
    # while we would rather have 'MR Image Storage'
    gdcm.ImageHelper.SetForcePixelSpacing( True )
    mrspacing = gdcm.ImageHelper.GetSpacingValue( r.GetFile() )
    # TODO: I cannot do simply the following:
    #image.SetSpacing( mrspacing )
    image.SetSpacing(0, mrspacing[0] )
    image.SetSpacing(1, mrspacing[1] )
    image.SetSpacing(2, mrspacing[2] )
    gdcm.ImageHelper.SetForceRescaleInterceptSlope( True )
    ris = gdcm.ImageHelper.GetRescaleInterceptSlopeValue( r.GetFile() )
    image.SetIntercept( ris[0] )
    image.SetSlope( ris[1] )

  outfilename = sys.argv[2]
  w = gdcm.ImageWriter()
  w.SetFileName( outfilename )
  w.SetFile( r.GetFile() )
  w.SetImage( image )
  if not w.Write():
    sys.exit(1)

  sys.exit(0)
