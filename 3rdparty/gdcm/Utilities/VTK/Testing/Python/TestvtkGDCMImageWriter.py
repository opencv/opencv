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

import vtkgdcm
import os,sys

def PrintProgress(object, event):
  assert event == "ProgressEvent"
  print("Progress:", object.GetProgress())

if __name__ == "__main__":
  try:
    filename = os.sys.argv[1]
  except:
    # failure
    print("Need a filename")
    sys.exit(1)

  # setup reader
  r = vtkgdcm.vtkGDCMImageReader()

  r.SetFileName( filename )
  r.AddObserver("ProgressEvent", PrintProgress)
  r.Update()
  print(r.GetOutput())
  # Write output
  writer = vtkgdcm.vtkGDCMImageWriter()
  writer.SetInput( r.GetOutput() )
  writer.SetMedicalImageProperties( r.GetMedicalImageProperties() )
  writer.SetFileName( "TestvtkGDCMImageWriterPython.dcm" )
  writer.Write()

  # Test succeed ?
  #sys.exit(sucess != 1)
