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

# This used to be important to load the vtkgdcm module first
# (VTK 5.2 contains the proper patch now)
import vtkgdcm
from vtkgdcm import vtkStringArray
from vtk.util import vtkConstants as vtkType
import gdcm # for gdcm.Directory
import os,sys

def PrintProgress(object, event):
  assert event == "ProgressEvent"
  print("Progress:", object.GetProgress())

if __name__ == "__main__":
  root = gdcm.Testing.GetDataExtraRoot()
  dirname = os.path.join(root, "gdcmSampleData/ForSeriesTesting/Perfusion/images" )
  print(dirname)

  # setup reader
  r = vtkgdcm.vtkGDCMThreadedImageReader2()
  dir = gdcm.Directory()

  # Did user pass in a directory:
  system = gdcm.System()
  if system.FileIsDirectory( dirname ):
    nfiles = dir.Load( dirname )
    files = dir.GetFilenames()
    # Need to construct full path out of the simple filename
    fullpath = vtkStringArray()
    for file in files:
      fullpath.InsertNextValue( file )
    r.SetFileNames( fullpath )
    assert fullpath.GetNumberOfValues() # Need at least one file
    # Now specify the property of the image:
    """
    Note (MM), I verified that even if there are multiple Series in this Study they are all compatible and thus
    can be loaded as a fake 3D (VTK) volume, only origin is changing & shift/scale .
    See:
    $ for i in `ls ForSeriesTesting/Perfusion/images/1.*`; do gdcminfo $i; done | sort | uniq
    """
    """
    gdcminfo ForSeriesTesting/Perfusion/images/1.3.46.670589.5.2.14.2198403904.1100092395.157798.dcm
    MediaStorage is 1.2.840.10008.5.1.4.1.1.4 [MR Image Storage]
    NumberOfDimensions: 2
    Dimensions: (256,256)
    Origin: (-115,-125.969,-17.068)
    Spacing: (0.898438,0.898438,5.5)
    DirectionCosines: (1,0,0,0,0.949631,-0.31337)
    Rescale Intercept/Slope: (-1985.36,1)
    SamplesPerPixel    :1
    BitsAllocated      :16
    BitsStored         :12
    HighBit            :11
    PixelRepresentation:0
    Orientation Label: AXIAL
    """
    dims = [0,0,0]
    dims[0] = 256
    dims[1] = 256
    dims[2] = nfiles
    # Even if Stored Pixel is UINT16, the World Value Pixel is Float
    datascalartype = vtkType.VTK_FLOAT
    spacing = [0.898438,0.898438,5.5]
    origin = [-115,-125.969,-17.068]
    intercept_slope = [-1985.36,1]
    numberOfScalarComponents = 1
    r.SetDataExtent( 0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1 )
    r.SetDataScalarType ( datascalartype )
    r.SetNumberOfScalarComponents( numberOfScalarComponents )
    r.SetDataOrigin( origin )
    r.SetDataSpacing( spacing )
    # Useless only for backward compatibily, the real shift/scale will be read from files:
    #r.SetShift( intercept_slope[0] )
    #r.SetScale( intercept_slope[1] )

    # Setup the ProgressEvent
    r.AddObserver("ProgressEvent", PrintProgress)
    r.Update()
    print(r.GetOutput())

  # Test succeed ?
  #sys.exit(sucess != 1)
