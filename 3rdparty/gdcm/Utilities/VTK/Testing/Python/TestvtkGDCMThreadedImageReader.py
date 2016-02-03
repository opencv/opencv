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

# This is important for now to load the vtkgdcm module first
import vtkgdcm
from vtk import vtkImageGradient
from vtk import vtkMultiThreader
from vtk import vtkDirectory
from vtk import vtkStructuredPointsWriter
from vtk.util import vtkConstants as vtkType
from vtkgdcm import vtkStringArray
import gdcm # for gdcm.Directory
import os,sys

def PrintProgress(object, event):
  assert event == "ProgressEvent"
  print "Progress:", object.GetProgress()

# Helper function to extract image dimension and type
# this info could also be coming from a database for example instead of read from a particular file
def ExecuteInformation(reader, filename, dimz = 1):
  import gdcm
  reffile = filename # filenames.GetValue(0) # Take first image as reference
  #print reader
  r = gdcm.ImageReader()
  r.SetFileName( reffile )
  sucess = r.Read()
  assert sucess
  #print r.GetImage().Print()
  image = r.GetImage()
  assert image.GetNumberOfDimensions() == 2 or image.GetNumberOfDimensions() == 3
  dims = [0,0,0]
  dims[0] = image.GetDimension(0)
  dims[1] = image.GetDimension(1)
  dims[2] = dimz # filenames.GetNumberOfValues()
  #print dims
  #print image.GetPixelFormat().GetTPixelFormat()
  pixelformat = image.GetPixelFormat().GetScalarType()
  datascalartype = vtkType.VTK_VOID # dummy should not happen
  if pixelformat == gdcm.PixelFormat.INT8:
    datascalartype = vtkType.VTK_SIGNED_CHAR
  elif pixelformat == gdcm.PixelFormat.UINT8:
    datascalartype = vtkType.VTK_UNSIGNED_CHAR
  elif pixelformat == gdcm.PixelFormat.INT16:
    datascalartype = vtkType.VTK_SHORT
  elif pixelformat == gdcm.PixelFormat.UINT16:
    datascalartype = vtkType.VTK_UNSIGNED_SHORT
  else:
    print "Unhandled PixelFormat: ", pixelformat
    sys.exit(1)
  #print datascalartype
  numberOfScalarComponents = image.GetPixelFormat().GetSamplesPerPixel()
  #print numberOfScalarComponents
  #print gdcm.PhotometricInterpretation.GetPIString( image.GetPhotometricInterpretation().PIType() )
  #reader.SetDataExtent( dataextent );
  reader.SetDataExtent( 0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1 )
  reader.SetDataScalarType ( datascalartype )
  reader.SetNumberOfScalarComponents( numberOfScalarComponents )

if __name__ == "__main__":
  try:
    filename = os.sys.argv[1]
  except:
    # failure
    print "Need a filename"
    sys.exit(1)

  # setup reader
  r = vtkgdcm.vtkGDCMThreadedImageReader()
  r.FileLowerLeftOn()
  #dir = vtkDirectory()
  dir = gdcm.Directory()

  # Did user pass in a directory:
  system = gdcm.System()
  if system.FileIsDirectory( filename ):
    nfiles = dir.Load( filename )
    files = dir.GetFilenames()
    # Need to construct full path out of the simple filename
    fullpath = vtkStringArray()
    for file in files:
      fullpath.InsertNextValue( file )
    r.SetFileNames( fullpath )
    assert fullpath.GetNumberOfValues() # Need at least one file
    ExecuteInformation(r, fullpath.GetValue(0), fullpath.GetNumberOfValues() )
    r.AddObserver("ProgressEvent", PrintProgress)
    r.Update()
    #print r.GetOutput()
    #print vtkMultiThreader.GetGlobalDefaultNumberOfThreads()
    #g = vtkImageGradient()
    #g.SetInput( r.GetOutput() )
    #g.AddObserver("ProgressEvent", PrintProgress)
    #g.Update()
    # Write output
    writer = vtkStructuredPointsWriter()
    writer.SetInput( r.GetOutput() )
    writer.SetFileName( "TestvtkGDCMThreadedImageReaderPython.vtk" )
    writer.SetFileTypeToBinary()
    #writer.Write()
  else:
    # TODO
    r.SetFileName( filename )
    ExecuteInformation(r, filename )
    r.Update()
    print r.GetOutput()
    #sys.exit(1)


  # Test succeed ?
  #sys.exit(sucess != 1)
