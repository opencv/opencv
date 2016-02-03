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

# Simply check that we can read mr.001 from VTKData

import vtkgdcm
import vtk

from vtk.util.misc import vtkGetDataRoot
VTK_DATA_ROOT = vtkGetDataRoot()

r = vtkgdcm.vtkGDCMImageReader()
r.SetFileName( VTK_DATA_ROOT + "/Data/mr.001" )
r.Update()

#print r.GetOutput()

# Pixel Spacing
# 0.78125, 0.78125, 0

# Image Position (Patient)
# -13.3034, -80.8219, 119.178

# try to rewrite it:
w = vtkgdcm.vtkGDCMImageWriter()
w.SetInput( r.GetOutput() )
w.SetMedicalImageProperties( r.GetMedicalImageProperties() )
w.SetDirectionCosines( r.GetDirectionCosines() )
w.SetFileName( "mr.001.dcm" )
w.Write()

# beach.tif
#tiffreader = vtk.vtkTIFFReader()
#tiffreader.SetFileName( VTK_DATA_ROOT + "/Data/beach.tif" )
#tiffreader.Update()
# print tiffreader.GetOutput()
# -> TIFF reader was apparently broken in VTK until some very recent
# version and thus image appear upside down, unless you also update VTKData :(

jpegreader = vtk.vtkJPEGReader()
jpegreader.SetFileName( VTK_DATA_ROOT + "/Data/beach.jpg" )
#jpegreader.Update()

# Need a new writer otherwise MedicalImageProperties are re-used...
w2 = vtkgdcm.vtkGDCMImageWriter()
#w2.SetInput( tiffreader.GetOutput() )
w2.SetInput( jpegreader.GetOutput() )
w2.SetFileName( "beach.dcm" )
w2.Write()
