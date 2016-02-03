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

fns = vtk.vtkStringArray()
fns.InsertNextValue( VTK_DATA_ROOT + "/Data/mr.001" )
fns.InsertNextValue( VTK_DATA_ROOT + "/Data/mr.001" )
fns.InsertNextValue( VTK_DATA_ROOT + "/Data/mr.001" )

r = vtkgdcm.vtkGDCMImageReader()
r.SetFileNames( fns )
r.Update()

print r.GetOutput()

# try to rewrite it:
w = vtkgdcm.vtkGDCMImageWriter()
w.SetFileDimensionality( 3 )
w.SetInput( r.GetOutput() )
w.SetMedicalImageProperties( r.GetMedicalImageProperties() )
w.SetDirectionCosines( r.GetDirectionCosines() )
w.SetFileName( "mr3.001.dcm" )
w.Write()
