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
import vtk

from vtk.util.misc import vtkGetDataRoot

#print vtkGetDataRoot()
VTK_DATA_ROOT = vtkGetDataRoot()
print(VTK_DATA_ROOT)

v16 = vtk.vtkVolume16Reader()
v16.SetDataDimensions(64, 64)
v16.SetDataByteOrderToLittleEndian()
v16.SetFilePrefix(VTK_DATA_ROOT + "/Data/headsq/quarter")
v16.SetImageRange(1, 93)
v16.SetDataSpacing(3.2, 3.2, 1.5)
#v16.Update()
#
#print v16.GetOutput()

w = vtkgdcm.vtkGDCMImageWriter()
w.SetInput( v16.GetOutput() )
w.SetFileDimensionality( 3 )
w.SetFileName( "sc.dcm" )
w.Write()

# Now pretend this is an MR Image Storage:
# Since image is 3D it should default to the new Enhance MR Image Storage:
med = w.GetMedicalImageProperties()
med.SetModality( "MR" )

w.SetFileName( "mr.dcm" )
w.Write()
