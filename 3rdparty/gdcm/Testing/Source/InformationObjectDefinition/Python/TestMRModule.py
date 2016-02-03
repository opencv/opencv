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

import gdcm
import os,sys

g = gdcm.Global.GetInstance()
g.LoadResourcesFiles()
defs = g.GetDefs()
modules = defs.GetModules()
macros = defs.GetMacros()
#module = modules.GetModule( "MR Image Module Attributes" )
module = modules.GetModule( "C.8.3.1" )
#print dir(module)

mentry = module.GetModuleEntryInMacros( macros, gdcm.Tag(0x0018,0x0087) )
#print dir(mentry)
print(mentry)
mentry = module.GetModuleEntryInMacros( macros, gdcm.Tag(0x0018,0x1080) )
print(mentry)

# Test succeed ?
sys.exit(0)
