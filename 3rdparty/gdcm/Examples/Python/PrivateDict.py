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
"""

import gdcm
import sys,os

if __name__ == "__main__":
  #gdcm.Trace.DebugOn()
  globInst = gdcm.Global.GetInstance()
  # Try to load Part3.xml file
  # This fils is too big for being accessible directly at runtime.
  globInst.LoadResourcesFiles()


  # Get a private tag from the runtime dicts. LoadResourcesFiles could
  # have failed but this has no impact on the private dict

  d = globInst.GetDicts()
  print d.GetDictEntry( gdcm.Tag(0x0029,0x0010) ,"SIEMENS CSA HEADER" )
  pd = d.GetPrivateDict()
  print pd.GetDictEntry( gdcm.PrivateTag(0x0029,0x0010,"SIEMENS CSA HEADER") )
