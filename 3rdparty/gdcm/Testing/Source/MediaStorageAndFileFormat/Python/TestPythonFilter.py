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

def TestPythonFilter(filename, verbose = False):
  r = gdcm.Reader()
  r.SetFileName( filename )
  sucess = r.Read()
  if( not sucess ): return 1

  file = r.GetFile()
  ds = file.GetDataSet()
  # Change gdcm struct into something swig can digest:
  pds = gdcm.PythonDataSet(ds)
  sf = gdcm.PythonFilter()
  pds.Start() # Make iterator go at begining
  dic1={}
  dic2={}
  sf.SetFile(file) # extremely important
  while(not pds.IsAtEnd() ):
    t = str(pds.GetCurrent().GetTag())
    print(t)
    res = sf.ToPyObject( pds.GetCurrent().GetTag() )
    dic2[t] = res[1]
    dic1[res[0]] = res[1]
    pds.Next()
  #print dic1
  #print dic2
  try:
    print("Pixel Representation=",dic2[ '(0028,0103)' ])
  except KeyError:
    print("Tag not found in dataset")
  return 0

if __name__ == "__main__":
  sucess = 0
  try:
    filename = os.sys.argv[1]
    sucess += TestPythonFilter( filename, True )
  except:
    # loop over all files:
    gdcm.Trace.WarningOff()
    t = gdcm.Testing()
    nfiles = t.GetNumberOfFileNames()
    for i in range(0,nfiles):
      filename = t.GetFileName(i)
      sucess += TestPythonFilter( filename )


  # Test succeed ?
  sys.exit(sucess == 0)
