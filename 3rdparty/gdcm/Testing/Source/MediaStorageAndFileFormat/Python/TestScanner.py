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

def TestScan(dirname, recursive = False):
  # Check the dirname is indeed a directory
  system = gdcm.System()
  if not system.FileIsDirectory(dirname):
    print("Need a directory")
    sys.exit(1)

  # Retrieve all the files within that dir (recursively?)
  d = gdcm.Directory()
  nfiles = d.Load( dirname, recursive )
  print("done retrieving all the",nfiles,"files")

  s = gdcm.Scanner()
  t1 = gdcm.Tag(0x0020,0x000d) # VR::UI
  t2 = gdcm.Tag(0x0020,0x000e) # VR::UI
  t3 = gdcm.Tag(0x0028,0x0011) # VR::US
  # Some fun tags, with dual VR:
  t4 = gdcm.Tag(0x0028,0x0106) # VR::US_SS
  t5 = gdcm.Tag(0x0028,0x0107) # VR::US_SS
  s.AddTag( t1 )
  s.AddTag( t2 )
  s.AddTag( t3 )
  s.AddTag( t4 )
  s.AddTag( t5 )
  b = s.Scan( d.GetFilenames() )
  if not b:
    print("Scanner failed")
    sys.exit(1)

  # Raw Values found:
  values  = s.GetValues()
  print("Values found for all tags are:")
  print(values)

  # get the main super-map :
  mappings = s.GetMappings()

  #file1 = d.GetFilenames()[0];
  #print file1
  #m1 = s.GetMapping( file1 )
  #print m1
  #print dir(m1)

  #for k,v in m1.iteritems():
  #  print "item", k,v


  # For each file get the value for tag t1:
  for f in d.GetFilenames():
    print("Working on:",f)
    mapping = s.GetMapping(f)
    pttv = gdcm.PythonTagToValue(mapping)
    # reset iterator to start position
    pttv.Start()
    # iterate until the end:
    while( not pttv.IsAtEnd() ):
      # get current value for tag and associated value:
      # if tag was not found, then it was simply not added to the internal std::map
      # Warning value can be None
      tag = pttv.GetCurrentTag()
      value = pttv.GetCurrentValue()
      print(tag,"->",value)
      # increment iterator
      pttv.Next()

if __name__ == "__main__":
  try:
    dirname = os.sys.argv[1]
    recursive = True
  except:
    t = gdcm.Testing()
    dirname = t.GetDataRoot()
    recursive = False
  TestScan( dirname, recursive)

  sys.exit(0)
