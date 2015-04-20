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
import sys,os

class ProgressWatcher(gdcm.SimpleSubjectWatcher):
  def ShowProgress(self, sender, event):
    pe = gdcm.ProgressEvent.Cast(event)
    print pe.GetProgress()
  def EndFilter(self):
    print "Yay ! I am done"

if __name__ == "__main__":
  directory = sys.argv[1]

  # Define the set of tags we are interested in
  t1 = gdcm.Tag(0x8,0x8);
  t2 = gdcm.Tag(0x10,0x10);

  # Iterate over directory
  d = gdcm.Directory();
  nfiles = d.Load( directory );
  if(nfiles == 0): sys.exit(1);
  # System.Console.WriteLine( "Files:\n" + d.toString() );

  filenames = d.GetFilenames()

  #  Get rid of any Warning while parsing the DICOM files
  gdcm.Trace.WarningOff()

  # instanciate Scanner:
  sp = gdcm.Scanner.New();
  s = sp.__ref__()
  w = ProgressWatcher(s, 'Watcher')

  s.AddTag( t1 );
  s.AddTag( t2 );
  b = s.Scan( filenames );
  if(not b): sys.exit(1);

  print "success" ;
  #print s

  pttv = gdcm.PythonTagToValue( s.GetMapping( filenames[1] ) )
  pttv.Start()
  # iterate until the end:
  while( not pttv.IsAtEnd() ):
    # get current value for tag and associated value:
    # if tag was not found, then it was simply not added to the internal std::map
    # Warning value can be None
    tag = pttv.GetCurrentTag()
    value = pttv.GetCurrentValue()
    print tag,"->",value
    # increment iterator
    pttv.Next()

  sys.exit(0)
