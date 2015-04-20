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
Usage:

 python SortImage.py dirname
"""

import gdcm
import sys

def PrintProgress(object, event):
  assert event == "ProgressEvent"
  print "Progress:", object.GetProgress()

def MySort(ds1, ds2):
  # compare ds1
  return False

if __name__ == "__main__":

  dirname = sys.argv[1]
  d = gdcm.Directory()
  d.Load( dirname )

  print d

  sorter = gdcm.Sorter()
  sorter.SetSortFunction( MySort )
  #sorter.AddObserver( "ProgressEvent", PrintProgress )
  sorter.Sort( d.GetFilenames() )

  print "Sorter:"
  print sorter
