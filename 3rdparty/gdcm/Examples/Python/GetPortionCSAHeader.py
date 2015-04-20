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

 python GetPortionCSAHeader.py input.dcm

Footnote:
  SIEMENS is not publishing any information on the CSA header. So any info extracted
  is at your own risk.
"""

import sys
import gdcm

if __name__ == "__main__":

  file = sys.argv[1]

  r = gdcm.Reader()
  r.SetFileName( file )
  if not r.Read():
    sys.exit(1)

  ds = r.GetFile().GetDataSet()
  csa_t1 = gdcm.CSAHeader()
  csa_t2 = gdcm.CSAHeader()
  #print csa
  t1 = csa_t1.GetCSAImageHeaderInfoTag();
  print t1
  t2 = csa_t2.GetCSASeriesHeaderInfoTag();
  print t2
  # Let's do it for t1:
  if ds.FindDataElement( t1 ):
    csa_t1.LoadFromDataElement( ds.GetDataElement( t1 ) )
    print csa_t1

  # Now let's pretend we are only interested in B_value and DiffusionGradientDirection entries:
  bvalues = csa_t1.GetCSAElementByName( "B_value" ) # WARNING: it is case sensitive !
  print bvalues

  diffgraddir = csa_t1.GetCSAElementByName( "DiffusionGradientDirection" ) # WARNING: it is case sensitive !
  print diffgraddir

  # repeat for t2 if you like it:
  if ds.FindDataElement( t2 ):
    csa_t2.LoadFromDataElement( ds.GetDataElement( t2 ) )
    # print csa_t2

  gdt = csa_t2.GetCSAElementByName( "GradientDelayTime" )
  print gdt

  bv = gdt.GetByteValue();
  #print bv
  str = bv.GetPointer()
  print str.split("\\")
