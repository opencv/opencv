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

def TestModifyFields(filename):
  outfilename = filename + ".rewrite"
  r = gdcm.Reader()
  r.SetFileName( filename )
  sucess = r.Read()
  #print r.GetFile().GetDataSet()

  ds = r.GetFile().GetDataSet()
  #print dir(ds)
  # eg, let's remove a tag
  removetag = gdcm.Tag(0x0043,0x106f)
  if ds.FindDataElement( removetag ):
    ds.Remove( removetag )

  # let's replace a value:
  replacetag = gdcm.Tag(0x0010,0x0010)
  if ds.FindDataElement( replacetag ):
    de = ds.GetDataElement( replacetag )
    #print dir(de)
    patname = "This^is^an^example"
    vl = gdcm.VL( len(patname) )
    de.SetByteValue( patname, vl )

  # let's insert a new dataelement
  # <entry group="0012" element="0062" vr="CS" vm="1" name="Patient Identity Removed"/>
  pir = gdcm.DataElement( gdcm.Tag(0x0012,0x0062) )
  pir.SetVR( gdcm.VR( gdcm.VR.CS ) ) # specify the VR explicitely
  yes = "YES"
  pir.SetByteValue( yes, gdcm.VL(len(yes)) )
  ds.Insert( pir )

  # try again but pretend we don't know the VR
  # <entry group="0012" element="0063" vr="LO" vm="1-n" name="De-identification Method"/>
  deidmethod = gdcm.Tag(0x0012,0x0063)
  # retrieve the supreme global instance, sum of all knowledge in da whole universe:
  dicomdicts = gdcm.GlobalInstance.GetDicts()
  dictel = dicomdicts.GetDictEntry( deidmethod )
  #print dictel.GetVR()
  deid = gdcm.DataElement( deidmethod )
  deid.SetVR( dictel.GetVR() )
  methodstr = "Well known Company"
  #deid.SetByteValue( methodstr, gdcm.VL(len(methodstr)) )
  deid.SetByteValue( methodstr, gdcm.VL(len(methodstr)) )
  ds.Insert( deid )

  #w = gdcm.Writer()
  #w.SetFileName( outfilename )
  #w.SetFile( r.GetFile() )
  #sucess = w.Write()
  return sucess

if __name__ == "__main__":
  sucess = 0
  try:
    filename = os.sys.argv[1]
    sucess += TestModifyFields( filename, True )
  except:
    # loop over all files:
    t = gdcm.Testing()
    nfiles = t.GetNumberOfFileNames()
    for i in range(0,nfiles):
      filename = t.GetFileName(i)
      sucess += TestModifyFields( filename )


  # Test succeed ?
  sys.exit(sucess == 0)
