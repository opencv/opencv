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
import os,sys,re

"""
You need to have dcmdump/dcmdrle/dcmdjpeg in your PATH
"""

def TestDCMTKMD5( filename, verbose = False ):
  blacklist = [
  # Get rid of DICOMDIR if any:
  'DICOMDIR',
  'DICOMDIR_MR_B_VA12A',
  'DICOMDIR-Philips-EasyVision-4200-Entries',
  'dicomdir_Acusson_WithPrivate_WithSR',
  'dicomdir_Pms_With_heavy_embedded_sequence',
  'dicomdir_Pms_WithVisit_WithPrivate_WithStudyComponents',
  'dicomdir_With_embedded_icons',
  # Unsupported file:
  'MR_Spectroscopy_SIEMENS_OF.dcm',
  'gdcm-CR-DCMTK-16-NonSamplePerPix.dcm', # this is not an image
  'ELSCINT1_PMSCT_RLE1.dcm',
  'SignedShortLosslessBug.dcm',
  'JPEGDefinedLengthSequenceOfFragments.dcm', # dcmtk 3.6.0 gives garbage
  'GE_DLX-8-MONO2-PrivateSyntax.dcm',
  'PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm',
  #'DermaColorLossLess.dcm', # technically I could support this one...
  #'LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm', # idem
  'ALOKA_SSD-8-MONO2-RLE-SQ.dcm'] # this one is not supported by dcmtk 3.5.4
  for f in blacklist:
    if f in filename:
      print("%s is on the black list, giving up"%filename)
      return 0
  #print filename
  #
  #dcmdump_exec = "dcmdump -dc -E +P 2,10 -s " + filename + " 2> /dev/null"
  # I had to remove the -dc for the following file:
  # GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm there is trailing space instead of \0
  dcmdump_exec = "dcmdump -E +P 2,10 -s " + filename + " 2> /dev/null"
  #print dcmdump_exec
  f = os.popen(dcmdump_exec)
  ret = f.read()
  #assert ret == 0
  #print ret
  jpegre = re.compile('^.*JPEGLossless.*$')
  jpegre2 = re.compile('^.*JPEGExtended.*$')
  jpegre3 = re.compile('^.*JPEGBaseline.*$')
  j2kre = re.compile('^.*JPEG2000.*$')
  jplsre = re.compile('^.*JPEGLS.*$')
  rlere = re.compile('^.*RLELossless.*$')
  lexre = re.compile('^.*LittleEndianExplicit.*$')
  leire = re.compile('^.*LittleEndianImplicit.*$')
  beire = re.compile('^.*BigEndianExplicit.*$')
  testing = gdcm.Testing()
  outputdir = testing.GetTempDirectory( "TestDCMTKMD5" )
  gdcm.System.MakeDirectory( outputdir )
  outputfilename = testing.GetTempFilename( filename, "TestDCMTKMD5" )
  executable_output_path = gdcm.GDCM_EXECUTABLE_OUTPUT_PATH
  gdcmraw = executable_output_path + '/gdcmraw -P'

  if not ret:
    #print "empty, problem with:", filename
    return 0
  elif type(ret) != type(''):
    print("problem of type with:", filename)
    return 0
  #print ret
  #print ret.__class__
  elif( jpegre.match( ret ) or jpegre2.match(ret) or jpegre3.match(ret) ):
    #print "jpeg: ",filename
    # +cn : conv-never
    # +px : color by pixel
    dcmdjpeg_exec = "dcmdjpeg +cn +px " + filename + " " + outputfilename
    ret = os.system( dcmdjpeg_exec )
    if ret:
      print("dcmdjpeg failed to decompress file. giving up")
      return 0

    gdcmraw_args = ' -i ' + outputfilename + ' -o ' + outputfilename + ".raw"
    gdcmraw += gdcmraw_args
    #print gdcmraw
    ret = os.system( gdcmraw )
    md5 = gdcm.Testing.ComputeFileMD5( outputfilename + ".raw" )
    ref = gdcm.Testing.GetMD5FromFile(filename)
    #print md5
    retval  = 0
    if ref != md5:
      print("md5 are different: %s should be: %s for file %s"%(md5,ref,filename))
      retval = 1
    #print outputfilename
    return retval
  elif( jplsre.match( ret ) ):
    #print "jpegls: ",filename
    dcmdjpls_exec = "dcmdjpls " + filename + " " + outputfilename
    ret = os.system( dcmdjpls_exec )
    if ret:
      print("failed with: ", dcmdjpls_exec)
      return 1

    gdcmraw_args = ' -i ' + outputfilename + ' -o ' + outputfilename + ".raw"
    gdcmraw += gdcmraw_args
    #print gdcmraw
    ret = os.system( gdcmraw )
    md5 = gdcm.Testing.ComputeFileMD5( outputfilename + ".raw" )
    ref = gdcm.Testing.GetMD5FromFile(filename)
    #print md5
    retval  = 0
    if ref != md5:
      print("md5 are different: %s should be: %s for file %s"%(md5,ref,filename))
      retval = 1
    #print outputfilename
    return retval
  elif( rlere.match( ret ) ):
    #print "rle: ",filename
    dcmdrle_exec = "dcmdrle " + filename + " " + outputfilename
    ret = os.system( dcmdrle_exec )
    if ret:
      print("failed with: ", dcmdrle_exec)
      return 1

    gdcmraw_args = ' -i ' + outputfilename + ' -o ' + outputfilename + ".raw"
    gdcmraw += gdcmraw_args
    #print gdcmraw
    ret = os.system( gdcmraw )
    md5 = gdcm.Testing.ComputeFileMD5( outputfilename + ".raw" )
    ref = gdcm.Testing.GetMD5FromFile(filename)
    #print md5
    retval  = 0
    if ref != md5:
      print("md5 are different: %s should be: %s for file %s"%(md5,ref,filename))
      retval = 1
    #print outputfilename
    return retval
  elif( j2kre.match( ret ) ):
    return 0
  elif( lexre.match( ret ) or leire.match(ret) or beire.match(ret) ):
    #print "rle: ",filename
    #dcmdrle_exec = "dcmdrle " + filename + " " + outputfilename
    #ret = os.system( dcmdrle_exec )

    gdcmraw_args = ' -i ' + filename + ' -o ' + outputfilename + ".raw"
    gdcmraw += gdcmraw_args
    #print gdcmraw
    ret = os.system( gdcmraw )
    if ret:
      print("failed with: ", gdcmraw)
      return 1
    md5 = gdcm.Testing.ComputeFileMD5( outputfilename + ".raw" )
    ref = gdcm.Testing.GetMD5FromFile(filename)
    #print md5
    retval  = 0
    if ref != md5:
      print("md5 are different: %s should be: %s for file %s"%(md5,ref,filename))
      retval = 1
    #print outputfilename
    return retval
  #else
  print("Unhandled:",filename,"with ret=",ret)
  return 1

if __name__ == "__main__":
  sucess = 0
  try:
    filename = os.sys.argv[1]
    sucess += TestDCMTKMD5( filename, True )
  except:
    # loop over all files:
    t = gdcm.Testing()
    gdcm.Trace.WarningOff()
    gdcm.Trace.DebugOff()
    nfiles = t.GetNumberOfFileNames()
    for i in range(0,nfiles):
      filename = t.GetFileName(i)
      sucess += TestDCMTKMD5( filename )

  # Test succeed ?
  sys.exit(sucess)
