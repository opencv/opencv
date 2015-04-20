#!/usr/bin/env python
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

# Loop over all .h file, extract the name since by convention this is the name
# of the class, and then try to load that name in the python shell

import sys,os,stat
import gdcm

blacklist = (
"_j2k" # :)
"_jp2" # :)
"treamimpl" # :)
"TestDriver"
# DataStructureAndEncodingDefinition
"ByteBuffer" # WTF ?
"ExplicitDataElement"
"CP246ExplicitDataElement"
"ImplicitDataElement"
"Element"
"ValueIO"
"ParseException"
"ByteSwapFilter"
"ExplicitImplicitDataElement"
"UNExplicitDataElement"
"UNExplicitImplicitDataElement"
"Attribute"
"VR16ExplicitDataElement"
"LO" # issue with swig
"String"
"CodeString"
"Parser"
"TagToVR"

# DataDict:
"TagToType"
"GroupDict"
"DictConverter"
# Information thingy :
"MacroEntry"
"XMLDictReader"
"TableReader"
"Table"
"XMLPrivateDictReader"
# Common
"LegacyMacro"
"Swapper"
"SmartPointer"
"Win32"
"StaticAssert"
"DeflateStream"
"Types"
"Exception"
"ByteSwap"
"Terminal"
"CryptoFactory"
"CAPICryptoFactory"
"CAPICryptographicMessageSyntax"
"FileNameEvent"
"OpenSSLCryptoFactory"
"OpenSSLCryptographicMessageSyntax"
"OpenSSLP7CryptoFactory"
"OpenSSLP7CryptographicMessageSyntax"
# MediaStorageAndFileFormat
"TagKeywords"
"ConstCharWrapper"
"ImageConverter"
"SerieHelper"
# Do not expose low level jpeg implementation detail
"JPEG8Codec"
"JPEG12Codec"
"JPEG16Codec"
"JPEG2000Codec"
# segment
"Segment"
"SegmentHelper"
"SegmentReader"
"SegmentWriter"
#mesh
"MeshPrimitive"
# surface
"Surface"
"SurfaceHelper"
"SurfaceReader"
"SurfaceWriter"
# For now remove the codec part:
"ImageCodec"
"DeltaEncodingCodec"
"RLECodec"
"RAWCodec"
"AudioCodec"
"EncapsulatedDocument"
"JPEGCodec"
"PVRGCodec"
"KAKADUCodec"
"JPEGLSCodec"
"PNMCodec"
"PGXCodec"
"PDFCodec"
"Decoder"
"Coder"
"ImageChangePhotometricInterpretation"
"IconImage" # FIXME
"StreamImageReader"
"StreamImageWriter"
"IconImageFilter"
"IconImageGenerator"
"DirectoryHelper"
"DataEvent"
"DataSetEvent"
# MEXD
"ApplicationContext"
"AAssociateRJPDU"
"AAssociateACPDU"
"ULBasicCallback"
"ULActionDT"
"ULActionAA"
"QueryPatient"
"ULActionAE"
"ImplementationVersionNameSub"
"NetworkStateID"
"AReleaseRQPDU"
"MoveStudyRootQuery"
"ULConnectionCallback"
"NetworkEvents"
"QueryImage"
"ULEvent"
"PresentationContextRQ"
"PDataTFPDU"
"PresentationContextAC"
"DIMSE"
"PresentationContextGenerator"
"AAssociateRQPDU"
"ImplementationUIDSub"
"ULConnection"
"PresentationContext"
"QueryStudy"
"MovePatientRootQuery"
"CEchoMessages"
"QueryFactory"
"ULConnectionManager"
"ULConnectionInfo"
"MaximumLengthSub"
"TransferSyntaxSub"
"ARTIMTimer"
"AbstractSyntax"
"CFindMessages"
"AsynchronousOperationsWindowSub"
"ImplementationClassUIDSub"
"UserInformation"
"CMoveMessages"
"PDUFactory"
"CStoreMessages"
"FindStudyRootQuery"
"AAbortPDU"
"BaseCompositeMessage"
"ULActionAR"
"AReleaseRPPDU"
"ULTransitionTable"
"PresentationDataValue"
"BasePDU"
"QuerySeries"
"ULAction"
"ULWritingCallback"
"CompositeMessageFactory"
"CommandDataSet"
"RoleSelectionSub"
"SOPClassExtendedNegociationSub"
"FindPatientRootQuery"
"ServiceClassApplicationInformation"
)

def processonedir(dirname):
  gdcmclasses = dir(gdcm)
  subtotal = 0
  for file in os.listdir(dirname):
    #print file[-2:]
    if file[-2:] != '.h': continue
    #print file[4:-2]
    gdcmclass = file[4:-2]
    if gdcmclass in gdcmclasses:
      print("ok:", gdcmclass)
    else:
      if not gdcmclass in blacklist:
        print("not wrapped:",gdcmclass)
        subtotal += 1
  return subtotal

if __name__ == "__main__":
  dirname = os.sys.argv[1]

  total = 0
  for d in os.listdir(dirname):
    if d == '.svn': continue
    pathname = os.path.join(dirname, d)
    #print "pathname:",pathname
    #print os.stat(pathname)
    mode = os.stat(pathname)[stat.ST_MODE]
    if stat.S_ISDIR(mode):
      print("processing directory:", pathname)
      total += processonedir(pathname)

  print("number of class not wrap:%d"%total)
  sys.exit(total)
