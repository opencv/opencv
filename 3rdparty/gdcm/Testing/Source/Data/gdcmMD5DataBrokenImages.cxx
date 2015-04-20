/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http:/gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/


/*
 * See TestWriter
 */

// This is the full md5 of the rewriten file. The file was manually check
// and is (should be) exactly what should have been written in the first place
// test was done using dcmtk 3.5.4 / dicom3tools
static const char * const gdcmMD5DataBrokenImages[][2] = {
// file has some garbage at the end, replace with a trailing end item.
{ "e8ed75f5e13cc20e96ee716bcc78351b" , "gdcm-JPEG-LossLess3a.dcm" }, // size match

// files are little endian implicit meta data header:
{ "8cb29ba0173c66e7adb4c54c6b0a5896" , "GE_DLX-8-MONO2-PrivateSyntax.dcm" },
{ "ed93b34819bf2acbacefb510476e8d4a" , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },

// stupidest file ever, 0x6 was sent in place of 0x4 ... sigh
{ "c8cce480eac80770a3c6e456c7d8d66f" , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" }, // size match

// big endian / little endian nightmare from PMS
{ "df0e01aae299317db1719e4de72b8937" , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" }, // size match and CheckbigEndian match => ok !
{ "df632a3b5ca38340faa612a23b907ac4" , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" }, // size match and CheckbigEndian match => ok !
{ "6bf002815fad665392acab24f09caa5e" , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" }, // size match and CheckbigEndian match => ok !

// little endian implicit meta header + a couple of attribute sent with correct, but odd length:
{ "e1b2956f781685fc9e46e0da26b8a0fd" , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },

// name says it all. dcmtk does not support this. dicom3tools confirmed that dataset is compatible
{ "66a75503221ef32b0236cf9f78e169ff" , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" }, // size mismatch

// item length are supposed to be 0, not FFFF...
{ "3cc629fa470efb114a14ca3909117eb8" , "SIEMENS-MR-RGB-16Bits.dcm" }, // size match
{ "0fb0cb12f2b038bbfe1014bbf2935026" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" }, // size match


// this is a private syntax from ge with a little endian dataset and big endian pixel data
// FMI is little endian implicit, it also contained a couple of odd length attributes.
// using dcmtk it looks like file are identical
{ "2d23a8d55425c88bdd5e90f866a11607" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" }, // size mismatch

// couple of weird stuff going on... dcdump confirmed dataset is identical in both file
{ "34abc36682a6e6ba22d7295931b39d85" , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" }, // size mismatch

// weird stuff going on. dcdump confirmed dataset are identical. dcmdump can read output, and size match.
{ "82fda19e1f2046a289fe1307b70510af" , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },

// yet another stupidest ever bug, 0xd was replaced with 0xa ... don't ask
{ "a047110a3935dc0fdda24d3b9e4769af" , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" }, // size match

// empty 16bits after VR should be 0 not garbage...
{ "6cded0f160edfab809cddfce2d562671" , "JDDICOM_Sample2.dcm" }, // size match

// simple issue, last fragment is odd, simply need to pad (easy, should be handle by most implementation)
{ "6234a50361f02cb9a739755d63cdd673" , "00191113.dcm" }, // size mismatch (obviously)
{ "28f9d4114b0699630a77d027910e5e41" , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },

// serious bug from gdcm 1.2.0, where VR=UN would be written on 16bits length sigh... no toolkit will ever be able to deal with that thing (and should not anyway)
{ "50752239f24669697897c4b6542bc161" , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },

// unordered dataset
{ "f221e76c6f0758877aa3cf13632480f4" , "dicomdir_Pms_WithVisit_WithPrivate_WithStudyComponents" }, // size match

// frankenstein-type dicom file:
// 1. Implicit encoding is used any time it is not known (most of the time, private element)
// 2. > (0x2001,0x1068) SQ ?   VR=<SQ>   VL=<0xffffffff> is sent twice (don't ask), second entry cannot be stored...
// dcdump kindda show file are somewhat compatible.
{ "69ca7a4300967cf2841da34d7904c6c4" , "TheralysGDCM120Bug.dcm" }, // size mismatch

// GDCM 1.0 generated file. At that time, VL for a start/end item delimitor would be set to 0xFFF... instead of 0x0
// dcmtk / dicom3tools do not seems to care about the value stored for VL, so does GDCM (now).
// As a side note the FMI was set to Little Endian Implicit ...
{ "ddf83cd708e58021a633588927d55ab8" , "BugGDCM2_UndefItemWrongVL.dcm" }, // size mismatch

{ "cb43a6ad60b8eacf718687b82126f625" , "NM_Kakadu44_SOTmarkerincons.dcm" }, // item size mismatch

// Item length are bogus (explicit length)
{ "1225ea0a03b93393f70c73be35e2619d" , "PhilipsInteraSeqTermInvLen.dcm" },

// Two Items in a single Frame JPEG compressed DICOM image:
{ "cd00658f54dbd2d2a9d02d64c6f6497e" , "JPEGInvalidSecondFrag.dcm" },

{ 0 ,0 }
};

