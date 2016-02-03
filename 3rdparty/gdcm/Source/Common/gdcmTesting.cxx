  /*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTesting.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmMD5.h"

#include <string.h> // strcmp
#include <stdlib.h> // malloc


namespace gdcm
{

#ifndef GDCM_BUILD_TESTING
#error how did that happen ?
#endif

#include "gdcmDataFileNames.cxx"
#include "gdcmMD5DataImages.cxx"
#include "gdcmMD5DataBrokenImages.cxx"
#include "gdcmMediaStorageDataFiles.cxx"
#include "gdcmStreamOffsetDataFiles.cxx"
// After gdcmStreamOffsetDataFiles:
#include "gdcmSelectedTagsOffsetDataFiles.cxx"
#include "gdcmSelectedPrivateGroupOffsetDataFiles.cxx"

bool Testing::ComputeMD5(const char *buffer, unsigned long buf_len,
  char digest_str[33])
{
  return MD5::Compute(buffer, buf_len, digest_str);
}

bool Testing::ComputeFileMD5(const char *filename, char *digest_str)
{
  return MD5::ComputeFile(filename, digest_str);
}

const char * const *Testing::GetFileNames()
{
  return gdcmDataFileNames;
}

unsigned int Testing::GetNumberOfFileNames()
{
  // Do not count NULL value:
  static const unsigned int size = sizeof(gdcmDataFileNames)/sizeof(*gdcmDataFileNames) - 1;
  return size;
}

const char * Testing::GetFileName(unsigned int file)
{
  if( file < Testing::GetNumberOfFileNames() ) return gdcmDataFileNames[file];
  return NULL;
}

Testing::MediaStorageDataFilesType Testing::GetMediaStorageDataFiles()
{
  return gdcmMediaStorageDataFiles;
}
unsigned int Testing::GetNumberOfMediaStorageDataFiles()
{
  // Do not count NULL value:
  static const unsigned int size = sizeof(gdcmMediaStorageDataFiles)/sizeof(*gdcmMediaStorageDataFiles) - 1;
  return size;
}
const char * const * Testing::GetMediaStorageDataFile(unsigned int file)
{
  if( file < Testing::GetNumberOfMediaStorageDataFiles() ) return gdcmMediaStorageDataFiles[file];
  // else return the {0x0, 0x0} sentinel:
  assert( *gdcmMediaStorageDataFiles[ Testing::GetNumberOfMediaStorageDataFiles() ] == 0 );
  return gdcmMediaStorageDataFiles[ Testing::GetNumberOfMediaStorageDataFiles() ];
}
const char * Testing::GetMediaStorageFromFile(const char *filepath)
{
  unsigned int i = 0;
  MediaStorageDataFilesType mediastorages = GetMediaStorageDataFiles();
  const char *p = mediastorages[i][0];
  Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = mediastorages[i][0];
    }
  // \postcondition always valid (before sentinel)
  assert( i <= GetNumberOfMediaStorageDataFiles() );
  return mediastorages[i][1];
}


Testing::MD5DataImagesType Testing::GetMD5DataImages()
{
  return gdcmMD5DataImages;
}
unsigned int Testing::GetNumberOfMD5DataImages()
{
  // Do not count NULL value:
  static const unsigned int size = sizeof(gdcmMD5DataImages)/sizeof(*gdcmMD5DataImages) - 1;
  return size;
}

const char * const * Testing::GetMD5DataImage(unsigned int file)
{
  if( file < Testing::GetNumberOfMD5DataImages() ) return gdcmMD5DataImages[file];
  // else return the {0x0, 0x0} sentinel:
  assert( *gdcmMD5DataImages[ Testing::GetNumberOfMD5DataImages() ] == 0 );
  return gdcmMD5DataImages[ Testing::GetNumberOfMD5DataImages() ];
}

const char * Testing::GetMD5FromFile(const char *filepath)
{
  if(!filepath) return NULL;
  unsigned int i = 0;
  MD5DataImagesType md5s = GetMD5DataImages();
  const char *p = md5s[i][1];
  Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = md5s[i][1];
    }
  // \postcondition always valid (before sentinel)
  assert( i <= GetNumberOfMD5DataImages() );
  return md5s[i][0];
}

const char * Testing::GetMD5FromBrokenFile(const char *filepath)
{
  int i = 0;
  Testing::MD5DataImagesType md5s = gdcmMD5DataBrokenImages; //GetMD5DataImages();
  const char *p = md5s[i][1];
  Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = md5s[i][1];
    }
  return md5s[i][0];
}

std::streamoff Testing::GetStreamOffsetFromFile(const char *filepath)
{
  if(!filepath) return 0;
  unsigned int i = 0;
  const StreamOffset* so = gdcmStreamOffsetDataFiles;
  const char *p = so[i].filename;
  Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = so[i].filename;
    }
  return so[i].offset;
}

std::streamoff Testing::GetSelectedPrivateGroupOffsetFromFile(const char *filepath)
{
  if(!filepath) return 0;
  unsigned int i = 0;
  const StreamOffset* so = gdcmSelectedPrivateGroupOffsetDataFiles;
  const char *p = so[i].filename;
  Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = so[i].filename;
    }
  return so[i].offset;
}

std::streamoff Testing::GetSelectedTagsOffsetFromFile(const char *filepath)
{
  if(!filepath) return 0;
  unsigned int i = 0;
  const StreamOffset* so = gdcmSelectedTagsOffsetDataFiles;
  const char *p = so[i].filename;
  Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = so[i].filename;
    }
  return so[i].offset;
}

// See TestImageReader + lossydump = true to generate this list:
struct LossyFile
{
  int lossyflag;
  const char *filename;
};

static const LossyFile gdcmLossyFilenames[] = {
{ 0,"SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" },
{ 0,"MR-MONO2-12-an2.acr" },
{ 0,"gdcm-ACR-LibIDO.acr" },
{ 0,"test.acr" },
{ 0,"MR-MONO2-12-angio-an1.acr" },
{ 0,"LIBIDO-8-ACR_NEMA-Lena_128_128.acr" },
{ 0,"libido1.0-vol.acr" },
{ 0,"gdcm-MR-SIEMENS-16-2.acr" },
{ 0,"CT-MONO2-12-lomb-an2.acr" },
{ 0,"LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
{ 0,"00191113.dcm" },
{ 0,"SignedShortLosslessBug.dcm" },
{ 0,"gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" },
{ 0,"MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },
{ 0,"ACUSON-24-YBR_FULL-RLE.dcm" },
{ 0,"D_CLUNIE_VL2_RLE.dcm" },
{ 0,"MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" },
{ 0,"MR_Philips-Intera_BreaksNOSHADOW.dcm" },
{ 0,"D_CLUNIE_MR2_JPLL.dcm" },
{ 0,"D_CLUNIE_XA1_JPLL.dcm" },
{ 1,"JPEG_LossyYBR.dcm" },
{ 0,"ALOKA_SSD-8-MONO2-RLE-SQ.dcm" },
{ 0,"PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
{ 0,"MR-MONO2-12-shoulder.dcm" },
{ 1,"D_CLUNIE_RG3_JPLY.dcm" },
{ 1,"PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
{ 0,"MR-MONO2-8-16x-heart.dcm" },
{ 0,"SIEMENS_ImageLocationUN.dcm" },
{ 0,"US-PAL-8-10x-echo.dcm" },
{ 0,"PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" },
{ 0,"LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
{ 0,"US-RGB-8-esopecho.dcm" },
{ 0,"GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" },
{ 0,"CT-SIEMENS-Icone-With-PaletteColor.dcm" },
{ 0,"LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" },
{ 0,"FUJI-10-MONO1-ACR_NEMA_2.dcm" },
{ 0,"D_CLUNIE_CT1_RLE.dcm" },
{ 0,"undefined_length_un_vr.dcm" },
{ 0,"D_CLUNIE_MR4_JPLL.dcm" },
{ 1,"DCMTK_JPEGExt_12Bits.dcm" },
{ 0,"CT_16b_signed-UsedBits13.dcm" },
{ 0,"DX_J2K_0Padding.dcm" },
{ 0,"KODAK_CompressedIcon.dcm" },
{ 0,"D_CLUNIE_CT2_JPLL.dcm" },
{ 0,"DermaColorLossLess.dcm" },
{ 0,"GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" },
{ 1,"D_CLUNIE_NM1_JPLY.dcm" },
{ 0,"MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" },
{ 1,"LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" },
{ 0,"D_CLUNIE_CT1_J2KR.dcm" },
{ 0,"LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" },
{ 0,"US-RGB-8-epicard.dcm" },
{ 0,"D_CLUNIE_MR3_RLE.dcm" },
{ 0,"LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" },
{ 0,"US-IRAD-NoPreambleStartWith0005.dcm" },
{ 0,"D_CLUNIE_RG2_JPLL.dcm" },
{ 0,"DMCPACS_ExplicitImplicit_BogusIOP.dcm" },
{ 0,"MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
{ 0,"MR-SIEMENS-DICOM-WithOverlays.dcm" },
{ 0,"SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" },
{ 0,"JDDICOM_Sample2-dcmdjpeg.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" },
{ 1,"D_CLUNIE_MR3_JPLY.dcm" },
{ 0,"MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" },
{ 0,"TheralysGDCM120Bug.dcm" },
{ 0,"PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
{ 1,"US-GE-4AICL142.dcm" },
{ 0,"MR16BitsAllocated_8BitsStored.dcm" },
{ 0,"3E768EB7.dcm" },
{ 0,"SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" },
{ 0,"GE_MR_0025xx1bProtocolDataBlock.dcm" },
{ 0,"MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
{ 0,"ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" },
{ 1,"D_CLUNIE_SC1_JPLY.dcm" },
{ 0,"CT-MONO2-16-chest.dcm" },
{ 0,"D_CLUNIE_MR4_RLE.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" },
{ 1,"ELSCINT1_JP2vsJ2K.dcm" },
{ 0,"D_CLUNIE_CT2_RLE.dcm" },
{ 0,"D_CLUNIE_MR2_RLE.dcm" },
{ 0,"OT-MONO2-8-a7.dcm" },
{ 0,"MR-MONO2-16-head.dcm" },
{ 0,"PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },
{ 1,"gdcm-JPEG-Extended.dcm" },
{ 0,"BugGDCM2_UndefItemWrongVL.dcm" },
{ 0,"D_CLUNIE_MR1_RLE.dcm" },
{ 0,"PICKER-16-MONO2-Nested_icon.dcm" },
{ 0,"D_CLUNIE_VL4_RLE.dcm" },
{ 0,"D_CLUNIE_RG1_RLE.dcm" },
{ 1,"JDDICOM_Sample2.dcm" },
{0,"AMIInvalidPrivateDefinedLengthSQasUN.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" },
{ 0,"CT-MONO2-8-abdo.dcm" },
{ 0,"D_CLUNIE_SC1_RLE.dcm" },
{ 0,"LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" },
{ 0,"D_CLUNIE_RG3_JPLL.dcm" },
{ 0,"SIEMENS_CSA2.dcm" },
{ 0,"LJPEG_BuginGDCM12.dcm" },
{ 0,"CT-SIEMENS-MissingPixelDataInIconSQ.dcm" },
{ 0,"05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
{ 0,"GE_CT_With_Private_compressed-icon.dcm" },
{ 1,"D_CLUNIE_XA1_JPLY.dcm" },
{ 0,"012345.002.050.dcm" },
{ 0,"TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" },
{ 1,"LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" },
{ 0,"gdcm-US-ALOKA-16.dcm" },
{ 0,"THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },
{ 0,"D_CLUNIE_CT1_JPLL.dcm" },
{ 0,"rle16loo.dcm" },
{ 0,"D_CLUNIE_US1_RLE.dcm" },
{ 0,"LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" },
{ 0,"RadBWLossLess.dcm" },
{ 1,"D_CLUNIE_MR1_JPLY.dcm" },
{ 0,"JPEGDefinedLengthSequenceOfFragments.dcm" },
{ 0,"GE_DLX-8-MONO2-PrivateSyntax.dcm" },
{ 0,"gdcm-JPEG-LossLess3a.dcm" },
{ 0,"TG18-CH-2k-01.dcm" },
{ 0,"OT-PAL-8-face.dcm" },
{ 0,"D_CLUNIE_NM1_RLE.dcm" },
{ 0,"rle16sti.dcm" },
{ 0,"GE_GENESIS-16-MONO2-WrongLengthItem.dcm" },
{ 1,"D_CLUNIE_CT1_J2KI.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" },
{ 0,"LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" },
{ 1,"D_CLUNIE_MR4_JPLY.dcm" },
{ 0,"OsirixFake16BitsStoredFakeSpacing.dcm" },
{ 0,"PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" },
{ 0,"MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
{ 0,"D_CLUNIE_CT1_JLSL.dcm" },
{ 1,"D_CLUNIE_CT1_JLSN.dcm" },
{ 0,"D_CLUNIE_RG3_RLE.dcm" },
{ 1,"SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" },
{ 0,"CT-MONO2-16-brain.dcm" },
{ 1,"D_CLUNIE_RG2_JPLY.dcm" },
{ 1,"MAROTECH_CT_JP2Lossy.dcm" },
{ 0,"D_CLUNIE_MR1_JPLL.dcm" },
{ 0,"ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" },
{ 0,"MR_SIEMENS_forceLoad29-1010_29-1020.dcm" },
{ 0,"simpleImageWithIcon.dcm" },
{ 0,"D_CLUNIE_MR3_JPLL.dcm" },
{ 0,"D_CLUNIE_RG1_JPLL.dcm" },
{ 0,"US-MONO2-8-8x-execho.dcm" },
{ 0,"XA-MONO2-8-12x-catheter.dcm" },
{ 0,"GE_LOGIQBook-8-RGB-HugePreview.dcm" },
{ 0,"gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },
{ 0,"D_CLUNIE_XA1_RLE.dcm" },
{ 0,"NM-MONO2-16-13x-heart.dcm" },
{ 0,"gdcm-JPEG-LossLessThoravision.dcm" },
{ 0,"GE_DLX-8-MONO2-Multiframe.dcm" },
{ 0,"PHILIPS_Intera-16-MONO2-Uncompress.dcm" },
{ 1,"D_CLUNIE_MR2_JPLY.dcm" },
{ 0,"05148044-mr-siemens-avanto-syngo.dcm" },
{ 0,"D_CLUNIE_VL3_RLE.dcm" },
{ 0,"D_CLUNIE_RG2_RLE.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },
{ 0,"KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
{ 0,"SIEMENS-MR-RGB-16Bits.dcm" },
{ 0,"CR-MONO1-10-chest.dcm" },
{ 0,"DX_GE_FALCON_SNOWY-VOI.dcm" },
{ 0,"US-IRAD-NoPreambleStartWith0003.dcm" },
{ 0,"MR-Brucker-CineTagging-NonSquarePixels.dcm" },
{ 0,"D_CLUNIE_VL6_RLE.dcm" },
{ 0,"MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" },
{ 0,"fffc0000UN.dcm" },
{ 0,"SIEMENS_Sonata-12-MONO2-SQ.dcm" },
{ 0,"ACUSON-24-YBR_FULL-RLE-b.dcm" },
{ 0,"CT-MONO2-16-ankle.dcm" },
{ 0,"GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },
{ 0,"CT-MONO2-16-ort.dcm" },
{ 0,"LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" },
{ 0,"D_CLUNIE_NM1_JPLL.dcm" },
{ 0,"D_CLUNIE_VL1_RLE.dcm" },
{ 0,"SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" },
{ 0,"00191113.dcm" },
{ 0,"TestVRSQUN2.dcm" },
{ 0,"PHILIPS_GDCM12xBug.dcm"},
{ 0,"PHILIPS_GDCM12xBug2.dcm"},
{ 0,"TestVRSQUN1.dcm"} ,
{ 1,"multiframegrayscalebytescis.dcm" },
{ 1,"multiframegrayscalewordscis.dcm" },
{ 1,"multiframesinglebitscis.dcm" },
{ 1,"multiframetruecolorscis.dcm" },
{ 1, "SinglePrecisionSC.dcm" },
{ 0, "signedtruecoloroldsc.dcm" },
{ 0, "o.dcm" },
{ 0, "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
{ 0, "IM-0001-0066.CommandTag00.dcm" },
{ 0, "NM_Kakadu44_SOTmarkerincons.dcm" },
{ 0, "GDCMJ2K_TextGBR.dcm" },
{ 0, "PhilipsInteraSeqTermInvLen.dcm" },
{ 0, "LIBIDO-24-ACR_NEMA-Rectangle.dcm" },
{ 0, "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
{ 0, "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
{ 0, "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },
{ 0, "NM-PAL-16-PixRep1.dcm" },
{ 0, "MEDILABInvalidCP246_EVRLESQasUN.dcm" },
{ 0, "JPEGInvalidSecondFrag.dcm" },
{ 0, "SC16BitsAllocated_8BitsStoredJ2K.dcm" },
{ 0, "SC16BitsAllocated_8BitsStoredJPEG.dcm" },
{ 0, NULL }
};


int Testing::GetLossyFlagFromFile(const char *filename)
{
  if( !filename ) return 0;
  gdcm::Filename fn = filename;
  const char *file = fn.GetName();
  const LossyFile *pfiles = gdcmLossyFilenames;
  while( pfiles->filename && strcmp(pfiles->filename, file) != 0 )
    {
    ++pfiles;
    }
  if( !(pfiles->filename) )
    {
    std::cerr << "Error: No ref table for: " << filename << std::endl;
    return -1;
    }
  assert( pfiles->filename ); // need to update ref table
  return pfiles->lossyflag;
}

const char *Testing::GetDataRoot()
{
  return GDCM_DATA_ROOT;
}

const char *Testing::GetDataExtraRoot()
{
  return GDCM_DATA_EXTRA_ROOT;
}

const char *Testing::GetPixelSpacingDataRoot()
{
  return GDCM_PIXEL_SPACING_DATA_ROOT;
}

const char *Testing::GetTempDirectory(const char * subdir)
{
  if( !subdir ) return GDCM_TEMP_DIRECTORY;
  // else
  static std::string buffer;
  std::string tmpdir = GDCM_TEMP_DIRECTORY;
  tmpdir += "/";
  tmpdir += subdir;
  buffer = tmpdir;
  return buffer.c_str();
}

const wchar_t *Testing::GetTempDirectoryW(const wchar_t * subdir)
{
  static std::wstring buffer;
  wchar_t wname[4096]; // FIXME
  size_t len = mbstowcs(wname,GDCM_TEMP_DIRECTORY,sizeof(wname)/sizeof(wchar_t));
  (void)len;
  if( !subdir )
    {
    buffer = wname;
    return buffer.c_str();
    }
  // else
  std::wstring tmpdir = wname;
  tmpdir += L"/";
  tmpdir += subdir;
  buffer = tmpdir;
  return buffer.c_str();
}

const char * Testing::GetTempFilename(const char *filename, const char * subdir)
{
  if( !filename ) return 0;

  static std::string buffer;
  std::string outfilename = GetTempDirectory(subdir);
  outfilename += "/";
  gdcm::Filename out(filename);
  outfilename += out.GetName();
  buffer = outfilename;

  return buffer.c_str();
}

void Testing::Print(std::ostream &os)
{
  os << "DataFileNames:\n";
  const char * const * filenames = gdcmDataFileNames;
  while( *filenames )
    {
    os << *filenames << "\n";
    ++filenames;
    }

  os << "MD5DataImages:\n";
  MD5DataImagesType md5s = gdcmMD5DataImages;
  while( (*md5s)[0] )
    {
    os << (*md5s)[0] << " -> " << (*md5s)[1] << "\n";
    ++md5s;
    }
}

const wchar_t* Testing::GetTempFilenameW(const wchar_t *filename, const wchar_t* subdir)
{
  // mbsrtowcs
  // mbstowcs
  if( !filename ) return 0;

  static std::wstring buffer;
  std::wstring outfilename = GetTempDirectoryW(subdir);
  outfilename += L"/";

  //gdcm::Filename out(filename);
  //outfilename += out.GetName();
  buffer = outfilename;
  buffer += filename;

  return buffer.c_str();
}

const char *Testing::GetSourceDirectory()
{
  return GDCM_SOURCE_DIR;
}

} // end of namespace gdcm
