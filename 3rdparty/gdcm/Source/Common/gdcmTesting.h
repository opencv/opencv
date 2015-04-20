/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTESTING_H
#define GDCMTESTING_H

#include "gdcmTypes.h"

#include <iostream>

namespace gdcm
{
/**
 * \brief class for testing
 * \details this class is used for the nightly regression system for GDCM
 * It makes heavily use of md5 computation
 *
 * \see gdcm::MD5 class for md5 computation
 */
//-----------------------------------------------------------------------------
class GDCM_EXPORT Testing
{
public :
  Testing() {};
  ~Testing() {};

  /// MD5 stuff
  /// digest_str needs to be at least : strlen = [2*16+1];
  /// string will be \0 padded. (md5 are 32 bytes long)
  /// Testing is not meant to be shipped with an installed GDCM release, always
  /// prefer the gdcm::MD5 API when doing md5 computation.
  static bool ComputeMD5(const char *buffer, unsigned long buf_len,
    char digest_str[33]);
  static bool ComputeFileMD5(const char *filename, char digest_str[33]);

  /// Print
  void Print(std::ostream &os = std::cout);

  /// return the table of fullpath to gdcmData DICOM files:
  static const char * const * GetFileNames();
  static unsigned int GetNumberOfFileNames();
  static const char * GetFileName(unsigned int file);

  /// return the table that map the media storage (as string) of a filename (gdcmData)
  typedef const char* const (*MediaStorageDataFilesType)[2];
  static MediaStorageDataFilesType GetMediaStorageDataFiles();
  static unsigned int GetNumberOfMediaStorageDataFiles();
  static const char * const * GetMediaStorageDataFile(unsigned int file);
  static const char * GetMediaStorageFromFile(const char *filepath);

  /// return the table that map the md5 (as in md5sum) of the Pixel Data associated
  /// to a filename
  typedef const char* const (*MD5DataImagesType)[2];
  static MD5DataImagesType GetMD5DataImages();
  static unsigned int GetNumberOfMD5DataImages();
  static const char * const * GetMD5DataImage(unsigned int file);
  static const char * GetMD5FromFile(const char *filepath);

  /// Return what should have been the md5 of file 'filepath'
  /// This is based on current GDCM implementation to decipher a broken DICOM file.
  static const char * GetMD5FromBrokenFile(const char *filepath);

  /// Return the offset of the very first pixel cell in the PixelData
  /// -1 if not found
  static std::streamoff GetStreamOffsetFromFile(const char *filepath);

  /// Return the offset just after Pixel Data Length (7fe0,0000) if found.
  /// Otherwise the offset of the very first pixel cell in Pixel Data
  /// -1 if not found
  static std::streamoff GetSelectedTagsOffsetFromFile(const char *filepath);

  /// Return the offset just after private attribute (0009,0010,"GEMS_IDEN_01") if found.
  /// Otherwise the offset of the next attribute
  /// -1 if not found
  static std::streamoff GetSelectedPrivateGroupOffsetFromFile(const char *filepath);

  /// Return the lossy flag of the given filename
  /// -1 -> Error
  ///  0 -> Lossless
  ///  1 -> Lossy
  static int GetLossyFlagFromFile(const char *filepath);

  /// Return the GDCM DATA ROOT
  static const char * GetDataRoot();

  /// Return the GDCM DATA EXTRA ROOT
  static const char * GetDataExtraRoot();

  /// Return the GDCM PIXEL SPACING DATA ROOT (See David Clunie website for dataset)
  static const char * GetPixelSpacingDataRoot();

  /// NOT THREAD SAFE
  /// Returns the temp directory as used in testing needing to output data:
  static const char * GetTempDirectory(const char * subdir = 0);

  /// NOT THREAD SAFE
  static const wchar_t *GetTempDirectoryW(const wchar_t * subdir = 0);

  /// NOT THREAD SAFE
  static const char * GetTempFilename(const char *filename, const char * subdir = 0);

  /// NOT THREAD SAFE
  static const wchar_t* GetTempFilenameW(const wchar_t *filename, const wchar_t* subdir = 0);

  static const char *GetSourceDirectory();
};
} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMTESTING_H
