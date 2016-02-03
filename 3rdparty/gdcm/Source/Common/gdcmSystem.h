/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSYSTEM_H
#define GDCMSYSTEM_H

#include "gdcmTypes.h"

namespace gdcm
{

/**
 * \brief Class to do system operation
 * \details OS independent functionalities
 */
class GDCM_EXPORT System
{
public:
  /// Create a directory name path
  static bool MakeDirectory(const char *path);
  /// Check whether the specified file exist on the sytem
  static bool FileExists(const char* filename);
  /// Check whether the file specified is a directory:
  static bool FileIsDirectory(const char* name);
  /// Check whether name is a symlink
  static bool FileIsSymlink(const char* name);
  /// remove a file named source
  static bool RemoveFile(const char* source);
  /// remove a directory named source
  static bool DeleteDirectory(const char *source);

  /// Return the last error
  static const char *GetLastSystemError();

  /// Return the filesize. 0 if file does not exist.
  /// \warning you need to use FileExists to differentiate between empty file and missing file.
  /// \warning for very large size file and on system where size_t is not appropriate to store
  /// off_t value the function will return 0.
  static size_t FileSize(const char* filename);

  /// Return the time of last modification of file
  /// 0 if the file does not exist
  static time_t FileTime(const char* filename);

  /// Return the directory the current process (executable) is located:
  /// NOT THREAD SAFE
  static const char *GetCurrentProcessFileName();

  /// Return the directory the current module is located:
  /// NOT THREAD SAFE
  static const char *GetCurrentModuleFileName();

  /// On some system (Apple) return the path to the current bundled 'Resources' directory
  /// NOT THREAD SAFE
  static const char *GetCurrentResourcesDirectory();

  // TODO some system calls
  // Chdir
  // copy a file

  /// Retrieve the hostname, only the first 255 byte are copyied.
  /// This may come handy to specify the Station Name
  static bool GetHostName(char hostname[255]);

  // In the following the size '22' is explicitly listed. You need to pass in
  // at least 22bytes of array. If the string is an output it will be
  // automatically padded ( array[21] == 0 ) for you.
  // Those functions: GetCurrentDateTime / FormatDateTime / ParseDateTime do
  // not return the &YYZZ part of the DT structure as defined in DICOM PS 3.5 -
  // 2008 In this case it is simple to split the date[22] into a DA and TM
  // structure

  /// Return the current data time, and format it as ASCII text.
  /// This is simply a call to gettimeofday + FormatDateTime, since WIN32 do not have an
  /// implementation for gettimeofday, this is more portable.
  /// The call time(0) is not precise for our resolution
  static bool GetCurrentDateTime(char date[22]);

  /// format as ASCII text a time_t with milliseconds
  /// See VR::DT from DICOM PS 3.5
  /// milliseconds is in the range [0, 999999]
  static bool FormatDateTime(char date[22], time_t t, long milliseconds = 0);

  /// Parse a date stored as ASCII text into a time_t structured (discard millisecond if any)
  static bool ParseDateTime(time_t &timep, const char date[22]);

  /// Parse a date stored as ASCII text into a time_t structured and millisecond
  /// \see FormatDateTime
  static bool ParseDateTime(time_t &timep, long &milliseconds, const char date[22]);

  /// Return the value for Timezone Offset From UTC as string.
  /// \warning not thread safe
  static const char *GetTimezoneOffsetFromUTC();

  /// Used internally by the UIDGenerator class to convert a uuid tape to a
  /// DICOM VR:UI type
  static size_t EncodeBytes(char *out, const unsigned char *data, int size);

  /// consistent func for C99 spec of strcasecmp/strncasecmp
  static int StrCaseCmp(const char *s1, const char *s2);
  /// \pre n != 0
  static int StrNCaseCmp(const char *s1, const char *s2, size_t n);

  /// Return current working directory
  /// Warning: if current working path is too long (>2048 bytes) the call will fail
  /// and call will return NULL
  /// NOT THREAD SAFE
  static const char * GetCWD();

  /// strtok_r
  static char *StrTokR(char *ptr, const char *sep, char **end);

  /// strsep
  /// param stringp is passed by pointer, it may be modified, you'll need to
  /// make a copy, in case you want to free the memory pointed at
  static char *StrSep(char **stringp, const char *delim);

  /// return `locale charmap`
  static const char *GetLocaleCharset();

  /// NOT THREAD SAFE
/*
  static void SetArgv0(const char *);
  static const char* GetArgv0();
*/

protected:
  static bool GetPermissions(const char* file, unsigned short& mode);
  static bool SetPermissions(const char* file, unsigned short mode);

private:
};

} // end namespace gdcm

#endif //GDCMSYSTEM_H
