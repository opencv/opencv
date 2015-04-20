/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILE_H
#define GDCMFILE_H

#include "gdcmObject.h"
#include "gdcmDataSet.h"
#include "gdcmFileMetaInformation.h"

namespace gdcm
{

/**
 * \brief a DICOM File
 * See PS 3.10 File: A File is an ordered string of zero or more bytes, where
 * the first byte is at the beginning of the file and the last byte at the end
 * of the File. Files are identified by a unique File ID and may by written,
 * read and/or deleted.
 *
 * \see Reader Writer
 */
class GDCM_EXPORT File : public Object
{
public:
  File() {};

  friend std::ostream &operator<<(std::ostream &os, const File &val);

  /// Read
  std::istream &Read(std::istream &is);

  /// Write
  std::ostream const &Write(std::ostream &os) const;

  /// Get File Meta Information
  const FileMetaInformation &GetHeader() const { return Header; }

  /// Get File Meta Information
  FileMetaInformation &GetHeader() { return Header; }

  /// Set File Meta Information
  void SetHeader( const FileMetaInformation &fmi ) { Header = fmi; }

  /// Get Data Set
  const DataSet &GetDataSet() const { return DS; }

  /// Get Data Set
  DataSet &GetDataSet() { return DS; }

  /// Set Data Set
  void SetDataSet( const DataSet &ds) { DS = ds; }

private:
  FileMetaInformation Header;
  DataSet DS;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const File &val)
{
  os << val.GetHeader() << std::endl;
  //os << val.GetDataSet() << std::endl; // FIXME
  assert(0);
  return os;
}

} // end namespace gdcm

#endif //GDCMFILE_H
