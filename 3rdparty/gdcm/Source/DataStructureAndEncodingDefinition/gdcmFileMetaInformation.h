/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILEMETAINFORMATION_H
#define GDCMFILEMETAINFORMATION_H

#include "gdcmPreamble.h"
#include "gdcmDataSet.h"
#include "gdcmDataElement.h"
#include "gdcmMediaStorage.h"
#include "gdcmTransferSyntax.h"
#include "gdcmExplicitDataElement.h"

namespace gdcm
{
/**
 * \brief Class to represent a File Meta Information
 *
 * \details FileMetaInformation is a Explicit Structured Set.  Whenever the
 * file contains an ImplicitDataElement DataSet, a conversion will take place.
 *
 * Definition:
 * The File Meta Information includes identifying information on the
 * encapsulated Data Set. This header consists of a 128 byte File Preamble,
 * followed by a 4 byte DICOM prefix, followed by the File Meta Elements shown
 * in Table 7.1-1. This header shall be present in every DICOM file.
 *
 * \see Writer Reader
 */
class GDCM_EXPORT FileMetaInformation : public DataSet
{
public:
  // FIXME: TransferSyntax::TS_END -> TransferSyntax::ImplicitDataElement
  FileMetaInformation():DataSetTS(TransferSyntax::TS_END),MetaInformationTS(TransferSyntax::Unknown),DataSetMS(MediaStorage::MS_END) {}

  friend std::ostream &operator<<(std::ostream &_os, const FileMetaInformation &_val);

  bool IsValid() const { return true; }

  TransferSyntax::NegociatedType GetMetaInformationTS() const { return MetaInformationTS; }
  void SetDataSetTransferSyntax(const TransferSyntax &ts);
  const TransferSyntax &GetDataSetTransferSyntax() const { return DataSetTS; }
  MediaStorage GetMediaStorage() const;
  std::string GetMediaStorageAsString() const;

  // FIXME: no virtual function means: duplicate code...
  void Insert(const DataElement& de) {
    if( de.GetTag().GetGroup() == 0x0002 )
      {
      InsertDataElement( de );
      }
    else
      {
      gdcmErrorMacro( "Cannot add element with group != 0x0002 in the file meta header: " << de );
      }
  }
  void Replace(const DataElement& de) {
    Remove(de.GetTag());
    Insert(de);
  }

  /// Read
  std::istream &Read(std::istream &is);
  std::istream &ReadCompat(std::istream &is);

  /// Write
  std::ostream &Write(std::ostream &os) const;

  /// Construct a FileMetaInformation from an already existing DataSet:
  void FillFromDataSet(DataSet const &ds);

  /// Get Preamble
  const Preamble &GetPreamble() const { return P; }
  Preamble &GetPreamble() { return P; }
  void SetPreamble(const Preamble &p) { P = p; }

  /// Override the GDCM default values:
  static void SetImplementationClassUID(const char * imp);
  static void AppendImplementationClassUID(const char * imp);
  static const char *GetImplementationClassUID();
  static void SetImplementationVersionName(const char * version);
  static const char *GetImplementationVersionName();
  static void SetSourceApplicationEntityTitle(const char * title);
  static const char *GetSourceApplicationEntityTitle();

  FileMetaInformation(FileMetaInformation const &fmi):DataSet(fmi)
    {
    DataSetTS = fmi.DataSetTS;
    MetaInformationTS = fmi.MetaInformationTS;
    DataSetMS = fmi.DataSetMS;
    }

    VL GetFullLength() const {
      return P.GetLength() + DataSet::GetLength<ExplicitDataElement>();
    }

protected:
  void ComputeDataSetTransferSyntax(); // FIXME

  template <typename TSwap>
  std::istream &ReadCompatInternal(std::istream &is);

  void Default();
  void ComputeDataSetMediaStorageSOPClass();

  TransferSyntax DataSetTS;
  TransferSyntax::NegociatedType MetaInformationTS;
  MediaStorage::MSType DataSetMS;

protected:
  static const char * GetFileMetaInformationVersion();
  static const char * GetGDCMImplementationClassUID();
  static const char * GetGDCMImplementationVersionName();
  static const char * GetGDCMSourceApplicationEntityTitle();

private:
  Preamble P;

//static stuff:
  static const char GDCM_FILE_META_INFORMATION_VERSION[];
  static const char GDCM_IMPLEMENTATION_CLASS_UID[];
  static const char GDCM_IMPLEMENTATION_VERSION_NAME[];
  static const char GDCM_SOURCE_APPLICATION_ENTITY_TITLE[];
  static std::string ImplementationClassUID;
  static std::string ImplementationVersionName;
  static std::string SourceApplicationEntityTitle;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const FileMetaInformation &val)
{
  os << val.GetPreamble() << std::endl;
  val.Print( os );
  return os;
}

} // end namespace gdcm

#endif //GDCMFILEMETAINFORMATION_H
