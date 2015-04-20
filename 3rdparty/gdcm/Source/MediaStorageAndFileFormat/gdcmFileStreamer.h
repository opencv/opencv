/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILESTREAMER_H
#define GDCMFILESTREAMER_H

#include "gdcmSubject.h"
#include "gdcmSmartPointer.h"

namespace gdcm
{
class FileStreamerInternals;

class Tag;
class PrivateTag;
/**
 * \brief FileStreamer
 * This class let a user create a massive DICOM DataSet from a template DICOM
 * file, by appending chunks of data.
 *
 * This class support two mode of operation:
 * 1. Creating a single DataElement by appending chunk after chunk of data.
 *
 * 2. Creating a set of DataElement within the same group, using a private
 * creator for start. New DataElement are added any time the user defined
 * maximum size for data element is reached.
 *
 * \warning any existing DataElement is removed, pick carefully which
 * DataElement to add.
 */
class GDCM_EXPORT FileStreamer : public Subject
{
public:
  FileStreamer();
  ~FileStreamer();

  /// Set input DICOM template filename
  void SetTemplateFileName(const char *filename_native);

  // Decide to check template or not (default: false)
  /// Instead of simply blindly copying the input DICOM Template file, GDCM will
  /// be used to check the input file, and correct any issues recognized within
  /// the file. Only use if you do not have control over the input template
  /// file.
  void CheckTemplateFileName(bool check);

  /// Set output filename (target file)
  void SetOutputFileName(const char *filename_native);

  /// Decide to check the Data Element to be written (default: off)
  /// The implementation has default strategy for checking validity of DataElement.
  /// Currently it only support checking for the following tags:
  /// - (7fe0,0010) Pixel Data
  bool CheckDataElement( const Tag & t );

  /// Start Single Data Element Operation
  /// This will delete any existing Tag t. Need to call it only once.
  bool StartDataElement( const Tag & t );
  /// Append to previously started Tag t
  bool AppendToDataElement( const Tag & t, const char *array, size_t len );
  /// Stop appending to tag t. This will compute the proper attribute length.
  bool StopDataElement( const Tag & t );
  /// Add a hint on the final size of the dataelement. When optimally chosen,
  /// this reduce the number of file in-place copying. Should be called before
  /// StartDataElement
  bool ReserveDataElement( size_t len );

  /// Start Private Group (multiple DataElement) Operation. Each newly added
  /// DataElement will have a length lower than \param maxsizede .
  /// When not specified, maxsizede is set to maximum size allowed by DICOM (= 2^32).
  /// startoffset can be used to specify the very first element you want to
  /// start with (instead of the first possible). Value should be in [0x0, 0xff]
  /// This will find the first available private creator.
  bool StartGroupDataElement( const PrivateTag & pt, size_t maxsizede = 0, uint8_t startoffset = 0 );
  /// Append to previously started private creator
  bool AppendToGroupDataElement( const PrivateTag & pt, const char *array, size_t len );
  /// Stop appending to private creator
  bool StopGroupDataElement( const PrivateTag & pt );
  /// Optimisation: pre-allocate the number of dataelement within the private
  /// group (ndataelement <= 256). Should be called before StartGroupDataElement
  bool ReserveGroupDataElement( unsigned short ndataelement );

  /// for wrapped language: instantiate a reference counted object
  static SmartPointer<FileStreamer> New() { return new FileStreamer; }

private:
  bool InitializeCopy();
  FileStreamerInternals *Internals;
};

} // end namespace gdcm

#endif //GDCMFILESTREAMER_H
