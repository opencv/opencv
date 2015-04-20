/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMOVERLAY_H
#define GDCMOVERLAY_H

#include "gdcmTypes.h"
#include "gdcmObject.h"

namespace gdcm
{

class OverlayInternal;
class ByteValue;
class DataSet;
class DataElement;
/**
 * \brief Overlay class
 * \note
 * see AreOverlaysInPixelData
 *
 * \todo
 *  Is there actually any way to recognize an overlay ? On images with multiple overlay I do not see
 *  any way to differenciate them (other than the group tag).
 *
 *  Example:
 */
class GDCM_EXPORT Overlay : public Object
{
public:
  Overlay();
  ~Overlay();
  /// Print
  void Print(std::ostream &) const;

  /// Update overlay from data element de:
  void Update(const DataElement & de);

  /// Set Group number
  void SetGroup(unsigned short group);
  /// Get Group number
  unsigned short GetGroup() const;
  /// set rows
  void SetRows(unsigned short rows);
  /// get rows
  unsigned short GetRows() const;
  /// set columns
  void SetColumns(unsigned short columns);
  /// get columns
  unsigned short GetColumns() const;
  /// set number of frames
  void SetNumberOfFrames(unsigned int numberofframes);
  /// set description
  void SetDescription(const char* description);
  /// get description
  const char *GetDescription() const;
  typedef enum {
    Invalid  = 0,
    Graphics = 1,
    ROI      = 2
  } OverlayType;
  /// set type
  void SetType(const char* type);
  /// get type
  const char *GetType() const;
  OverlayType GetTypeAsEnum() const;
  static const char *GetOverlayTypeAsString(OverlayType ot);
  static OverlayType GetOverlayTypeFromString(const char *);
  /// set origin
  void SetOrigin(const signed short origin[2]);
  /// get origin
  const signed short * GetOrigin() const;
  /// set frame origin
  void SetFrameOrigin(unsigned short frameorigin);
  /// set bits allocated
  void SetBitsAllocated(unsigned short bitsallocated);
  /// return bits allocated
  unsigned short GetBitsAllocated() const;
  /// set bit position
  void SetBitPosition(unsigned short bitposition);
  /// return bit position
  unsigned short GetBitPosition() const;

  /// set overlay from byte array + length
  void SetOverlay(const char *array, size_t length);
  ///
  bool GrabOverlayFromPixelData(DataSet const &ds);

  /// Return the Overlay Data as ByteValue:
  /// Not thread safe
  const ByteValue &GetOverlayData() const;

  /// Return whether or not the Overlay is empty:
  bool IsEmpty() const;

  /// return true if all bits are set to 0
  bool IsZero() const;

  /// return if the Overlay is stored in the pixel data or not
  bool IsInPixelData() const;

  /// Set wether or no the OverlayData is in the Pixel Data:
  void IsInPixelData(bool b);

  /// Decode the internal OverlayData (packed bits) into unpacked representation
  void Decompress(std::ostream &os) const;

  /// Retrieve the size of the buffer needed to hold the Overlay
  /// as specified by Col & Row parameters
  size_t GetUnpackBufferLength() const;

  /// Retrieve the unpack buffer for Overlay. This is an error if
  /// the size if below GetUnpackBufferLength()
  bool GetUnpackBuffer(char *buffer, size_t len) const;

  Overlay(Overlay const &ov);

private:
  OverlayInternal *Internal;
};

} // end namespace gdcm

#endif //GDCMOVERLAY_H
