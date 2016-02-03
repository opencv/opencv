/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSEGMENTWRITER_H
#define GDCMSEGMENTWRITER_H

#include <gdcmWriter.h>
#include <gdcmSegment.h>

namespace gdcm
{

/**
  * \brief  This class defines a segment writer.
  * It writes attributes of group 0x0062.
  *
  * \see  PS 3.3 C.8.20.2 and C.8.23
  */
class GDCM_EXPORT SegmentWriter : public Writer
{
public:
  typedef std::vector< SmartPointer< Segment > > SegmentVector;

  SegmentWriter();

  virtual ~SegmentWriter();

  /// Write
  bool Write(); // Set to protected ?

  //**        Segment getters/setters     **//
  unsigned int GetNumberOfSegments() const;
  void SetNumberOfSegments(const unsigned int size);

  const SegmentVector & GetSegments() const;
  SegmentVector & GetSegments();
  SmartPointer< Segment > GetSegment(const unsigned int idx = 0) const;

  void AddSegment(SmartPointer< Segment > segment);

  void SetSegments(SegmentVector & segments);

protected:

  bool PrepareWrite();


  SegmentVector Segments;
};

}

#endif // GDCMSEGMENTWRITER_H
