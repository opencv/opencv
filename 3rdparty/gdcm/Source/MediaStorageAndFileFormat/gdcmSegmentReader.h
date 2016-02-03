/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSEGMENTREADER_H
#define GDCMSEGMENTREADER_H

#include <map>

#include <gdcmReader.h>
#include <gdcmSegment.h>

namespace gdcm
{

/**
  * \brief  This class defines a segment reader.
  * It reads attributes of group 0x0062.
  *
  * \see  PS 3.3 C.8.20.2 and C.8.23
  */
class GDCM_EXPORT SegmentReader : public Reader
{
public:
  typedef std::vector< SmartPointer< Segment > > SegmentVector;

  SegmentReader();

  virtual ~SegmentReader();

  /// Read
  virtual bool Read(); // Set to protected ?

  //**        Segment getters/setters     **//
  const SegmentVector GetSegments() const;
  SegmentVector GetSegments();

//  unsigned int GetNumberOfSegments();

protected:

  typedef std::map< unsigned long, SmartPointer< Segment > > SegmentMap;

  bool ReadSegments();

  bool ReadSegment(const Item & segmentItem, const unsigned int idx);


  SegmentMap Segments;  // The key value is item number (in segment sequence)
                        // or the surface number (for a surface segmentation).

};

}

#endif // GDCMSEGMENTREADER_H
