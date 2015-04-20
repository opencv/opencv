/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSegmentWriter.h"
#include "gdcmAttribute.h"

namespace gdcm
{

SegmentWriter::SegmentWriter()
{
}

SegmentWriter::~SegmentWriter()
{
}

unsigned int SegmentWriter::GetNumberOfSegments() const
{
  return (unsigned int)Segments.size();
}

void SegmentWriter::SetNumberOfSegments(const unsigned int size)
{
  Segments.resize(size);
}

const SegmentWriter::SegmentVector & SegmentWriter::GetSegments() const
{
  return Segments;
}

SegmentWriter::SegmentVector & SegmentWriter::GetSegments()
{
  return Segments;
}

SmartPointer< Segment > SegmentWriter::GetSegment(const unsigned int idx /*= 0*/) const
{
  assert( idx < Segments.size() );
  return Segments[idx];
}

void SegmentWriter::AddSegment(SmartPointer< Segment > segment)
{
  Segments.push_back(segment);
}

void SegmentWriter::SetSegments(SegmentVector & segments)
{
  Segments = segments;
}

bool SegmentWriter::PrepareWrite()
{
  File &      file    = GetFile();
  DataSet &   ds      = file.GetDataSet();

  // Segment Sequence
  SmartPointer<SequenceOfItems> segmentsSQ;
  if( !ds.FindDataElement( Tag(0x0062, 0x0002) ) )
  {
    segmentsSQ = new SequenceOfItems;
    DataElement detmp( Tag(0x0062, 0x0002) );
    detmp.SetVR( VR::SQ );
    detmp.SetValue( *segmentsSQ );
    detmp.SetVLToUndefined();
    ds.Insert( detmp );
  }
  segmentsSQ = ds.GetDataElement( Tag(0x0062, 0x0002) ).GetValueAsSQ();
  segmentsSQ->SetLengthToUndefined();

{
  // Fill the Segment Sequence
  const unsigned int              numberOfSegments  = this->GetNumberOfSegments();
  assert( numberOfSegments );
  const size_t nbItems           = segmentsSQ->GetNumberOfItems();
  if (nbItems < numberOfSegments)
  {
    const size_t diff           = numberOfSegments - nbItems;
    const size_t nbOfItemToMake = (diff > 0?diff:0);
    for(unsigned int i = 1; i <= nbOfItemToMake; ++i)
    {
      Item item;
      item.SetVLToUndefined();
      segmentsSQ->AddItem(item);
    }
  }
}
  // else Should I remove items?

  std::vector< SmartPointer< Segment > >::const_iterator  it0            = Segments.begin();
  std::vector< SmartPointer< Segment > >::const_iterator  it0End         = Segments.end();
  unsigned int                                            itemNumber    = 1;
  unsigned long                                           surfaceNumber = 1;
  for (; it0 != it0End; it0++)
  {
    SmartPointer< Segment > segment = *it0;
    assert( segment );

    Item &    segmentItem = segmentsSQ->GetItem(itemNumber);
    DataSet & segmentDS   = segmentItem.GetNestedDataSet();

    // Segment Number (Type 1)
    Attribute<0x0062, 0x0004> segmentNumberAt;
    unsigned short segmentNumber = segment->GetSegmentNumber();
    if (segmentNumber == 0)
      segmentNumber = (unsigned short)itemNumber;
    segmentNumberAt.SetValue( segmentNumber );
    segmentDS.Replace( segmentNumberAt.GetAsDataElement() );

    // Segment Label (Type 1)
    const char * segmentLabel = segment->GetSegmentLabel();
    if ( strcmp(segmentLabel, "") != 0 )
    {
      gdcmWarningMacro("No segment label specified");
    }
    Attribute<0x0062, 0x0005> segmentLabelAt;
    segmentLabelAt.SetValue( segmentLabel );
    segmentDS.Replace( segmentLabelAt.GetAsDataElement() );

    // Segment Description (Type 3)
    const char * segmentDescription = segment->GetSegmentDescription();
    if ( strcmp(segmentDescription, "") != 0 )
    {
      Attribute<0x0062, 0x0006> segmentDescriptionAt;
      segmentDescriptionAt.SetValue( segmentDescription );
      segmentDS.Replace( segmentDescriptionAt.GetAsDataElement() );
    }

    // Segment Algorithm Type (Type 1)
    const char * segmentAlgorithmType = Segment::GetALGOTypeString( segment->GetSegmentAlgorithmType() );
    if ( segmentAlgorithmType == 0 )
    {
      gdcmWarningMacro("No segment algorithm type specified");
      Attribute<0x0062, 0x0008> segmentAlgorithmTypeAt;
      segmentAlgorithmTypeAt.SetValue( "" );
      segmentDS.Replace( segmentAlgorithmTypeAt.GetAsDataElement() );
    }
    else
    {
    Attribute<0x0062, 0x0008> segmentAlgorithmTypeAt;
    segmentAlgorithmTypeAt.SetValue( segmentAlgorithmType );
    segmentDS.Replace( segmentAlgorithmTypeAt.GetAsDataElement() );
    }

    //*****   GENERAL ANATOMY MANDATORY MACRO ATTRIBUTES   *****//
    {
      const SegmentHelper::BasicCodedEntry & anatReg = segment->GetAnatomicRegion();
      if (anatReg.IsEmpty())
      {
        gdcmWarningMacro("Anatomic region not specified or incomplete");
      }

      // Anatomic Region Sequence (0008,2218) Type 1
      SmartPointer<SequenceOfItems> anatRegSQ;
      const Tag anatRegSQTag(0x0008, 0x2218);
      if( !segmentDS.FindDataElement( anatRegSQTag ) )
      {
        anatRegSQ = new SequenceOfItems;
        DataElement detmp( anatRegSQTag );
        detmp.SetVR( VR::SQ );
        detmp.SetValue( *anatRegSQ );
        detmp.SetVLToUndefined();
        segmentDS.Insert( detmp );
      }
      anatRegSQ = segmentDS.GetDataElement( anatRegSQTag ).GetValueAsSQ();
      anatRegSQ->SetLengthToUndefined();

      // Fill the Anatomic Region Sequence
      const size_t nbItems = anatRegSQ->GetNumberOfItems();
      if (nbItems < 1)  // Only one item is a type 1
      {
        Item item;
        item.SetVLToUndefined();
        anatRegSQ->AddItem(item);
      }

      Item &    anatRegItem = anatRegSQ->GetItem(1);
      DataSet & anatRegDS   = anatRegItem.GetNestedDataSet();

      //*****   CODE SEQUENCE MACRO ATTRIBUTES   *****//
      {
        // Code Value (Type 1)
        Attribute<0x0008, 0x0100> codeValueAt;
        codeValueAt.SetValue( anatReg.CV );
        anatRegDS.Replace( codeValueAt.GetAsDataElement() );

        // Coding Scheme (Type 1)
        Attribute<0x0008, 0x0102> codingSchemeAt;
        codingSchemeAt.SetValue( anatReg.CSD );
        anatRegDS.Replace( codingSchemeAt.GetAsDataElement() );

        // Code Meaning (Type 1)
        Attribute<0x0008, 0x0104> codeMeaningAt;
        codeMeaningAt.SetValue( anatReg.CM );
        anatRegDS.Replace( codeMeaningAt.GetAsDataElement() );
      }
    }

    //*****   Segmented Property Category Code Sequence   *****//
    {
      const SegmentHelper::BasicCodedEntry & propCat = segment->GetPropertyCategory();
      if (propCat.IsEmpty())
      {
        gdcmWarningMacro("Segmented property category not specified or incomplete");
      }

      // Segmented Property Category Code Sequence (0062,0003) Type 1
      SmartPointer<SequenceOfItems> propCatSQ;
      const Tag propCatSQTag(0x0062, 0x0003);
      if( !segmentDS.FindDataElement( propCatSQTag ) )
      {
        propCatSQ = new SequenceOfItems;
        DataElement detmp( propCatSQTag );
        detmp.SetVR( VR::SQ );
        detmp.SetValue( *propCatSQ );
        detmp.SetVLToUndefined();
        segmentDS.Insert( detmp );
      }
      propCatSQ = segmentDS.GetDataElement( propCatSQTag ).GetValueAsSQ();
      propCatSQ->SetLengthToUndefined();

      // Fill the Segmented Property Category Code Sequence
      const size_t nbItems = propCatSQ->GetNumberOfItems();
      if (nbItems < 1)  // Only one item is a type 1
      {
        Item item;
        item.SetVLToUndefined();
        propCatSQ->AddItem(item);
      }

      Item &    propCatItem = propCatSQ->GetItem(1);
      DataSet & propCatDS   = propCatItem.GetNestedDataSet();

      //*****   CODE SEQUENCE MACRO ATTRIBUTES   *****//
      {
        // Code Value (Type 1)
        Attribute<0x0008, 0x0100> codeValueAt;
        codeValueAt.SetValue( propCat.CV );
        propCatDS.Replace( codeValueAt.GetAsDataElement() );

        // Coding Scheme (Type 1)
        Attribute<0x0008, 0x0102> codingSchemeAt;
        codingSchemeAt.SetValue( propCat.CSD );
        propCatDS.Replace( codingSchemeAt.GetAsDataElement() );

        // Code Meaning (Type 1)
        Attribute<0x0008, 0x0104> codeMeaningAt;
        codeMeaningAt.SetValue( propCat.CM );
        propCatDS.Replace( codeMeaningAt.GetAsDataElement() );
      }
    }

    //*****   Segmented Property Type Code Sequence   *****//
    {
      const SegmentHelper::BasicCodedEntry & propType = segment->GetPropertyType();
      if (propType.IsEmpty())
      {
        gdcmWarningMacro("Segmented property type not specified or incomplete");
      }

      // Segmented Property Type Code Sequence (0062,000F) Type 1
      SmartPointer<SequenceOfItems> propTypeSQ;
      const Tag propTypeSQTag(0x0062, 0x000F);
      if( !segmentDS.FindDataElement( propTypeSQTag ) )
      {
        propTypeSQ = new SequenceOfItems;
        DataElement detmp( propTypeSQTag );
        detmp.SetVR( VR::SQ );
        detmp.SetValue( *propTypeSQ );
        detmp.SetVLToUndefined();
        segmentDS.Insert( detmp );
      }
      propTypeSQ = segmentDS.GetDataElement( propTypeSQTag ).GetValueAsSQ();
      propTypeSQ->SetLengthToUndefined();

      // Fill the Segmented Property Type Code Sequence
      const size_t nbItems = propTypeSQ->GetNumberOfItems();
      if (nbItems < 1)  // Only one item is a type 1
      {
        Item item;
        item.SetVLToUndefined();
        propTypeSQ->AddItem(item);
      }

      Item &    propTypeItem = propTypeSQ->GetItem(1);
      DataSet & propTypeDS   = propTypeItem.GetNestedDataSet();

      //*****   CODE SEQUENCE MACRO ATTRIBUTES   *****//
      {
        // Code Value (Type 1)
        Attribute<0x0008, 0x0100> codeValueAt;
        codeValueAt.SetValue( propType.CV );
        propTypeDS.Replace( codeValueAt.GetAsDataElement() );

        // Coding Scheme (Type 1)
        Attribute<0x0008, 0x0102> codingSchemeAt;
        codingSchemeAt.SetValue( propType.CSD );
        propTypeDS.Replace( codingSchemeAt.GetAsDataElement() );

        // Code Meaning (Type 1)
        Attribute<0x0008, 0x0104> codeMeaningAt;
        codeMeaningAt.SetValue( propType.CM );
        propTypeDS.Replace( codeMeaningAt.GetAsDataElement() );
      }
    }

    //*****   Surface segmentation    *****//
    const unsigned long surfaceCount = segment->GetSurfaceCount();
    if (surfaceCount > 0)
    {
      // Surface Count
      Attribute<0x0066, 0x002A> surfaceCountAt;
      surfaceCountAt.SetValue( (unsigned int)surfaceCount );
      segmentDS.Replace( surfaceCountAt.GetAsDataElement() );

      //*****   Referenced Surface Sequence   *****//
      SmartPointer<SequenceOfItems> segmentsRefSQ;
      if( !segmentDS.FindDataElement( Tag(0x0066, 0x002B) ) )
      {
        segmentsRefSQ = new SequenceOfItems;
        DataElement detmp( Tag(0x0066, 0x002B) );
        detmp.SetVR( VR::SQ );
        detmp.SetValue( *segmentsRefSQ );
        detmp.SetVLToUndefined();
        segmentDS.Insert( detmp );
      }
      segmentsRefSQ = segmentDS.GetDataElement( Tag(0x0066, 0x002B) ).GetValueAsSQ();
      segmentsRefSQ->SetLengthToUndefined();

      // Fill the Segment Surface Generation Algorithm Identification Sequence
      const size_t nbItems        = segmentsRefSQ->GetNumberOfItems();
      if (nbItems < surfaceCount)
      {
        const size_t diff           = surfaceCount - nbItems;
        const size_t nbOfItemToMake = (diff > 0?diff:0);
        for(unsigned int i = 1; i <= nbOfItemToMake; ++i)
        {
          Item item;
          item.SetVLToUndefined();
          segmentsRefSQ->AddItem(item);
        }
      }
      // else Should I remove items?

      std::vector< SmartPointer< Surface > >                  surfaces          = segment->GetSurfaces();
      std::vector< SmartPointer< Surface > >::const_iterator  it                = surfaces.begin();
      std::vector< SmartPointer< Surface > >::const_iterator  itEnd             = surfaces.end();
      unsigned int                                            itemSurfaceNumber = 1;
      for (; it != itEnd; it++)
      {
        SmartPointer< Surface > surface = *it;

        Item &    segmentsRefItem = segmentsRefSQ->GetItem( itemSurfaceNumber++ );
        DataSet & segmentsRefDS   = segmentsRefItem.GetNestedDataSet();

        // Referenced Surface Number
        Attribute<0x0066, 0x002C> refSurfaceNumberAt;
        unsigned long refSurfaceNumber = surface->GetSurfaceNumber();
        if (refSurfaceNumber == 0)
        {
          refSurfaceNumber = surfaceNumber++;
          surface->SetSurfaceNumber( refSurfaceNumber );
        }
        refSurfaceNumberAt.SetValue( (unsigned int)refSurfaceNumber );
        segmentsRefDS.Replace( refSurfaceNumberAt.GetAsDataElement() );

        //*****   Segment Surface Source Instance Sequence   *****//
        {
//          SmartPointer<SequenceOfItems> surfaceSourceSQ;
//          if( !segmentsRefDS.FindDataElement( Tag(0x0066, 0x002E) ) )
//          {
//            surfaceSourceSQ = new SequenceOfItems;
//            DataElement detmp( Tag(0x0066, 0x002E) );
//            detmp.SetVR( VR::SQ );
//            detmp.SetValue( *surfaceSourceSQ );
//            detmp.SetVLToUndefined();
//            segmentsRefDS.Insert( detmp );
//          }
//          surfaceSourceSQ = segmentsRefDS.GetDataElement( Tag(0x0066, 0x002E) ).GetValueAsSQ();
//          surfaceSourceSQ->SetLengthToUndefined();

          //NOTE: If surfaces are derived from image, include 'Image SOP Instance Reference Macro' PS 3.3 Table C.10-3.
          //      How to know it?
        }
      }
    }
    else
    {
      // Segment Algorithm Name (Type 1)
      const char * segmentAlgorithmName = segment->GetSegmentAlgorithmName();
      if ( strcmp(segmentAlgorithmName, "") != 0 )
      {
        gdcmWarningMacro("No segment algorithm name specified");
      }
      Attribute<0x0062, 0x0009> segmentAlgorithmNameAt;
      segmentAlgorithmNameAt.SetValue( segmentAlgorithmName );
      segmentDS.Replace( segmentAlgorithmNameAt.GetAsDataElement() );
    }

    ++itemNumber;
  }

  return true;
}

bool SegmentWriter::Write()
{
  if( !PrepareWrite() )
  {
    return false;
  }

  assert( Stream );
  if( !Writer::Write() )
  {
    return false;
  }

  return true;
}

}
