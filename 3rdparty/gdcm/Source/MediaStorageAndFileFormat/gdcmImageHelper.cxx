/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageHelper.h"
#include "gdcmMediaStorage.h"
#include "gdcmFile.h"
#include "gdcmDataSet.h"
#include "gdcmDataElement.h"
#include "gdcmItem.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmGlobal.h"
#include "gdcmDictEntry.h"
#include "gdcmDicts.h"
#include "gdcmAttribute.h"
#include "gdcmImage.h"
#include "gdcmDirectionCosines.h"
#include "gdcmSegmentedPaletteColorLookupTable.h"
#include "gdcmByteValue.h"

#include <math.h> // fabs

  /* TODO:
   *
   * (0028,9145) SQ (Sequence with undefined length #=1)     # u/l, 1 PixelValueTransformationSequence
   * (fffe,e000) na (Item with undefined length #=3)         # u/l, 1 Item
   * (0028,1052) DS [0.00000]                                #   8, 1 RescaleIntercept
   * (0028,1053) DS [1.00000]                                #   8, 1 RescaleSlope
   * (0028,1054) LO [US]                                     #   2, 1 RescaleType
   * (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
   * (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
   *
   * Same goes for window level...
   */

namespace gdcm
{

bool ImageHelper::ForceRescaleInterceptSlope = false;
bool ImageHelper::ForcePixelSpacing = false;

bool GetOriginValueFromSequence(const DataSet& ds, const Tag& tfgs, std::vector<double> &ori)
{
  if( !ds.FindDataElement( tfgs ) ) return false;
  //const SequenceOfItems * sqi = ds.GetDataElement( tfgs ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
  assert( sqi );
  // Get first item:
  const Item &item = sqi->GetItem(1);
  const DataSet & subds = item.GetNestedDataSet();
  // Plane position Sequence
  const Tag tpms(0x0020,0x9113);
  if( !subds.FindDataElement(tpms) ) return false;
  //const SequenceOfItems * sqi2 = subds.GetDataElement( tpms ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi2 = subds.GetDataElement( tpms ).GetValueAsSQ();
  assert( sqi2 );
  const Item &item2 = sqi2->GetItem(1);
  const DataSet & subds2 = item2.GetNestedDataSet();
  //
  const Tag tps(0x0020,0x0032);
  if( !subds2.FindDataElement(tps) ) return false;
  const DataElement &de = subds2.GetDataElement( tps );
  //assert( bv );
  Attribute<0x0020,0x0032> at;
  at.SetFromDataElement( de );
  //at.Print( std::cout );
  ori.push_back( at.GetValue(0) );
  ori.push_back( at.GetValue(1) );
  ori.push_back( at.GetValue(2) );

  return true;
}

bool GetDirectionCosinesValueFromSequence(const DataSet& ds, const Tag& tfgs, std::vector<double> &dircos)
{
  if( !ds.FindDataElement( tfgs ) ) return false;
  //const SequenceOfItems * sqi = ds.GetDataElement( tfgs ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
  assert( sqi );
  // Get first item:
  const Item &item = sqi->GetItem(1);
  const DataSet & subds = item.GetNestedDataSet();
  // Plane position Sequence
  const Tag tpms(0x0020,0x9116);
  if( !subds.FindDataElement(tpms) ) return false;
  //const SequenceOfItems * sqi2 = subds.GetDataElement( tpms ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi2 = subds.GetDataElement( tpms ).GetValueAsSQ();
  assert( sqi2 && sqi2->GetNumberOfItems() );
  // Take it from the first item
  const Item &item2 = sqi2->GetItem(1);
  const DataSet & subds2 = item2.GetNestedDataSet();
  //
  const Tag tps(0x0020,0x0037);
  if( !subds2.FindDataElement(tps) ) return false;
  const DataElement &de = subds2.GetDataElement( tps );
  //assert( bv );
  Attribute<0x0020,0x0037> at;
  at.SetFromDataElement( de );
  dircos.push_back( at.GetValue(0) );
  dircos.push_back( at.GetValue(1) );
  dircos.push_back( at.GetValue(2) );
  dircos.push_back( at.GetValue(3) );
  dircos.push_back( at.GetValue(4) );
  dircos.push_back( at.GetValue(5) );

  return true;
}

bool GetInterceptSlopeValueFromSequence(const DataSet& ds, const Tag& tfgs, std::vector<double> &intslope)
{
  if( !ds.FindDataElement( tfgs ) ) return false;
  //const SequenceOfItems * sqi = ds.GetDataElement( tfgs ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
  assert( sqi );
  // Get first item:
  const Item &item = sqi->GetItem(1);
  const DataSet & subds = item.GetNestedDataSet();
  // (0028,9145) SQ (Sequence with undefined length)               # u/l,1 Pixel Value Transformation Sequence
  const Tag tpms(0x0028,0x9145);
  if( !subds.FindDataElement(tpms) ) return false;
  //const SequenceOfItems * sqi2 = subds.GetDataElement( tpms ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi2 = subds.GetDataElement( tpms ).GetValueAsSQ();
  assert( sqi2 );
  const Item &item2 = sqi2->GetItem(1);
  const DataSet & subds2 = item2.GetNestedDataSet();
    {
    //  (0028,1052) DS [0]                                        # 2,1 Rescale Intercept
    const Tag tps(0x0028,0x1052);
    if( !subds2.FindDataElement(tps) ) return false;
    const DataElement &de = subds2.GetDataElement( tps );
    //assert( bv );
    Attribute<0x0028,0x1052> at;
    at.SetFromDataElement( de );
    //at.Print( std::cout );
    intslope.push_back( at.GetValue() );
    }
    {
    // (0028,1053) DS [5.65470085470085]                         # 16,1 Rescale Slope
    const Tag tps(0x0028,0x1053);
    if( !subds2.FindDataElement(tps) ) return false;
    const DataElement &de = subds2.GetDataElement( tps );
    //assert( bv );
    Attribute<0x0028,0x1053> at;
    at.SetFromDataElement( de );
    //at.Print( std::cout );
    intslope.push_back( at.GetValue() );
    }

  assert( intslope.size() == 2 );
  return true;
}

bool ComputeZSpacingFromIPP(const DataSet &ds, double &zspacing)
{
  // first we need to get the direction cosines:
  const Tag t1(0x5200,0x9229);
  const Tag t2(0x5200,0x9230);
  std::vector<double> cosines;
  // For some reason TOSHIBA-EnhancedCT.dcm is storing the direction cosines in the per-frame section
  // and not the shared one... oh well
  bool b1 = GetDirectionCosinesValueFromSequence(ds, t1, cosines)
    || GetDirectionCosinesValueFromSequence(ds,t2,cosines);
  if(!b1)
    {
    cosines.resize( 6 );
    bool b2 = ImageHelper::GetDirectionCosinesFromDataSet(ds, cosines);
    if( b2 )
      {
      gdcmWarningMacro( "Image Orientation (Patient) cannot be stored here!. Continuing" );
      b1 = b2;
      }
    else
      {
      gdcmErrorMacro( "Image Orientation (Patient) was not found" );
      cosines[0] = 1;
      cosines[1] = 0;
      cosines[2] = 0;
      cosines[3] = 0;
      cosines[4] = 1;
      cosines[5] = 0;
      }
    }
  assert( b1 && cosines.size() == 6 ); // yeah we really need that

  const Tag tfgs(0x5200,0x9230);
  if( !ds.FindDataElement( tfgs ) ) return false;
  //const SequenceOfItems * sqi = ds.GetDataElement( tfgs ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
  assert( sqi );
  double normal[3];
  DirectionCosines dc( &cosines[0] );
  dc.Cross( normal );

  // For each item
  std::vector<double> distances;
  SequenceOfItems::SizeType nitems = sqi->GetNumberOfItems();
  std::vector<double> dircos_subds2; dircos_subds2.resize(6);
  for(SequenceOfItems::SizeType i0 = 1; i0 <= nitems; ++i0)
    {
    const Item &item = sqi->GetItem(i0);
    const DataSet & subds = item.GetNestedDataSet();
    // (0020,9113) SQ (Sequence with undefined length #=1)     # u/l, 1 PlanePositionSequence
    const Tag tpms(0x0020,0x9113);
    if( !subds.FindDataElement(tpms) ) return false;
    //const SequenceOfItems * sqi2 = subds.GetDataElement( tpms ).GetSequenceOfItems();
    SmartPointer<SequenceOfItems> sqi2 = subds.GetDataElement( tpms ).GetValueAsSQ();
    assert( sqi2 );
    const Item &item2 = sqi2->GetItem(1);
    const DataSet & subds2 = item2.GetNestedDataSet();
    // Check Image Orientation (Patient)
    if( ImageHelper::GetDirectionCosinesFromDataSet(subds2, dircos_subds2) )
      {
      assert( dircos_subds2 == cosines );
      }
    // (0020,0032) DS [-82.5\-82.5\1153.75]                    #  20, 3 ImagePositionPatient
    const Tag tps(0x0020,0x0032);
    if( !subds2.FindDataElement(tps) ) return false;
    const DataElement &de = subds2.GetDataElement( tps );
    Attribute<0x0020,0x0032> ipp;
    ipp.SetFromDataElement(de);
    double dist = 0;
    for (int i = 0; i < 3; ++i) dist += normal[i]*ipp[i];
    distances.push_back( dist );
    }
  assert( distances.size() == nitems );
  double meanspacing = 0;
  double prev = distances[0];
  for(unsigned int i = 1; i < nitems; ++i)
    {
    const double current = distances[i] - prev;
    meanspacing += current;
    prev = distances[i];
    }
  bool timeseries = false;
  if( nitems > 1 )
    {
    meanspacing /= (double)(nitems - 1);
    if( meanspacing == 0.0 )
      {
      // Could be a time series. Assume time spacing of 1. for now:
      gdcmDebugMacro( "Assuming time series for Z-spacing" );
      meanspacing = 1.0;
      timeseries = true;
      }
    }

  zspacing = meanspacing;
  assert( zspacing != 0.0 ); // technically this should not happen

  if( !timeseries )
    {
    // Check spacing is consistent:
    const double ZTolerance = 1e-3; // ??? FIXME
    prev = distances[0];
    for(unsigned int i = 1; i < nitems; ++i)
      {
      const double current = distances[i] - prev;
      if( fabs(current - zspacing) > ZTolerance )
        {
        // For now simply gives up
        gdcmErrorMacro( "This Enhanced Multiframe is not supported for now. Sorry" );
        return false;
        }
      prev = distances[i];
      }
    }
  return true;
}

// EnhancedMRImageStorage & EnhancedCTImageStorage
bool GetSpacingValueFromSequence(const DataSet& ds, const Tag& tfgs, std::vector<double> &sp)
{
  //  (0028,9110) SQ (Sequence with undefined length #=1)     # u/l, 1 PixelMeasuresSequence
  //      (fffe,e000) na (Item with undefined length #=2)         # u/l, 1 Item
  //        (0018,0050) DS [0.5]                                    #   4, 1 SliceThickness
  //        (0028,0030) DS [0.322\0.322]                            #  12, 2 PixelSpacing
  // <entry group="5200" element="9229" vr="SQ" vm="1" name="Shared Functional Groups Sequence"/>
  //const Tag tfgs(0x5200,0x9229);
  //const Tag tfgs(0x5200,0x9230);
  //assert( ds.FindDataElement( tfgs ) );
  if( !ds.FindDataElement( tfgs ) ) return false;
  //const SequenceOfItems * sqi = ds.GetDataElement( tfgs ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
  assert( sqi );
  // Get first item:
  const Item &item = sqi->GetItem(1);
  const DataSet & subds = item.GetNestedDataSet();
  // <entry group="0028" element="9110" vr="SQ" vm="1" name="Pixel Measures Sequence"/>
  const Tag tpms(0x0028,0x9110);
  if( !subds.FindDataElement(tpms) ) return false;
  //const SequenceOfItems * sqi2 = subds.GetDataElement( tpms ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi2 = subds.GetDataElement( tpms ).GetValueAsSQ();
  assert( sqi2 );
  const Item &item2 = sqi2->GetItem(1);
  const DataSet & subds2 = item2.GetNestedDataSet();
  // <entry group="0028" element="0030" vr="DS" vm="2" name="Pixel Spacing"/>
  const Tag tps(0x0028,0x0030);
  if( !subds2.FindDataElement(tps) ) return false;
  const DataElement &de = subds2.GetDataElement( tps );
  //assert( bv );
  Attribute<0x0028,0x0030> at;
  at.SetFromDataElement( de );
  //at.Print( std::cout );
  sp.push_back( at.GetValue(1) );
  sp.push_back( at.GetValue(0) );

  // BUG ! Check for instace:
  // gdcmData/BRTUM001.dcm
  // Slice Thickness is 5.0 while the Zspacing should be 6.0 !
#if 0
  // Do the 3rd dimension zspacing:
  // <entry group="0018" element="0050" vr="DS" vm="1" name="Slice Thickness"/>
  const Tag tst(0x0018,0x0050);
  if( !subds2.FindDataElement(tst) ) return false;
  const DataElement &de2 = subds2.GetDataElement( tst );
  Attribute<0x0018,0x0050> at2;
  at2.SetFromDataElement( de2 );
  //at2.Print( std::cout );
  sp.push_back( at2.GetValue(0) );
#endif
  double zspacing;
  bool b = ComputeZSpacingFromIPP(ds, zspacing);
  if( !b ) return false;

  sp.push_back( zspacing );

  return true;
}

// UltrasoundMultiframeImageStorage
bool GetUltraSoundSpacingValueFromSequence(const DataSet& ds, std::vector<double> &sp)
{
/*
(0018,6011) SQ (Sequence with explicit length #=1)      # 196, 1 SequenceOfUltrasoundRegions
  (fffe,e000) na (Item with explicit length #=15)         # 188, 1 Item
    (0018,6012) US 1                                        #   2, 1 RegionSpatialFormat
    (0018,6014) US 1                                        #   2, 1 RegionDataType
    (0018,6016) UL 0                                        #   4, 1 RegionFlags
    (0018,6018) UL 0                                        #   4, 1 RegionLocationMinX0
    (0018,601a) UL 0                                        #   4, 1 RegionLocationMinY0
    (0018,601c) UL 479                                      #   4, 1 RegionLocationMaxX1
    (0018,601e) UL 479                                      #   4, 1 RegionLocationMaxY1
    (0018,6020) SL 0                                        #   4, 1 ReferencePixelX0
    (0018,6022) SL 0                                        #   4, 1 ReferencePixelY0
    (0018,6024) US 3                                        #   2, 1 PhysicalUnitsXDirection
    (0018,6026) US 3                                        #   2, 1 PhysicalUnitsYDirection
    (0018,6028) FD 0                                        #   8, 1 ReferencePixelPhysicalValueX
    (0018,602a) FD 0                                        #   8, 1 ReferencePixelPhysicalValueY
    (0018,602c) FD 0.002                                    #   8, 1 PhysicalDeltaX
    (0018,602e) FD 0.002                                    #   8, 1 PhysicalDeltaY
  (fffe,e00d) na (ItemDelimitationItem for re-encoding)   #   0, 0 ItemDelimitationItem
(fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
*/
  const Tag tsqusreg(0x0018,0x6011);
  if( !ds.FindDataElement( tsqusreg ) ) return false;
  //const SequenceOfItems * sqi = ds.GetDataElement( tsqusreg ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( tsqusreg ).GetValueAsSQ();
  assert( sqi );
  // Get first item:
  const Item &item = sqi->GetItem(1);
  const DataSet & subds = item.GetNestedDataSet();
  //  (0018,602c) FD 0.002                                    #   8, 1 PhysicalDeltaX
  //  (0018,602e) FD 0.002                                    #   8, 1 PhysicalDeltaY
  Attribute<0x0018,0x602c> at1;
  Attribute<0x0018,0x602e> at2;
  const DataElement &de1 = subds.GetDataElement( at1.GetTag() );
  at1.SetFromDataElement( de1 );
  assert( at1.GetNumberOfValues() == 1 );
  const DataElement &de2 = subds.GetDataElement( at2.GetTag() );
  at2.SetFromDataElement( de2 );
  assert( at2.GetNumberOfValues() == 1 );
  sp.push_back( at1.GetValue() );
  sp.push_back( at2.GetValue() );

  return true;
}



/* Enhanced stuff looks like:

    (0020,9113) SQ (Sequence with undefined length #=1)     # u/l, 1 PlanePositionSequence
      (fffe,e000) na (Item with undefined length #=1)         # u/l, 1 Item
        (0020,0032) DS [73.5100815890831\-129.65028828174\189.777023529388] #  50, 3 ImagePositionPatient
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
    (0020,9116) SQ (Sequence with undefined length #=1)     # u/l, 1 PlaneOrientationSequence
      (fffe,e000) na (Item with undefined length #=1)         # u/l, 1 Item
        (0020,0037) DS [0.01604138687252\0.99942564964294\-0.0298495516180\0.0060454937629... # 102, 6 ImageOrientationPatient
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
    (0028,9110) SQ (Sequence with undefined length #=1)     # u/l, 1 PixelMeasuresSequence
      (fffe,e000) na (Item with undefined length #=2)         # u/l, 1 Item
        (0018,0050) DS [1]                                      #   2, 1 SliceThickness
        (0028,0030) DS [0.83333331346511\0.83333331346511]      #  34, 2 PixelSpacing
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/

std::vector<double> ImageHelper::GetOriginValue(File const & f)
{
  std::vector<double> ori;
  MediaStorage ms;
  ms.SetFromFile(f);
  const DataSet& ds = f.GetDataSet();

  if( ms == MediaStorage::EnhancedCTImageStorage
   || ms == MediaStorage::EnhancedMRImageStorage
   || ms == MediaStorage::OphthalmicTomographyImageStorage
   || ms == MediaStorage::SegmentationStorage )
    {
    const Tag t1(0x5200,0x9229);
    const Tag t2(0x5200,0x9230);
    if( GetOriginValueFromSequence(ds,t1, ori)
     || GetOriginValueFromSequence(ds, t2, ori) )
      {
      assert( ori.size() == 3 );
      return ori;
      }
    ori.resize( 3 );
    gdcmWarningMacro( "Could not find Origin" );
    return ori;
    }
  if( ms == MediaStorage::NuclearMedicineImageStorage )
    {
    const Tag t1(0x0054,0x0022);
    if( ds.FindDataElement( t1 ) )
      {
      SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( t1 ).GetValueAsSQ();
      if( sqi && sqi->GetNumberOfItems() >= 1)
        {
        // Get first item:
        const Item &item = sqi->GetItem(1);
        const DataSet & subds = item.GetNestedDataSet();
        const Tag timagepositionpatient(0x0020, 0x0032);
        assert( subds.FindDataElement( timagepositionpatient ) );
        Attribute<0x0020,0x0032> at = {{0,0,0}}; // default value if empty
        at.SetFromDataSet( subds );
        ori.resize( at.GetNumberOfValues() );
        for( unsigned int i = 0; i < at.GetNumberOfValues(); ++i )
          {
          ori[i] = at.GetValue(i);
          }
        return ori;
        }
      }
    ori.resize( 3 );
    gdcmWarningMacro( "Could not find Origin" );
    return ori;
    }
  ori.resize( 3 );

  // else
  const Tag timagepositionpatient(0x0020, 0x0032);
  if( ms != MediaStorage::SecondaryCaptureImageStorage && ds.FindDataElement( timagepositionpatient ) )
    {
    const DataElement& de = ds.GetDataElement( timagepositionpatient );
    Attribute<0x0020,0x0032> at = {{0,0,0}}; // default value if empty
    at.SetFromDataElement( de );
    for( unsigned int i = 0; i < at.GetNumberOfValues(); ++i )
      {
      ori[i] = at.GetValue(i);
      }
    }
  else
    {
    ori[0] = 0;
    ori[1] = 0;
    ori[2] = 0;
    }
  assert( ori.size() == 3 );
  return ori;
}

bool ImageHelper::GetDirectionCosinesFromDataSet(DataSet const & ds, std::vector<double> & dircos)
{
  // \precondition: this dataset is not a secondary capture !
  // else
  const Tag timageorientationpatient(0x0020, 0x0037);
  if( ds.FindDataElement( timageorientationpatient ) /*&& !ds.GetDataElement( timageorientationpatient ).IsEmpty()*/ )
    {
    const DataElement& de = ds.GetDataElement( timageorientationpatient );
    Attribute<0x0020,0x0037> at = {{1,0,0,0,1,0}}; // default value if empty
    at.SetFromDataElement( de );
    for( unsigned int i = 0; i < at.GetNumberOfValues(); ++i )
      {
      dircos[i] = at.GetValue(i);
      }
    DirectionCosines dc( &dircos[0] );
    if( !dc.IsValid() )
      {
      dc.Normalize();
      if( dc.IsValid() )
        {
        gdcmWarningMacro( "DirectionCosines was not normalized. Fixed" );
        const double * p = dc;
        dircos = std::vector<double>(p, p + 6);
        //return dircos;
        }
      else
        {
        // PAPYRUS_CR_InvalidIOP.dcm
        gdcmWarningMacro( "Could not get DirectionCosines. Will be set to unit vector." );
        //dircos[0] = 1;
        //dircos[1] = 0;
        //dircos[2] = 0;
        //dircos[3] = 0;
        //dircos[4] = 1;
        //dircos[5] = 0;
        return false;
        }
      }
    return true;
    }
  return false;
}

std::vector<double> ImageHelper::GetDirectionCosinesValue(File const & f)
{
  std::vector<double> dircos;
  MediaStorage ms;
  ms.SetFromFile(f);
  const DataSet& ds = f.GetDataSet();

  if( ms == MediaStorage::EnhancedCTImageStorage
   || ms == MediaStorage::EnhancedMRImageStorage
   || ms == MediaStorage::SegmentationStorage )
    {
    const Tag t1(0x5200,0x9229);
    const Tag t2(0x5200,0x9230);
    if( GetDirectionCosinesValueFromSequence(ds,t1, dircos)
     || GetDirectionCosinesValueFromSequence(ds, t2, dircos) )
      {
      assert( dircos.size() == 6 );
      return dircos;
      }
    else
      {
      dircos.resize( 6 );
      bool b2 = ImageHelper::GetDirectionCosinesFromDataSet(ds, dircos);
      if( b2 )
        {
        gdcmWarningMacro( "Image Orientation (Patient) cannot be stored here!. Continuing" );
        }
      else
        {
        gdcmErrorMacro( "Image Orientation (Patient) was not found" );
        dircos[0] = 1;
        dircos[1] = 0;
        dircos[2] = 0;
        dircos[3] = 0;
        dircos[4] = 1;
        dircos[5] = 0;
        }
      return dircos;
      }
    }
  if( ms == MediaStorage::NuclearMedicineImageStorage )
    {
    const Tag t1(0x0054,0x0022);
    if( ds.FindDataElement( t1 ) )
      {
      SmartPointer<SequenceOfItems> sqi = ds.GetDataElement( t1 ).GetValueAsSQ();
      if( sqi && sqi->GetNumberOfItems() >= 1 )
        {
        // Get first item:
        const Item &item = sqi->GetItem(1);
        const DataSet & subds = item.GetNestedDataSet();

        dircos.resize( 6 );
        bool b2 = ImageHelper::GetDirectionCosinesFromDataSet(subds, dircos);
        if( b2 )
          {
          }
        else
          {
          gdcmErrorMacro( "Image Orientation (Patient) was not found" );
          dircos[0] = 1;
          dircos[1] = 0;
          dircos[2] = 0;
          dircos[3] = 0;
          dircos[4] = 1;
          dircos[5] = 0;
          }
        return dircos;
        }
      }
    }

  dircos.resize( 6 );
  if( ms == MediaStorage::SecondaryCaptureImageStorage || !GetDirectionCosinesFromDataSet(ds, dircos) )
    {
    dircos[0] = 1;
    dircos[1] = 0;
    dircos[2] = 0;
    dircos[3] = 0;
    dircos[4] = 1;
    dircos[5] = 0;
    }

  assert( dircos.size() == 6 );
  return dircos;
}

void ImageHelper::SetForceRescaleInterceptSlope(bool b)
{
  ForceRescaleInterceptSlope = b;
}

bool ImageHelper::GetForceRescaleInterceptSlope()
{
  return ForceRescaleInterceptSlope;
}

void ImageHelper::SetForcePixelSpacing(bool b)
{
  ForcePixelSpacing = b;
}

bool ImageHelper::GetForcePixelSpacing()
{
  return ForcePixelSpacing;
}

bool GetRescaleInterceptSlopeValueFromDataSet(const DataSet& ds, std::vector<double> & interceptslope)
{
  Attribute<0x0028,0x1052> at1;
  bool intercept = ds.FindDataElement(at1.GetTag());
  if( intercept )
    {
    if( !ds.GetDataElement(at1.GetTag()).IsEmpty() )
      {
      at1.SetFromDataElement( ds.GetDataElement(at1.GetTag()) );
      interceptslope[0] = at1.GetValue();
      }
    }
  Attribute<0x0028,0x1053> at2;
  bool slope = ds.FindDataElement(at2.GetTag());
  if ( slope )
    {
    if( !ds.GetDataElement(at2.GetTag()).IsEmpty() )
      {
      at2.SetFromDataElement( ds.GetDataElement(at2.GetTag()) );
      interceptslope[1] = at2.GetValue();
      if( interceptslope[1] == 0 )
        {
        // come' on ! WTF
        gdcmDebugMacro( "Cannot have slope == 0. Defaulting to 1.0 instead" );
        interceptslope[1] = 1;
        }
      }
    }
  return true;
}


/// This function returns pixel information about an image from its dataset
/// That includes samples per pixel and bit depth (in that order)
/// Returns a PixelFormat
PixelFormat ImageHelper::GetPixelFormatValue(const File& f)
{
  // D 0028|0011 [US] [Columns] [512]
  //[10/20/10 9:05:07 AM] Mathieu Malaterre:
  PixelFormat pf;
  const DataSet& ds = f.GetDataSet();
  // D 0028|0100 [US] [Bits Allocated] [16]
  {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0100) );
    Attribute<0x0028,0x0100> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetBitsAllocated( at.GetValue() );
  }
  // D 0028|0101 [US] [Bits Stored] [12]
  {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0101) );
    Attribute<0x0028,0x0101> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetBitsStored( at.GetValue() );
  }
  // D 0028|0102 [US] [High Bit] [11]
  {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0102) );
    Attribute<0x0028,0x0102> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetHighBit( at.GetValue() );
  }
  // D 0028|0103 [US] [Pixel Representation] [0]
  {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0103) );
    Attribute<0x0028,0x0103> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetPixelRepresentation( at.GetValue() );
  }
  // (0028,0002) US 1                                        #   2, 1 SamplesPerPixel
  {
  //if( ds.FindDataElement( Tag(0x0028, 0x0002) ) )
  {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0002) );
    Attribute<0x0028,0x0002> at = { 1 };
    at.SetFromDataSet( ds );
    pf.SetSamplesPerPixel( at.GetValue() );
  }
  // else pf will default to 1...
  }
  return pf;

}
/// This function checks tags (0x0028, 0x0010) and (0x0028, 0x0011) for the
/// rows and columns of the image in pixels (as opposed to actual distances).
/// Also checks 0054, 0081 for the number of z slices for a 3D image
/// If that tag is not present, default the z dimension to 1
std::vector<unsigned int> ImageHelper::GetDimensionsValue(const File& f)
{
  DataSet const & ds = f.GetDataSet();

  MediaStorage ms;
  ms.SetFromFile(f);
  std::vector<unsigned int> theReturn(3);
#if 0
  if( ms == MediaStorage::VLWholeSlideMicroscopyImageStorage )
    {
      {
      Attribute<0x0048,0x0006> at = { 0 };
      at.SetFromDataSet( ds );
      theReturn[0] = at.GetValue();
      }
      {
      Attribute<0x0048,0x0007> at = { 0 };
      at.SetFromDataSet( ds );
      theReturn[1] = at.GetValue();
      }
    theReturn[2] = 1;
    }
  else
#endif
    {
      {
      Attribute<0x0028,0x0011> at = { 0 };
      at.SetFromDataSet( ds );
      theReturn[0] = at.GetValue();
      }
      {
      Attribute<0x0028,0x0010> at = { 0 };
      at.SetFromDataSet( ds );
      theReturn[1] = at.GetValue();
      }
      {
      Attribute<0x0028,0x0008> at = { 0 };
      at.SetFromDataSet( ds );
      int numberofframes = at.GetValue();
      theReturn[2] = 1;
      if( numberofframes > 1 )
        {
        theReturn[2] = at.GetValue();
        }
      }
    // ACR-NEMA legacy
      {
      Attribute<0x0028,0x0005> at = { 0 };
      if( ds.FindDataElement( at.GetTag() ) )
        {
        const DataElement &de = ds.GetDataElement( at.GetTag() );
        // SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm picks VR::SS instead...
        if( at.GetVR().Compatible( de.GetVR() ) )
          {
          at.SetFromDataSet( ds );
          int imagedimensions = at.GetValue();
          if( imagedimensions == 3 )
            {
            Attribute<0x0028,0x0012> at2 = { 0 };
            at2.SetFromDataSet( ds );
            theReturn[2] = at2.GetValue();
            }
          }
        else
          {
          gdcmWarningMacro( "Sorry cant read attribute (wrong VR): " << at.GetTag() );
          }
        }
      }
    }

  return theReturn;
}

void ImageHelper::SetDimensionsValue(File& f, const Image & img)
{
  const unsigned int *dims = img.GetDimensions();
  MediaStorage ms;
  ms.SetFromFile(f);
  DataSet& ds = f.GetDataSet();
  assert( MediaStorage::IsImage( ms ) );
#if 0
  if( ms == MediaStorage::VLWholeSlideMicroscopyImageStorage )
    {
    Attribute<0x0048,0x0006> columns;
    columns.SetValue( dims[0] );
    ds.Replace( columns.GetAsDataElement() );
    Attribute<0x0048,0x0007> rows;
    rows.SetValue( dims[1] );
    ds.Replace( rows.GetAsDataElement() );
    if( dims[2] > 1 )
      {
      assert( 0 );
      }
    }
  else
#endif
    {
    Attribute<0x0028,0x0010> rows;
    rows.SetValue( (uint16_t)dims[1] );
    ds.Replace( rows.GetAsDataElement() );
    Attribute<0x0028,0x0011> columns;
    columns.SetValue( (uint16_t)dims[0] );
    ds.Replace( columns.GetAsDataElement() );
    if( dims[2] > 1 )
      {
      Attribute<0x0028,0x0008> numframes = { 0 };
      ds.Replace( numframes.GetAsDataElement() );
      }
    }
}

std::vector<double> ImageHelper::GetRescaleInterceptSlopeValue(File const & f)
{
  std::vector<double> interceptslope;
  MediaStorage ms;
  ms.SetFromFile(f);
  const DataSet& ds = f.GetDataSet();

  if( ms == MediaStorage::EnhancedCTImageStorage
   || ms == MediaStorage::EnhancedMRImageStorage
   || ms == MediaStorage::SegmentationStorage )
    {
    const Tag t1(0x5200,0x9229);
    const Tag t2(0x5200,0x9230);
    if( GetInterceptSlopeValueFromSequence(ds,t1, interceptslope)
     || GetInterceptSlopeValueFromSequence(ds,t2, interceptslope) )
      {
      assert( interceptslope.size() == 2 );
      return interceptslope;
      }
    else
      {
      interceptslope.resize( 2 );
      interceptslope[0] = 0;
      interceptslope[1] = 1;
      bool b = GetRescaleInterceptSlopeValueFromDataSet(ds, interceptslope);
      gdcmAssertMacro( b ); (void)b;
      return interceptslope;
      }
    }

  // else
  interceptslope.resize( 2 );
  interceptslope[0] = 0;
  interceptslope[1] = 1;
  if( ms == MediaStorage::CTImageStorage
 || ms == MediaStorage::ComputedRadiographyImageStorage
 /*|| ms == MediaStorage::MRImageStorage // FIXME ! */
 || ms == MediaStorage::PETImageStorage
 || ms == MediaStorage::SecondaryCaptureImageStorage
 || ms == MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage
 || ms == MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage
 || ForceRescaleInterceptSlope
  )
    {
    bool b = GetRescaleInterceptSlopeValueFromDataSet(ds, interceptslope);
    gdcmAssertMacro( b ); (void)b;
    }
  else if (
    ms == MediaStorage::RTDoseStorage
  )
    {
    // TODO. Should I check FrameIncrementPointer ? (0028,0009) AT (3004,000c)
    Attribute<0x3004,0x000e> gridscaling = { 0 };
    gridscaling.SetFromDataSet( ds );
    interceptslope[0] = 0;
    interceptslope[1] = gridscaling.GetValue();
    if( interceptslope[1] == 0 )
      {
      // come' on ! WTF
      gdcmWarningMacro( "Cannot have slope == 0. Defaulting to 1.0 instead" );
      interceptslope[1] = 1;
      }
    }

  // \post condition slope can never be 0:
  assert( interceptslope[1] != 0. );
  return interceptslope;
}

Tag ImageHelper::GetSpacingTagFromMediaStorage(MediaStorage const &ms)
{
  Tag t;

  // gdcmData/MR00010001.dcm => GeneralElectricMagneticResonanceImageStorage
  // (0018,0088) DS []                                       #   4, 1 SpacingBetweenSlices
  // (0028,0030) DS [ 0.8593750000\0.8593750000]             #  26, 2 PixelSpacing
  switch(ms)
    {
  case MediaStorage::EnhancedUSVolumeStorage:
  // Enhanced stuff are handled elsewere... look carefully :)
  //case MediaStorage::EnhancedMRImageStorage:
  //case MediaStorage::EnhancedCTImageStorage:
  //  gdcmWarningMacro( "Enhanced image are not currently supported. Spacing will be wrong" );
  case MediaStorage::CTImageStorage:
  case MediaStorage::MRImageStorage:
  case MediaStorage::RTDoseStorage:
  case MediaStorage::NuclearMedicineImageStorage:
  case MediaStorage::PETImageStorage:
  case MediaStorage::GeneralElectricMagneticResonanceImageStorage:
  case MediaStorage::PhilipsPrivateMRSyntheticImageStorage:
  case MediaStorage::VLPhotographicImageStorage: // VL Image IOD
  case MediaStorage::VLMicroscopicImageStorage:
    // (0028,0030) DS [2.0\2.0]                                #   8, 2 PixelSpacing
    t = Tag(0x0028,0x0030);
    break;
  case MediaStorage::ComputedRadiographyImageStorage: // See pixelspacingtestimages/DISCIMG/IMAGES/CRIMAGE
  case MediaStorage::DigitalXRayImageStorageForPresentation:
  case MediaStorage::DigitalXRayImageStorageForProcessing:
  case MediaStorage::DigitalMammographyImageStorageForPresentation:
  case MediaStorage::DigitalMammographyImageStorageForProcessing:
  case MediaStorage::DigitalIntraoralXrayImageStorageForPresentation:
  case MediaStorage::DigitalIntraoralXRayImageStorageForProcessing:
  case MediaStorage::XRayAngiographicImageStorage:
  case MediaStorage::XRayRadiofluoroscopingImageStorage:
  case MediaStorage::XRayAngiographicBiPlaneImageStorageRetired:
    // (0018,1164) DS [0.5\0.5]                                #   8, 2 ImagerPixelSpacing
    t = Tag(0x0018,0x1164);
    break;
  case MediaStorage::RTImageStorage: // gdcmData/SYNGORTImage.dcm
    t = Tag(0x3002,0x0011); // ImagePlanePixelSpacing
    break;
  case MediaStorage::SecondaryCaptureImageStorage:
  case MediaStorage::MultiframeSingleBitSecondaryCaptureImageStorage:
  case MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage:
  case MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage:
  case MediaStorage::MultiframeTrueColorSecondaryCaptureImageStorage:
    // See PS 3.3-2008. Table C.8-25 SC IMAGE MODULE ATTRIBUTES
    // and Table C.8-25b SC MULTI-FRAME IMAGE MODULE ATTRIBUTES
    t = Tag(0x0018,0x2010);
    break;
  case MediaStorage::HardcopyGrayscaleImageStorage:
    t = Tag(0xffff,0xffff);
    break;
  case MediaStorage::GEPrivate3DModelStorage: // FIXME FIXME !!!
  case MediaStorage::Philips3D:
  case MediaStorage::VideoEndoscopicImageStorage:
    gdcmWarningMacro( "FIXME" );
    t = Tag(0xffff,0xffff);
    break;
  case MediaStorage::UltrasoundMultiFrameImageStorage:
    // gdcmData/US-MONO2-8-8x-execho.dcm
    // this should be handled somewhere else
    //assert(0);
    gdcmWarningMacro( "FIXME" );
    t = Tag(0xffff,0xffff);
    break;
  case MediaStorage::UltrasoundImageStorage: // ??
  case MediaStorage::UltrasoundImageStorageRetired:
  case MediaStorage::UltrasoundMultiFrameImageStorageRetired:
    // (0028,0034) IS [4\3]                                    #   4, 2 PixelAspectRatio
    t = Tag(0xffff,0xffff); // FIXME
    t = Tag(0x0028,0x0034); // FIXME
    break;
  default:
    gdcmDebugMacro( "Do not handle: " << ms );
    t = Tag(0xffff,0xffff);
    break;
    }

  // should only override unless Modality set it already
  // basically only Secondary Capture should reach that point
  if( ForcePixelSpacing && t == Tag(0xffff,0xffff) )
    {
    t = Tag(0x0028,0x0030);
    }

  return t;
}

Tag ImageHelper::GetZSpacingTagFromMediaStorage(MediaStorage const &ms)
{
  Tag t;

  switch(ms)
    {
  case MediaStorage::EnhancedUSVolumeStorage:
  case MediaStorage::MRImageStorage:
  case MediaStorage::NuclearMedicineImageStorage: // gdcmData/Nm.dcm
  case MediaStorage::GeneralElectricMagneticResonanceImageStorage:
    // (0018,0088) DS [3]                                      #   2, 1 SpacingBetweenSlices
    t = Tag(0x0018,0x0088);
    break;
  // No spacing AFAIK for those:
/*
$ dciodvfy gdcmData/D_CLUNIE_CT1_JPLL.dcm
CTImage
Warning - Attribute is not present in standard DICOM IOD - (0x0018,0x0088) DS Spacing Between Slices
Warning - Dicom dataset contains attributes not present in standard DICOM IOD - this is a Standard Extended SOP Class
*/
  case MediaStorage::PETImageStorage: // ??
  case MediaStorage::CTImageStorage:
  case MediaStorage::RTImageStorage:
  // ImagerPixelSpacing section:
  case MediaStorage::ComputedRadiographyImageStorage: // ??
  case MediaStorage::DigitalXRayImageStorageForPresentation:
  case MediaStorage::DigitalXRayImageStorageForProcessing:
  case MediaStorage::DigitalMammographyImageStorageForPresentation:
  case MediaStorage::DigitalMammographyImageStorageForProcessing:
  case MediaStorage::DigitalIntraoralXrayImageStorageForPresentation:
  case MediaStorage::DigitalIntraoralXRayImageStorageForProcessing:
  case MediaStorage::XRayAngiographicImageStorage:
  case MediaStorage::XRayRadiofluoroscopingImageStorage:
  case MediaStorage::XRayAngiographicBiPlaneImageStorageRetired:
  // US:
  case MediaStorage::UltrasoundImageStorage:
  case MediaStorage::UltrasoundMultiFrameImageStorage:
  case MediaStorage::UltrasoundImageStorageRetired:
  case MediaStorage::UltrasoundMultiFrameImageStorageRetired:
  // SC:
  case MediaStorage::SecondaryCaptureImageStorage:
  case MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage:
  case MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage:
  case MediaStorage::HardcopyGrayscaleImageStorage:
    t = Tag(0xffff,0xffff);
    break;
  case MediaStorage::RTDoseStorage: // gdcmData/BogugsItemAndSequenceLengthCorrected.dcm
    t = Tag(0x3004,0x000c);
    break;
  case MediaStorage::GEPrivate3DModelStorage:
  case MediaStorage::Philips3D:
  case MediaStorage::VideoEndoscopicImageStorage:
    gdcmWarningMacro( "FIXME" );
    t = Tag(0xffff,0xffff);
    break;
  default:
    gdcmDebugMacro( "Do not handle Z spacing for: " << ms );
    t = Tag(0xffff,0xffff);
    break;
    }

  if( ForcePixelSpacing && t == Tag(0xffff,0xffff) )
    {
    t = Tag(0x0018,0x0088);
    }
  return t;
}

std::vector<double> ImageHelper::GetSpacingValue(File const & f)
{
  std::vector<double> sp;
  sp.reserve(3);
  MediaStorage ms;
  ms.SetFromFile(f);
  const DataSet& ds = f.GetDataSet();

  if( ms == MediaStorage::EnhancedCTImageStorage
    || ms == MediaStorage::EnhancedMRImageStorage
    || ms == MediaStorage::OphthalmicTomographyImageStorage
    || ms == MediaStorage::SegmentationStorage )
    {
    // <entry group="5200" element="9230" vr="SQ" vm="1" name="Per-frame Functional Groups Sequence"/>
    const Tag t1(0x5200,0x9229);
    const Tag t2(0x5200,0x9230);
    if( GetSpacingValueFromSequence(ds,t1, sp)
      || GetSpacingValueFromSequence(ds, t2, sp) )
      {
      assert( sp.size() == 3 );
      return sp;
      }
    // Else.
    // How do I send an error ?
    sp.resize( 3 ); // FIXME !!
    sp[2] = 1.;
    gdcmWarningMacro( "Could not find Spacing" );
    return sp;
    }
  else if( ms == MediaStorage::UltrasoundMultiFrameImageStorage )
    {
    if( GetUltraSoundSpacingValueFromSequence(ds, sp) )
      {
      // 3rd dimension is too difficult to handle for now...
      // (0018,1065) DS [0\ 957\ 990\ 990\1023\1023\ 990\ 990\1023\ 957\1023\1023\1023\ 990... # 562,113 FrameTimeVector
      sp.push_back( 1.0 );
      return sp;
      }
    else
      {
      // TODO this one is easy:
      // (0028,0009) AT (0018,1063)                              #   4, 1 FrameIncrementPointer
      // -> (0018,1063) DS [76.000000]                              #  10, 1 FrameTime

      gdcmWarningMacro( "No spacing value found" );
      sp.push_back( 1.0 );
      sp.push_back( 1.0 );
      sp.push_back( 1.0 );
      return sp;
      }
    }

  Tag spacingtag = GetSpacingTagFromMediaStorage(ms);
  if( spacingtag != Tag(0xffff,0xffff) && ds.FindDataElement( spacingtag ) && !ds.GetDataElement( spacingtag ).IsEmpty() )
    {
    const DataElement& de = ds.GetDataElement( spacingtag );
    const Global &g = GlobalInstance;
    const Dicts &dicts = g.GetDicts();
    const DictEntry &entry = dicts.GetDictEntry(de.GetTag());
    const VR & vr = entry.GetVR();
    assert( vr.Compatible( de.GetVR() ) );
    switch(vr)
      {
    case VR::DS:
        {
        Element<VR::DS,VM::VM1_n> el;
        std::stringstream ss;
        const ByteValue *bv = de.GetByteValue();
        assert( bv );
        std::string s = std::string( bv->GetPointer(), bv->GetLength() );
        ss.str( s );
        // Stupid file: CT-MONO2-8-abdo.dcm
        // The spacing is something like that: [0.2\0\0.200000]
        // I would need to throw an expection that VM is not compatible
        el.SetLength( entry.GetVM().GetLength() * entry.GetVR().GetSizeof() );
        std::string::size_type found = s.find('\\');
        if( found != std::string::npos )
          {
          el.Read( ss );
          assert( el.GetLength() == 2 );
          for(unsigned int i = 0; i < el.GetLength(); ++i)
            {
            if( el.GetValue(i) )
              {
              sp.push_back( el.GetValue(i) );
              }
            else
              {
              gdcmWarningMacro( "Cant have a spacing of 0" );
              sp.push_back( 1.0 );
              }
            }
          std::swap( sp[0], sp[1]);
          }
        else
          {
          double singleval;
          ss >> singleval;
          if( singleval == 0.0 )
            {
            singleval = 1.0;
            }
          sp.push_back( singleval );
          sp.push_back( singleval );
          }
        assert( sp.size() == (unsigned int)entry.GetVM() );
        }
      break;
    case VR::IS:
        {
        Element<VR::IS,VM::VM1_n> el;
        std::stringstream ss;
        const ByteValue *bv = de.GetByteValue();
        assert( bv );
        std::string s = std::string( bv->GetPointer(), bv->GetLength() );
        ss.str( s );
        el.SetLength( entry.GetVM().GetLength() * entry.GetVR().GetSizeof() );
        el.Read( ss );
        for(unsigned int i = 0; i < el.GetLength(); ++i)
          sp.push_back( el.GetValue(i) );
        assert( sp.size() == (unsigned int)entry.GetVM() );
        }
      break;
    default:
      assert(0);
      break;
      }
    }
  else
    {
    sp.push_back( 1.0 );
    sp.push_back( 1.0 );
    }
  assert( sp.size() == 2 );
  // Make sure multiframe:
  std::vector<unsigned int> dims = ImageHelper::GetDimensionsValue( f );

  // Do Z:
  Tag zspacingtag = ImageHelper::GetZSpacingTagFromMediaStorage(ms);
  if( zspacingtag != Tag(0xffff,0xffff) && ds.FindDataElement( zspacingtag ) )
    {
    const DataElement& de = ds.GetDataElement( zspacingtag );
    if( de.IsEmpty() )
      {
      sp.push_back( 1.0 );
      }
    else
      {
      const Global &g = GlobalInstance;
      const Dicts &dicts = g.GetDicts();
      const DictEntry &entry = dicts.GetDictEntry(de.GetTag());
      const VR & vr = entry.GetVR();
      assert( de.GetVR() == vr || de.GetVR() == VR::INVALID || de.GetVR() == VR::UN );
      if( entry.GetVM() == VM::VM1 )
        {
        switch(vr)
          {
        case VR::DS:
            {
            Element<VR::DS,VM::VM1_n> el;
            std::stringstream ss;
            const ByteValue *bv = de.GetByteValue();
            assert( bv );
            std::string s = std::string( bv->GetPointer(), bv->GetLength() );
            ss.str( s );
            el.SetLength( entry.GetVM().GetLength() * entry.GetVR().GetSizeof() );
            el.Read( ss );
            for(unsigned int i = 0; i < el.GetLength(); ++i)
              {
              const double value = el.GetValue(i);
              sp.push_back( value );
              }
            //assert( sp.size() == entry.GetVM() );
            }
          break;
        default:
          assert(0);
          break;
          }
        }
      else
        {
        assert( entry.GetVM() == VM::VM2_n );
        assert( vr == VR::DS );
        Attribute<0x28,0x8> numberoframes;
        const DataElement& de1 = ds.GetDataElement( numberoframes.GetTag() );
        numberoframes.SetFromDataElement( de1 );
        Attribute<0x3004,0x000c> gridframeoffsetvector;
        const DataElement& de2 = ds.GetDataElement( gridframeoffsetvector.GetTag() );
        gridframeoffsetvector.SetFromDataElement( de2 );
        double v1 = gridframeoffsetvector[0];
        double v2 = gridframeoffsetvector[1];
        // FIXME. I should check consistency
        sp.push_back( v2 - v1 );
        }
      }
    }
  else if( ds.FindDataElement( Tag(0x0028,0x0009) ) ) // Frame Increment Pointer
    {
    const DataElement& de = ds.GetDataElement( Tag(0x0028,0x0009) );
    Attribute<0x0028,0x0009,VR::AT,VM::VM1> at;
    at.SetFromDataElement( de );
    assert( ds.FindDataElement( at.GetTag() ) );
    if( ds.FindDataElement( at.GetValue() ) )
      {
/*
$ dcmdump D_CLUNIE_NM1_JPLL.dcm" | grep 0028,0009
(0028,0009) AT (0054,0010)\(0054,0020)                  #   8, 2 FrameIncrementPointer
*/
      const DataElement& de2 = ds.GetDataElement( at.GetValue() );
      if( at.GetValue() == Tag(0x0018,0x1063) && at.GetNumberOfValues() == 1 )
        {
        Attribute<0x0018,0x1063> at2;
        at2.SetFromDataElement( de2 );
        if( dims[2] > 1 )
          {
          sp.push_back( at2.GetValue() );
          }
        else
          {
          if( at2.GetValue() != 0. )
            {
            gdcmErrorMacro( "Number of Frame should be equal to 0" );
            sp.push_back( 0.0 );
            }
          else
            {
            sp.push_back( 1.0 );
            }
          }
        }
      else
        {
        gdcmWarningMacro( "Dont know how to handle spacing for: " << de );
        sp.push_back( 1.0 );
        }
      }
    else
      {
      gdcmErrorMacro( "Tag: " << at.GetTag() << " was found to point to missing"
        "Tag: " << at.GetValue() << " default to 1.0." );
      sp.push_back( 1.0 );
      }
    }
  else
    {
    sp.push_back( 1.0 );
    }

  assert( sp.size() == 3 );
  assert( sp[0] != 0. );
  assert( sp[1] != 0. );
  //if( ms != MediaStorage::MRImageStorage )
  //  assert( sp[2] != 0. );
  return sp;
}

void ImageHelper::SetSpacingValue(DataSet & ds, const std::vector<double> & spacing)
{
  MediaStorage ms;
  ms.SetFromDataSet(ds);
  assert( MediaStorage::IsImage( ms ) );
  if( ms == MediaStorage::SecondaryCaptureImageStorage )
    {
    Tag pixelspacing(0x0028,0x0030);
    Tag imagerpixelspacing(0x0018,0x1164);
    Tag spacingbetweenslice(0x0018,0x0088);
    //ds.Remove( pixelspacing );
    //ds.Remove( imagerpixelspacing );
    //ds.Remove( spacingbetweenslice );
    //return;
    }
  assert( spacing.size() == 3 );

  if( ms == MediaStorage::EnhancedCTImageStorage
   || ms == MediaStorage::EnhancedMRImageStorage
   || ms == MediaStorage::SegmentationStorage )
    {
/*
    (0028,9110) SQ (Sequence with undefined length #=1)     # u/l, 1 PixelMeasuresSequence
      (fffe,e000) na (Item with undefined length #=2)         # u/l, 1 Item
        (0018,0050) DS [5.00000]                                #   8, 1 SliceThickness
        (0028,0030) DS [0.820312\0.820312]                      #  18, 2 PixelSpacing
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/
    const Tag tfgs(0x5200,0x9229);
    SmartPointer<SequenceOfItems> sqi;
    if( !ds.FindDataElement( tfgs ) )
      {
      sqi = new SequenceOfItems;
      DataElement de( tfgs );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi );
      de.SetVLToUndefined();
      ds.Insert( de );
      }
    //sqi = (SequenceOfItems*)ds.GetDataElement( tfgs ).GetSequenceOfItems();
    sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item1 = sqi->GetItem(1);
    DataSet &subds = item1.GetNestedDataSet();
    const Tag tpms(0x0028,0x9110);
    if( !subds.FindDataElement( tpms ) )
      {
      SequenceOfItems *sqi2 = new SequenceOfItems;
      DataElement de( tpms );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi2 );
      de.SetVLToUndefined();
      subds.Insert( de );
      }

    //sqi = (SequenceOfItems*)subds.GetDataElement( tpms ).GetSequenceOfItems();
    sqi = subds.GetDataElement( tpms ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item2 = sqi->GetItem(1);
    DataSet &subds2 = item2.GetNestedDataSet();

    // <entry group="0028" element="9110" vr="SQ" vm="1" name="Pixel Measures Sequence"/>
    //Attribute<0x0018,0x0050> at2;
    //at2.SetValue( spacing[2] );
    Attribute<0x0028,0x0030> at1;
    at1.SetValue( spacing[1], 0 );
    at1.SetValue( spacing[0], 1 );
    subds2.Replace( at1.GetAsDataElement() );
    //subds2.Replace( at2.GetAsDataElement() );

    return;
    }


  Tag spacingtag = GetSpacingTagFromMediaStorage(ms);
  Tag zspacingtag = GetZSpacingTagFromMediaStorage(ms);
  //std::vector<Tag> spacingtags;
  //spacingtags.push_back( spacingtag );
  //spacingtags.push_back( zspacingtag );
    {
    const Tag &currentspacing = spacingtag;
    if( currentspacing != Tag(0xffff,0xffff) )
      {
      DataElement de( currentspacing );
      const Global &g = GlobalInstance;
      const Dicts &dicts = g.GetDicts();
      const DictEntry &entry = dicts.GetDictEntry(de.GetTag());
      const VR & vr = entry.GetVR();
      const VM & vm = entry.GetVM(); (void)vm;
      assert( de.GetVR() == vr || de.GetVR() == VR::INVALID );
      switch(vr)
        {
      case VR::DS:
          {
          Element<VR::DS,VM::VM1_n> el;
          el.SetLength( entry.GetVM().GetLength() * vr.GetSizeof() );
          assert( entry.GetVM() == VM::VM2 );
          for( unsigned int i = 0; i < entry.GetVM().GetLength(); ++i)
            {
            el.SetValue( spacing[i], i );
            }
          el.SetValue( spacing[1], 0 );
          el.SetValue( spacing[0], 1 );
          //assert( el.GetValue(0) == spacing[0] && el.GetValue(1) == spacing[1] );
          std::stringstream os;
          el.Write( os );
          de.SetVR( VR::DS );
          if( os.str().size() % 2 ) os << " ";
          VL::Type osStrSize = (VL::Type)os.str().size();
          de.SetByteValue( os.str().c_str(),osStrSize );
          ds.Replace( de );
          }
        break;
      case VR::IS:
          {
          Element<VR::IS,VM::VM1_n> el;
          el.SetLength( entry.GetVM().GetLength() * vr.GetSizeof() );
          assert( entry.GetVM() == VM::VM2 );
          for( unsigned int i = 0; i < entry.GetVM().GetLength(); ++i)
            {
            el.SetValue( (int)spacing[i], i );
            }
          //assert( el.GetValue(0) == spacing[0] && el.GetValue(1) == spacing[1] );
          std::stringstream os;
          el.Write( os );
          de.SetVR( VR::IS );
          if( os.str().size() % 2 ) os << " ";
          VL::Type osStrSize = (VL::Type)os.str().size();
          de.SetByteValue( os.str().c_str(), osStrSize );
          ds.Replace( de );
          }
        break;
      default:
        assert(0);
        }
      }
    }
    {
    const Tag &currentspacing = zspacingtag;
    if( currentspacing != Tag(0xffff,0xffff) )
      {
      DataElement de( currentspacing );
      const Global &g = GlobalInstance;
      const Dicts &dicts = g.GetDicts();
      const DictEntry &entry = dicts.GetDictEntry(de.GetTag());
      const VR & vr = entry.GetVR();
      const VM & vm = entry.GetVM(); (void)vm;
      assert( de.GetVR() == vr || de.GetVR() == VR::INVALID );
      if( entry.GetVM() == VM::VM2_n )
        {
        assert( vr == VR::DS );
        assert( de.GetTag() == Tag(0x3004,0x000c) );
        Attribute<0x28,0x8> numberoframes;
        // Make we are multiframes:
        if( ds.FindDataElement( numberoframes.GetTag() ) )
          {
          const DataElement& de1 = ds.GetDataElement( numberoframes.GetTag() );
          numberoframes.SetFromDataElement( de1 );

          Element<VR::DS,VM::VM2_n> el;
          el.SetLength( numberoframes.GetValue() * vr.GetSizeof() );
          assert( entry.GetVM() == VM::VM2_n );
          double spacing_start = 0;
          assert( 0 < numberoframes.GetValue() );
          for( int i = 0; i < numberoframes.GetValue(); ++i)
            {
            el.SetValue( spacing_start, i );
            spacing_start += spacing[2];
            }
          //assert( el.GetValue(0) == spacing[0] && el.GetValue(1) == spacing[1] );
          std::stringstream os;
          el.Write( os );
          de.SetVR( VR::DS );
          if( os.str().size() % 2 ) os << " ";
          VL::Type osStrSize = (VL::Type)os.str().size();
          de.SetByteValue( os.str().c_str(), osStrSize );
          ds.Replace( de );
          }
        }
      else
        {
        switch(vr)
          {
        case VR::DS:
            {
            Element<VR::DS,VM::VM1_n> el;
            el.SetLength( entry.GetVM().GetLength() * vr.GetSizeof() );
            assert( entry.GetVM() == VM::VM1 );
            for( unsigned int i = 0; i < entry.GetVM().GetLength(); ++i)
              {
              el.SetValue( spacing[i+2], i );
              }
            //assert( el.GetValue(0) == spacing[0] && el.GetValue(1) == spacing[1] );
            std::stringstream os;
            el.Write( os );
            de.SetVR( VR::DS );
            if( os.str().size() % 2 ) os << " ";
            VL::Type osStrSize = (VL::Type)os.str().size();
            de.SetByteValue( os.str().c_str(), osStrSize );
            ds.Replace( de );
            }
          break;
        default:
          assert(0);
          }
        }
      }
    }

}

void SetDataElementInSQAsItemNumber(DataSet & ds, DataElement const & de, Tag const & sqtag, unsigned int itemidx)
{
    const Tag tfgs = sqtag; //(0x5200,0x9230);
    SmartPointer<SequenceOfItems> sqi;
    if( !ds.FindDataElement( tfgs ) )
      {
      sqi = new SequenceOfItems;
      DataElement detmp( tfgs );
      detmp.SetVR( VR::SQ );
      detmp.SetValue( *sqi );
      detmp.SetVLToUndefined();
      ds.Insert( detmp );
      }
    //sqi = (SequenceOfItems*)ds.GetDataElement( tfgs ).GetSequenceOfItems();
    sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( sqi->GetNumberOfItems() < itemidx )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item1 = sqi->GetItem(itemidx);
    DataSet &subds = item1.GetNestedDataSet();
    const Tag tpms(0x0020,0x9113);
    if( !subds.FindDataElement( tpms ) )
      {
      SequenceOfItems *sqi2 = new SequenceOfItems;
      DataElement detmp( tpms );
      detmp.SetVR( VR::SQ );
      detmp.SetValue( *sqi2 );
      detmp.SetVLToUndefined();
      subds.Insert( detmp );
      }

    //sqi = (SequenceOfItems*)subds.GetDataElement( tpms ).GetSequenceOfItems();
    sqi = subds.GetDataElement( tpms ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item2 = sqi->GetItem(1);
    DataSet &subds2 = item2.GetNestedDataSet();

    //Attribute<0x0020,0x0032> ipp = {{0,0,0}}; // default value
    //ipp.SetValue( origin[0], 0);
    //ipp.SetValue( origin[1], 1);
    //ipp.SetValue( origin[2], 2);

    subds2.Replace( de );
}

void ImageHelper::SetOriginValue(DataSet & ds, const Image & image)
{
  const double *origin = image.GetOrigin();
  //assert( origin.size() == 3 );
  MediaStorage ms;
  ms.SetFromDataSet(ds);
  assert( MediaStorage::IsImage( ms ) );

  if( ms == MediaStorage::SecondaryCaptureImageStorage )
    {
    // https://sourceforge.net/p/gdcm/bugs/322/
    // default behavior is simply to pass
    return;
    }

  // FIXME Hardcoded
  if( ms != MediaStorage::CTImageStorage
   && ms != MediaStorage::MRImageStorage
   && ms != MediaStorage::RTDoseStorage
   //&& ms != MediaStorage::ComputedRadiographyImageStorage
   && ms != MediaStorage::SegmentationStorage
   && ms != MediaStorage::EnhancedMRImageStorage
   && ms != MediaStorage::EnhancedCTImageStorage )
    {
    // FIXME: should I remove the ipp tag ???
    return;
    }

  if( ms == MediaStorage::EnhancedCTImageStorage
   || ms == MediaStorage::EnhancedMRImageStorage
   || ms == MediaStorage::SegmentationStorage )
    {
/*
    (0020,9113) SQ (Sequence with undefined length #=1)     # u/l, 1 PlanePositionSequence
      (fffe,e000) na (Item with undefined length #=1)         # u/l, 1 Item
        (0020,0032) DS [40.0000\-105.000\105.000]               #  24, 3 ImagePositionPatient
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/

    const Tag tfgs(0x5200,0x9230);

    Attribute<0x0020,0x0032> ipp = {{0,0,0}}; // default value
    double zspacing = image.GetSpacing(2);
    unsigned int dimz = image.GetDimension(2);
    const double *cosines = image.GetDirectionCosines();
    DirectionCosines dc( cosines );

    double normal[3];
    dc.Cross( normal );

    for(unsigned int i = 0; i < dimz; ++i )
      {
      double new_origin[3];
      for (int j = 0; j < 3; j++)
        {
        // the n'th slice is n * z-spacing aloung the IOP-derived
        // z-axis
        new_origin[j] = origin[j] + normal[j] * i * zspacing;
        }

      ipp.SetValue( new_origin[0], 0);
      ipp.SetValue( new_origin[1], 1);
      ipp.SetValue( new_origin[2], 2);
      SetDataElementInSQAsItemNumber(ds, ipp.GetAsDataElement(), tfgs, i+1);
      }

    return;
    }

  // Image Position (Patient)
  Attribute<0x0020,0x0032> ipp = {{0,0,0}}; // default value
  ipp.SetValue( origin[0], 0);
  ipp.SetValue( origin[1], 1);
  ipp.SetValue( origin[2], 2);

  ds.Replace( ipp.GetAsDataElement() );
}

void ImageHelper::SetDirectionCosinesValue(DataSet & ds, const std::vector<double> & dircos)
{
  MediaStorage ms;
  ms.SetFromDataSet(ds);
  assert( MediaStorage::IsImage( ms ) );

  if( ms == MediaStorage::SecondaryCaptureImageStorage )
    {
    // https://sourceforge.net/p/gdcm/bugs/322/
    // default behavior is simply to pass
    return;
    }

  // FIXME Hardcoded
  if( ms != MediaStorage::CTImageStorage
   && ms != MediaStorage::MRImageStorage
   && ms != MediaStorage::RTDoseStorage
   //&& ms != MediaStorage::ComputedRadiographyImageStorage
   && ms != MediaStorage::SegmentationStorage
   && ms != MediaStorage::EnhancedMRImageStorage
   && ms != MediaStorage::EnhancedCTImageStorage )
    {
    // FIXME: should I remove the iop tag ???
    return;
    }

  // Image Orientation (Patient)
  Attribute<0x0020,0x0037> iop = {{1,0,0,0,1,0}}; // default value

  assert( dircos.size() == 6 );
  DirectionCosines dc( &dircos[0] );
  if( !dc.IsValid() )
    {
    gdcmWarningMacro( "Direction Cosines are not valid. Using default value (1\\0\\0\\0\\1\\0)" );
    }
  else
    {
    iop.SetValue( dircos[0], 0);
    iop.SetValue( dircos[1], 1);
    iop.SetValue( dircos[2], 2);
    iop.SetValue( dircos[3], 3);
    iop.SetValue( dircos[4], 4);
    iop.SetValue( dircos[5], 5);
    }

  if( ms == MediaStorage::EnhancedCTImageStorage
   || ms == MediaStorage::EnhancedMRImageStorage
   || ms == MediaStorage::SegmentationStorage )
    {
/*
    (0020,9116) SQ (Sequence with undefined length #=1)     # u/l, 1 PlaneOrientationSequence
      (fffe,e000) na (Item with undefined length #=1)         # u/l, 1 Item
        (0020,0037) DS [0.00000\1.00000\0.00000\0.00000\0.00000\-1.00000] #  48, 6 ImageOrientationPatient
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/
    const Tag tfgs(0x5200,0x9229);
    SmartPointer<SequenceOfItems> sqi;
    if( !ds.FindDataElement( tfgs ) )
      {
      sqi = new SequenceOfItems;
      DataElement de( tfgs );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi );
      de.SetVLToUndefined();
      ds.Insert( de );
      }
    //sqi = (SequenceOfItems*)ds.GetDataElement( tfgs ).GetSequenceOfItems();
    sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item1 = sqi->GetItem(1);
    DataSet &subds = item1.GetNestedDataSet();
    const Tag tpms(0x0020,0x9116);
    if( !subds.FindDataElement( tpms ) )
      {
      SequenceOfItems *sqi2 = new SequenceOfItems;
      DataElement de( tpms );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi2 );
      de.SetVLToUndefined();
      subds.Insert( de );
      }

    //sqi = (SequenceOfItems*)subds.GetDataElement( tpms ).GetSequenceOfItems();
    sqi = subds.GetDataElement( tpms ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item2 = sqi->GetItem(1);
    DataSet &subds2 = item2.GetNestedDataSet();

    subds2.Replace( iop.GetAsDataElement() );
    return;
    }

  ds.Replace( iop.GetAsDataElement() );
}

void ImageHelper::SetRescaleInterceptSlopeValue(File & f, const Image & img)
{
  MediaStorage ms;
  // SetFromFile is required here, SetFromDataSet is not enough for all cases
  ms.SetFromFile(f);
  assert( MediaStorage::IsImage( ms ) );
  DataSet &ds = f.GetDataSet();

  // FIXME Hardcoded
  if( ms != MediaStorage::CTImageStorage
   && ms != MediaStorage::ComputedRadiographyImageStorage
   && ms != MediaStorage::MRImageStorage // FIXME !
   && ms != MediaStorage::PETImageStorage
   && ms != MediaStorage::RTDoseStorage
   && ms != MediaStorage::SecondaryCaptureImageStorage
   && ms != MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage
   && ms != MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage
   && ms != MediaStorage::EnhancedMRImageStorage
   && ms != MediaStorage::EnhancedCTImageStorage
   && ms != MediaStorage::SegmentationStorage )
    {
    if( img.GetIntercept() != 0. || img.GetSlope() != 1. )
      {
      throw "Impossible"; // Please report
      }
    return;
    }

  if( ms == MediaStorage::SegmentationStorage ) return; // seg storage cannot have rescale slope
  if( ms == MediaStorage::EnhancedCTImageStorage
   || ms == MediaStorage::EnhancedMRImageStorage
  )
    {
/*
    (0020,9116) SQ (Sequence with undefined length #=1)     # u/l, 1 PlaneOrientationSequence
      (fffe,e000) na (Item with undefined length #=1)         # u/l, 1 Item
        (0020,0037) DS [0.00000\1.00000\0.00000\0.00000\0.00000\-1.00000] #  48, 6 ImageOrientationPatient
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/
    const Tag tfgs(0x5200,0x9229);
    SmartPointer<SequenceOfItems> sqi;
    if( !ds.FindDataElement( tfgs ) )
      {
      sqi = new SequenceOfItems;
      DataElement de( tfgs );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi );
      de.SetVLToUndefined();
      ds.Insert( de );
      }
    //sqi = (SequenceOfItems*)ds.GetDataElement( tfgs ).GetSequenceOfItems();
    sqi = ds.GetDataElement( tfgs ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item1 = sqi->GetItem(1);
    DataSet &subds = item1.GetNestedDataSet();
    const Tag tpms(0x0028,0x9145);
    if( !subds.FindDataElement( tpms ) )
      {
      SequenceOfItems *sqi2 = new SequenceOfItems;
      DataElement de( tpms );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi2 );
      de.SetVLToUndefined();
      subds.Insert( de );
      }

    //sqi = (SequenceOfItems*)subds.GetDataElement( tpms ).GetSequenceOfItems();
    sqi = subds.GetDataElement( tpms ).GetValueAsSQ();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item2 = sqi->GetItem(1);
    DataSet &subds2 = item2.GetNestedDataSet();

    Attribute<0x0028,0x1052> at1;
    at1.SetValue( img.GetIntercept() );
    subds2.Insert( at1.GetAsDataElement() );
    Attribute<0x0028,0x1053> at2;
    at2.SetValue( img.GetSlope() );
    subds2.Insert( at2.GetAsDataElement() );

    return;
    }

  if( ms == MediaStorage::RTDoseStorage )
    {
    if( img.GetIntercept() != 0 )
      {
      gdcmErrorMacro( "Cannot have an intercept value for RTDOSE, only Scaling allowed" );
      return;
      }
    Attribute<0x3004,0x00e> at2;
    at2.SetValue( img.GetSlope() );
    ds.Replace( at2.GetAsDataElement() );

    Attribute<0x0028,0x0009> framePointer;
    framePointer.SetNumberOfValues(1);
    framePointer.SetValue( Tag(0x3004,0x000C) );
    ds.Replace( framePointer.GetAsDataElement() );

    return;
    }

  // Question: should I always insert them ?
  // Answer: not always, let's discard MR if (1,0):
  if( ms == MediaStorage::MRImageStorage && img.GetIntercept() == 0. && img.GetSlope() == 1. )
    {
    }
  else
    {
    Attribute<0x0028,0x1052> at1;
    at1.SetValue( img.GetIntercept() );
    ds.Replace( at1.GetAsDataElement() );
    Attribute<0x0028,0x1053> at2;
    at2.SetValue( img.GetSlope() );
    ds.Replace( at2.GetAsDataElement() );

    Attribute<0x0028,0x1054> at3; // Rescale Type
    at3.SetValue( "US" ); // FIXME
    if( ms == MediaStorage::SecondaryCaptureImageStorage )
      {
      // As per 3-2009, US is the only valid enumerated value:
      ds.Replace( at3.GetAsDataElement() );
      }
    else
      {
      // In case user decide to override the default:
      ds.ReplaceEmpty( at3.GetAsDataElement() );
      }
    }
}

bool ImageHelper::ComputeSpacingFromImagePositionPatient(const std::vector<double> & imageposition, std::vector<double> & spacing)
{
  if( imageposition.size() % 3 != 0 )
    {
    return false;
    }
  std::vector<double>::const_iterator it = imageposition.begin();
  //const double x0 = *it++;
  //const double y0 = *it++;
  //const double z0 = *it++;
  spacing[0] = spacing[1] = spacing[2] = 0.;
  for( ; it != imageposition.end(); ++it)
    {
    const double x = *it++;
    const double y = *it++;
    const double z = *it;
    spacing[0] += x;
    spacing[1] += y;
    spacing[2] += z;
    }
  size_t n = imageposition.size() / 3;
  spacing[0] /= (double)n;
  spacing[1] /= (double)n;
  spacing[2] /= (double)n;

  return true;
}


//functions to get more information from a file
//useful for the stream image reader, which fills in necessary image information
//distinctly from the reader-style data input
//code is borrowed from gdcmPixmapReader::ReadImage(MediaStorage const &ms)
PhotometricInterpretation ImageHelper::GetPhotometricInterpretationValue(File const& f){
  // 5. Photometric Interpretation
  // D 0028|0004 [CS] [Photometric Interpretation] [MONOCHROME2 ]
  PixelFormat pf = GetPixelFormatValue(f);
  const Tag tphotometricinterpretation(0x0028, 0x0004);
  const ByteValue *photometricinterpretation =
    ImageHelper::GetPointerFromElement(tphotometricinterpretation, f);
  PhotometricInterpretation pi = PhotometricInterpretation::UNKNOW;
  if( photometricinterpretation )
    {
    std::string photometricinterpretation_str(
      photometricinterpretation->GetPointer(),
      photometricinterpretation->GetLength() );
    pi = PhotometricInterpretation::GetPIType( photometricinterpretation_str.c_str() );
    }
  else
    {
    if( pf.GetSamplesPerPixel() == 1 )
      {
      gdcmWarningMacro( "No PhotometricInterpretation found, default to MONOCHROME2" );
      pi = PhotometricInterpretation::MONOCHROME2;
      }
    else if( pf.GetSamplesPerPixel() == 3 )
      {
      gdcmWarningMacro( "No PhotometricInterpretation found, default to RGB" );
      pi = PhotometricInterpretation::RGB;
      }
    else if( pf.GetSamplesPerPixel() == 4 )
      {
      gdcmWarningMacro( "No PhotometricInterpretation found, default to RGB" );
      pi = PhotometricInterpretation::ARGB;
      }
    }

  bool isacrnema = false;
  DataSet ds = f.GetDataSet();
  const Tag trecognitioncode(0x0008,0x0010);
  if( ds.FindDataElement( trecognitioncode ) && !ds.GetDataElement( trecognitioncode ).IsEmpty() )
    {
    // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
    // PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm
    gdcmDebugMacro( "Mixture of ACR NEMA and DICOM file" );
    isacrnema = true;
  }
  if( !pf.GetSamplesPerPixel() || (pi.GetSamplesPerPixel() != pf.GetSamplesPerPixel()) )
    {
    if( pi != PhotometricInterpretation::UNKNOW )
      {
      pf.SetSamplesPerPixel( pi.GetSamplesPerPixel() );
      }
    else if ( isacrnema )
      {
      assert ( pf.GetSamplesPerPixel() == 0 );
      assert ( pi == PhotometricInterpretation::UNKNOW );
      pf.SetSamplesPerPixel( 1 );
      pi = PhotometricInterpretation::MONOCHROME2;
      }
    else
      {
      gdcmWarningMacro( "Cannot recognize image type. Does not looks like ACR-NEMA and is missing both Sample Per Pixel AND PhotometricInterpretation. Please report" );
      }
    }
  return pi;
}
//returns the configuration of colors in a plane, either RGB RGB RGB or RRR GGG BBB
//code is borrowed from gdcmPixmapReader::ReadImage(MediaStorage const &ms)
unsigned int ImageHelper::GetPlanarConfigurationValue(const File& f){
  // 4. Planar Configuration
  // D 0028|0006 [US] [Planar Configuration] [1]
  const Tag planarconfiguration = Tag(0x0028, 0x0006);
  PixelFormat pf = GetPixelFormatValue(f);
  unsigned int pc = 0;
  // FIXME: Whatif planaconfiguration is send in a grayscale image... it would be empty...
  // well hopefully :(
  DataSet const & ds = f.GetDataSet();
  if( ds.FindDataElement( planarconfiguration ) && !ds.GetDataElement( planarconfiguration ).IsEmpty() )
    {
    const DataElement& de = ds.GetDataElement( planarconfiguration );
    Attribute<0x0028,0x0006> at = { 0 };
    at.SetFromDataElement( de );

    //unsigned int pc = ReadUSFromTag( planarconfiguration, ss, conversion );
    pc = at.GetValue();
    if( pc && pf.GetSamplesPerPixel() != 3 )
      {
      gdcmDebugMacro( "Cannot have PlanarConfiguration=1, when Sample Per Pixel != 3" );
      pc = 0;
      }
    }
  return pc;
}

  //returns the lookup table of an image file
SmartPointer<LookupTable> ImageHelper::GetLUT(File const& f){

  DataSet const & ds = f.GetDataSet();
  PixelFormat pf = GetPixelFormatValue(f);
  PhotometricInterpretation pi = GetPhotometricInterpretationValue(f);
  // Do the Palette Color:
  // 1. Modality LUT Sequence
  bool modlut = ds.FindDataElement(Tag(0x0028,0x3000) );
  if( modlut )
    {
    gdcmWarningMacro( "Modality LUT (0028,3000) are not handled. Image will not be displayed properly" );
    }
  // 2. LUTData (0028,3006)
  // technically I do not need to warn about LUTData since either modality lut XOR VOI LUT need to
  // be sent to require a LUT Data...
  bool lutdata = ds.FindDataElement(Tag(0x0028,0x3006) );
  if( lutdata )
    {
    gdcmWarningMacro( "LUT Data (0028,3006) are not handled. Image will not be displayed properly" );
    }
  // 3. VOILUTSequence (0028,3010)
  bool voilut = ds.FindDataElement(Tag(0x0028,0x3010) );
  if( voilut )
    {
    gdcmWarningMacro( "VOI LUT (0028,3010) are not handled. Image will not be displayed properly" );
    }
  // (0028,0120) US 32767                                    #   2, 1 PixelPaddingValue
  bool pixelpaddingvalue = ds.FindDataElement(Tag(0x0028,0x0120));

  // PS 3.3 - 2008 / C.7.5.1.1.2 Pixel Padding Value and Pixel Padding Range Limit
  if(pixelpaddingvalue)
    {
    // Technically if Pixel Padding Value is 0 on MONOCHROME2 image, then appearance should be fine...
    bool vizissue = true;
    if( pf.GetPixelRepresentation() == 0 )
      {
      Element<VR::US,VM::VM1> ppv;
      if( !ds.GetDataElement(Tag(0x0028,0x0120) ).IsEmpty() )
        {
        ppv.SetFromDataElement( ds.GetDataElement(Tag(0x0028,0x0120)) ); //.GetValue() );
        if( pi == PhotometricInterpretation::MONOCHROME2 && ppv.GetValue() == 0 )
          {
          vizissue = false;
          }
        }
      }
    else if( pf.GetPixelRepresentation() == 1 )
      {
      gdcmDebugMacro( "TODO" );
      }
    // test if there is any viz issue:
    if( vizissue )
      {
      gdcmDebugMacro( "Pixel Padding Value (0028,0120) is not handled. Image will not be displayed properly" );
      }
    }
  SmartPointer<LookupTable> lut = new LookupTable;
  const Tag testseglut(0x0028, (0x1221 + 0));
  if( ds.FindDataElement( testseglut ) )
    {
    lut = new SegmentedPaletteColorLookupTable;
    }
  //SmartPointer<SegmentedPaletteColorLookupTable> lut = new SegmentedPaletteColorLookupTable;
  lut->Allocate( pf.GetBitsAllocated() );

  // for each red, green, blue:
  for(int i=0; i<3; ++i)
    {
    // (0028,1101) US 0\0\16
    // (0028,1102) US 0\0\16
    // (0028,1103) US 0\0\16
    const Tag tdescriptor(0x0028, (uint16_t)(0x1101 + i));
    //const Tag tdescriptor(0x0028, 0x3002);
    Element<VR::US,VM::VM3> el_us3 = {{ 0, 0, 0}};
    // Now pass the byte array to a DICOMizer:
    el_us3.SetFromDataElement( ds[tdescriptor] ); //.GetValue() );
    lut->InitializeLUT( LookupTable::LookupTableType(i),
      el_us3[0], el_us3[1], el_us3[2] );

    // (0028,1201) OW
    // (0028,1202) OW
    // (0028,1203) OW
    const Tag tlut(0x0028, (uint16_t)(0x1201 + i));
    //const Tag tlut(0x0028, 0x3006);

    // Segmented LUT
    // (0028,1221) OW
    // (0028,1222) OW
    // (0028,1223) OW
    const Tag seglut(0x0028, (uint16_t)(0x1221 + i));
    if( ds.FindDataElement( tlut ) )
      {
      const ByteValue *lut_raw = ds.GetDataElement( tlut ).GetByteValue();
      if( lut_raw )
        {
        // LookupTableType::RED == 0
        lut->SetLUT( LookupTable::LookupTableType(i),
          (unsigned char*)lut_raw->GetPointer(), lut_raw->GetLength() );
        //assert( pf.GetBitsAllocated() == el_us3.GetValue(2) );
        }
      else
        {
        lut->Clear();
        }

      unsigned long check =
        (el_us3.GetValue(0) ? el_us3.GetValue(0) : 65536)
        * el_us3.GetValue(2) / 8;
      assert( !lut->Initialized() || check == lut_raw->GetLength() ); (void)check;
      }
    else if( ds.FindDataElement( seglut ) )
      {
      const ByteValue *lut_raw = ds.GetDataElement( seglut ).GetByteValue();
      if( lut_raw )
        {
        lut->SetLUT( LookupTable::LookupTableType(i),
          (unsigned char*)lut_raw->GetPointer(), lut_raw->GetLength() );
        //assert( pf.GetBitsAllocated() == el_us3.GetValue(2) );
        }
      else
        {
        lut->Clear();
        }

      //unsigned long check =
      //  (el_us3.GetValue(0) ? el_us3.GetValue(0) : 65536)
       // * el_us3.GetValue(2) / 8;
      //assert( check == lut_raw->GetLength() ); (void)check;
      }
    else
      {
      assert(0);
      }
    }
  if( ! lut->Initialized() ) {
    gdcmDebugMacro("LUT was uninitialized!");
  }
  return lut;
}


const ByteValue* ImageHelper::GetPointerFromElement(Tag const &tag, const File& inF) {

  const DataSet &ds = inF.GetDataSet();
  if( ds.FindDataElement( tag ) )
    {
    const DataElement &de = ds.GetDataElement( tag );
    return de.GetByteValue();
    }
  return 0;
}

MediaStorage ImageHelper::ComputeMediaStorageFromModality(const char *modality,
  unsigned int dimension, PixelFormat const & pixeltype,
  PhotometricInterpretation const & pi,
  double intercept , double slope
  )
{
  MediaStorage ms = MediaStorage::SecondaryCaptureImageStorage;
  ms.GuessFromModality(modality, dimension );

  // refine for SC family
  if( dimension != 2 &&
    (ms == MediaStorage::SecondaryCaptureImageStorage // dim 2
  || ms == MediaStorage::MultiframeSingleBitSecondaryCaptureImageStorage ) // dim 3
  )
    {
    // A.8.3.4 Multi-frame Grayscale Byte SC Image IOD Content Constraints
/*
- Samples per Pixel (0028,0002) shall be 1
- Photometric Interpretation (0028,0004) shall be MONOCHROME2
- Bits Allocated (0028,0100) shall be 8
- Bits Stored (0028,0101) shall be 8
- High Bit (0028,0102) shall be 7
- Pixel Representation (0028,0103) shall be 0
- Planar Configuration (0028,0006) shall not be present
*/
    if( dimension == 3 &&
      pixeltype.GetSamplesPerPixel() == 1 &&
      pi == PhotometricInterpretation::MONOCHROME2 &&
      pixeltype.GetBitsAllocated() == 8 &&
      pixeltype.GetBitsStored() == 8 &&
      pixeltype.GetHighBit() == 7 &&
      pixeltype.GetPixelRepresentation() == 0
    )
      {
      ms = MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage;
      if( intercept != 0 || slope != 1 )
        {
        // Table C.8-25b SC MULTI-FRAME IMAGE MODULE ATTRIBUTES
        // Note: This specifies an identity Modality LUT transformation.
        gdcmErrorMacro( "Cannot have shift/scale" );
        return MediaStorage::MS_END;
        }
      }
    else if( dimension == 3 &&
      pixeltype.GetSamplesPerPixel() == 1 &&
      pi == PhotometricInterpretation::MONOCHROME2 &&
      pixeltype.GetBitsAllocated() == 1 &&
      pixeltype.GetBitsStored() == 1 &&
      pixeltype.GetHighBit() == 0 &&
      pixeltype.GetPixelRepresentation() == 0
    )
      {
      ms = MediaStorage::MultiframeSingleBitSecondaryCaptureImageStorage;
      if( intercept != 0 || slope != 1 )
        {
        gdcmDebugMacro( "Cannot have shift/scale" );
        return MediaStorage::MS_END;
        }
      }
    else if( dimension == 3 &&
      pixeltype.GetSamplesPerPixel() == 1 &&
      pi == PhotometricInterpretation::MONOCHROME2 &&
      pixeltype.GetBitsAllocated() == 16 &&
      pixeltype.GetBitsStored() <= 16 && pixeltype.GetBitsStored() >= 9 &&
      pixeltype.GetHighBit() == pixeltype.GetBitsStored() - 1 &&
      pixeltype.GetPixelRepresentation() == 0
    )
      {
      ms = MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage;
      if( intercept != 0 || slope != 1 )
        {
        gdcmDebugMacro( "Cannot have shift/scale" );
        return MediaStorage::MS_END;
        }
      }
    else if( dimension == 3 &&
      pixeltype.GetSamplesPerPixel() == 3 &&
      pi == PhotometricInterpretation::RGB &&
      pixeltype.GetBitsAllocated() == 8 &&
      pixeltype.GetBitsStored() == 8 &&
      pixeltype.GetHighBit() == 7 &&
      pixeltype.GetPixelRepresentation() == 0
    )
      {
      ms = MediaStorage::MultiframeTrueColorSecondaryCaptureImageStorage;
      if( intercept != 0 || slope != 1 )
        {
        gdcmDebugMacro( "Cannot have shift/scale" );
        return MediaStorage::MS_END;
        }
      }
    else
      {
      gdcmDebugMacro( "Cannot handle Multi Frame image in SecondaryCaptureImageStorage" );
      return MediaStorage::MS_END;
      }
    }
  return ms;
}

} // end namespace gdcm
