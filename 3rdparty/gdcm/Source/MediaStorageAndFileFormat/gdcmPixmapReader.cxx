/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPixmapReader.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmValue.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmElement.h"
#include "gdcmPhotometricInterpretation.h"
#include "gdcmSegmentedPaletteColorLookupTable.h"
#include "gdcmTransferSyntax.h"
#include "gdcmLookupTable.h"
#include "gdcmAttribute.h"
#include "gdcmIconImage.h"
#include "gdcmPrivateTag.h"
#include "gdcmJPEGCodec.h"
#include "gdcmImageHelper.h"

namespace gdcm
{
PixmapReader::PixmapReader():PixelData(new Pixmap)
{
}

PixmapReader::~PixmapReader()
{
}

const Pixmap& PixmapReader::GetPixmap() const
{
  return *PixelData;
}
Pixmap& PixmapReader::GetPixmap()
{
  return *PixelData;
}

//void PixmapReader::SetPixmap(Pixmap const &img)
//{
//  PixelData = img;
//}


bool PixmapReader::Read()
{
  if( !Reader::Read() )
    {
    // cemra_bug/IM-0001-0066.dcm
    // will return from the parser with an error
    // but a partial Pixel Data can be seen
    return false;
    }

  const FileMetaInformation &header = F->GetHeader();
  const DataSet &ds = F->GetDataSet();
  const TransferSyntax &ts = header.GetDataSetTransferSyntax();

  // Need to set the type of image we are dealing with:
  PixelData->SetTransferSyntax( ts );

  bool res = false;
  /* Does it really make sense to check for Media Storage SOP Class UID?
   * I need then to check consistency with 0008 0016 Instance SOP Class UID
   * ... I don't think there is an end.
   * I'd rather go the old way check a bunch of tags (From Image Plane
   * Module).
   */
  MediaStorage ms = header.GetMediaStorage();
  bool isImage = MediaStorage::IsImage( ms );
  if( isImage )
    {
    // I cannot leave this here, since ELSCINT1 / LOSSLESS RICE declares CT Image Storage,
    // when in fact this is a private Media Storage (no Pixel Data element)
    //assert( ds.FindDataElement( Tag(0x7fe0,0x0010 ) ) );
    assert( ts != TransferSyntax::TS_END && ms != MediaStorage::MS_END );
    // Good it's the easy case. It's declared as an Image:
    //PixelData->SetCompressionFromTransferSyntax( ts );
    res = ReadImage(ms);
    }
  //else if( ms == MediaStorage::MRSpectroscopyStorage )
  //  {
  //  res = ReadImage(ms);
  //  }
  else
    {
    MediaStorage ms2 = ds.GetMediaStorage();
    //assert( !ds.FindDataElement( Tag(0x7fe0,0x0010 ) ) );
    if( ms == MediaStorage::MediaStorageDirectoryStorage && ms2 == MediaStorage::MS_END )
      {
      gdcmDebugMacro( "DICOM file is not an Image file but a : " <<
        MediaStorage::GetMSString(ms) << " SOP Class UID" );
      res = false;
      }
    else if( ms == ms2 && ms != MediaStorage::MS_END )
      {
      gdcmDebugMacro( "DICOM file is not an Image file but a : " <<
        MediaStorage::GetMSString(ms) << " SOP Class UID" );
      res = false;
      }
    else
      {
      if( ms2 != MediaStorage::MS_END )
        {
        bool isImage2 = MediaStorage::IsImage( ms2 );
        if( isImage2 )
          {
          gdcmDebugMacro( "After all it might be a DICOM file "
            "(Mallinckrodt-like)" );

          //PixelData->SetCompressionFromTransferSyntax( ts );
          //PixelData->SetCompressionType( Compression::RAW );
          res = ReadImage(ms2);
          }
        else
          {
          ms2.SetFromFile( *F );
          if( MediaStorage::IsImage( ms2 ) )
            {
            res = ReadImage(ms2);
            }
          else
            {
            gdcmDebugMacro( "DICOM file is not an Image file but a : " <<
              MediaStorage::GetMSString(ms2) << " SOP Class UID" );
            res = false;
            }
          }
        }
      else if( ts == TransferSyntax::ImplicitVRBigEndianACRNEMA || header.IsEmpty() )
        {
        // Those transfer syntax have a high probability of being ACR NEMA
        gdcmDebugMacro( "Looks like an ACR-NEMA file" );
        // Hopefully all ACR-NEMA are RAW:
        //PixelData->SetCompressionType( Compression::RAW );
        res = ReadACRNEMAImage();
        }
      else // there is a Unknown Media Storage Syntax
        {
        assert( ts != TransferSyntax::TS_END && ms == MediaStorage::MS_END );
        // god damit I don't know what to do...
        gdcmWarningMacro( "Attempting to read this file as a DICOM file"
          "\nDesperate attempt" );
        MediaStorage ms3;
        ms3.SetFromFile( GetFile() );
        if( ms3 != MediaStorage::MS_END )
          {
          res = ReadImage(ms3);
          }
        else
          {
          // Giving up
          res = false;
          }
        }
      }
    }

  //if(res) PixelData->Print( std::cout );
  return res;
}

// PICKER-16-MONO2-Nested_icon.dcm
void DoIconImage(const DataSet& rootds, Pixmap& image)
{
  const Tag ticonimage(0x0088,0x0200);
  IconImage &pixeldata = image.GetIconImage();
  if( rootds.FindDataElement( ticonimage ) )
    {
    const DataElement &iconimagesq = rootds.GetDataElement( ticonimage );
    //const SequenceOfItems* sq = iconimagesq.GetSequenceOfItems();
    SmartPointer<SequenceOfItems> sq = iconimagesq.GetValueAsSQ();
    // Is SQ empty ?
    if( !sq ) return;
    SequenceOfItems::ConstIterator it = sq->Begin();
    const DataSet &ds = it->GetNestedDataSet();

    // D 0028|0011 [US] [Columns] [512]
      {
      //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0011) );
      Attribute<0x0028,0x0011> at = { 0 };
      at.SetFromDataSet( ds );
      pixeldata.SetDimension(0, at.GetValue() );
      }

    // D 0028|0010 [US] [Rows] [512]
      {
      //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0010) );
      Attribute<0x0028,0x0010> at = { 0 };
      at.SetFromDataSet( ds );
      pixeldata.SetDimension(1, at.GetValue() );
      }

    PixelFormat pf;
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
    pixeldata.SetPixelFormat( pf );
    // D 0028|0004 [CS] [Photometric Interpretation] [MONOCHROME2 ]
    const Tag tphotometricinterpretation(0x0028, 0x0004);
    PhotometricInterpretation pi = PhotometricInterpretation::MONOCHROME2;
    if( ds.FindDataElement( tphotometricinterpretation ) )
      {
      const ByteValue *photometricinterpretation = ds.GetDataElement( tphotometricinterpretation ).GetByteValue();
      std::string photometricinterpretation_str(
        photometricinterpretation->GetPointer(),
        photometricinterpretation->GetLength() );
      pi = PhotometricInterpretation::GetPIType(
        photometricinterpretation_str.c_str());
      }
    assert( pi != PhotometricInterpretation::UNKNOW);
    pixeldata.SetPhotometricInterpretation( pi );

    //
    if ( pi == PhotometricInterpretation::PALETTE_COLOR )
      {
      SmartPointer<LookupTable> lut = new LookupTable;
      const Tag testseglut(0x0028, (0x1221 + 0));
      if( ds.FindDataElement( testseglut ) )
        {
        assert(0 && "Please report this image");
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
        Element<VR::US,VM::VM3> el_us3;
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
          assert( lut_raw );
          // LookupTableType::RED == 0
          lut->SetLUT( LookupTable::LookupTableType(i),
            (unsigned char*)lut_raw->GetPointer(), lut_raw->GetLength() );
          //assert( pf.GetBitsAllocated() == el_us3.GetValue(2) );

          unsigned long check =
            (el_us3.GetValue(0) ? el_us3.GetValue(0) : 65536)
            * el_us3.GetValue(2) / 8;
          assert( check == lut_raw->GetLength()
            || check + 1 == lut_raw->GetLength() ); (void)check;
          }
        else if( ds.FindDataElement( seglut ) )
          {
          const ByteValue *lut_raw = ds.GetDataElement( seglut ).GetByteValue();
          assert( lut_raw );
          lut->SetLUT( LookupTable::LookupTableType(i),
            (unsigned char*)lut_raw->GetPointer(), lut_raw->GetLength() );
          //assert( pf.GetBitsAllocated() == el_us3.GetValue(2) );

          //unsigned long check =
          //  (el_us3.GetValue(0) ? el_us3.GetValue(0) : 65536)
          //  * el_us3.GetValue(2) / 8;
          //assert( check == lut_raw->GetLength() ); (void)check;
          }
        else
          {
          gdcmWarningMacro( "Icon Sequence is incomplete. Giving up" );
          pixeldata.Clear();
          return;
          }
        }
      pixeldata.SetLUT(*lut);
      }

    const Tag tpixeldata = Tag(0x7fe0, 0x0010);
    if( !ds.FindDataElement( tpixeldata ) )
      {
      gdcmWarningMacro( "Icon Sequence is incomplete. Giving up" );
      pixeldata.Clear();
      return;
      }
    const DataElement& de = ds.GetDataElement( tpixeldata );
    pixeldata.SetDataElement( de );

    // Pass TransferSyntax:
    // Warning This is legal for the Icon to be uncompress in a compressed image
    // We need to set the appropriate TS here:
    const ByteValue *bv = de.GetByteValue();
    if( bv )
      pixeldata.SetTransferSyntax( TransferSyntax::ImplicitVRLittleEndian );
    else
      pixeldata.SetTransferSyntax( image.GetTransferSyntax() );
    }
}

// GE_DLX-8-MONO2-Multiframe.dcm
void DoCurves(const DataSet& ds, Pixmap& pixeldata)
{
  unsigned int numcurves;
  if( (numcurves = Curve::GetNumberOfCurves( ds )) )
    {
    pixeldata.SetNumberOfCurves( numcurves );

    Tag curve(0x5000,0x0000);
    bool finished = false;
    unsigned int idxcurves = 0;
    while( !finished )
      {
      const DataElement &de = ds.FindNextDataElement( curve );
      // Are we done:
      if( de.GetTag().GetGroup() > 0x50FF ) // last possible curve curve
        {
        finished = true;
        }
      else if( de.GetTag().IsPrivate() ) // GEMS owns some 0x5003
        {
        // Move on to the next public one:
        curve.SetGroup( (uint16_t)(de.GetTag().GetGroup() + 1) );
        curve.SetElement( 0 );
        }
      else
        {
        // Yay! this is an curve element
        Curve &ov = pixeldata.GetCurve(idxcurves);
        ++idxcurves; // move on to the next one
        curve = de.GetTag();
        uint16_t currentcurve = curve.GetGroup();
        assert( !(currentcurve % 2) ); // 0x6001 is not an curve...
        // Now loop on all element from this current group:
        DataElement de2 = de;
        while( de2.GetTag().GetGroup() == currentcurve )
          {
          ov.Update(de2);
          curve.SetElement( (uint16_t)(de2.GetTag().GetElement() + 1) );
          de2 = ds.FindNextDataElement( curve );
          // Next element:
          //curve.SetElement( curve.GetElement() + 1 );
          }
        // If we exit the loop we have pass the current curve and potentially point to the next one:
        //curve.SetElement( curve.GetElement() + 1 );
        //ov.Print( std::cout );
        }
      }
    //std::cout << "Num of curves: " << numcurves << std::endl;
    assert( idxcurves == numcurves );
    }
}

unsigned int GetNumberOfOverlaysInternal(DataSet const & ds, std::vector<uint16_t> & overlaylist)
{
  Tag overlay(0x6000,0x0000); // First possible overlay
  bool finished = false;
  unsigned int numoverlays = 0;
  while( !finished )
    {
    const DataElement &de = ds.FindNextDataElement( overlay );
    if( de.GetTag().GetGroup() > 0x60FF ) // last possible curve
      {
      finished = true;
      }
    else if( de.GetTag().IsPrivate() )
      {
      // Move on to the next public one:
      overlay.SetGroup( (uint16_t)(de.GetTag().GetGroup() + 1) );
      overlay.SetElement( 0 ); // reset just in case...
      }
    else
      {
      // Yeah this is a potential overlay element, let's check this is not a broken LEADTOOL image,
      // or prova0001.dcm:
      // (5000,0000) UL 0                                        #   4, 1 GenericGroupLength
      // (6000,0000) UL 0                                        #   4, 1 GenericGroupLength
      // (6001,0000) UL 28                                       #   4, 1 PrivateGroupLength
      // (6001,0010) LT [PAPYRUS 3.0]                            #  12, 1 PrivateCreator
      // (6001,1001) LT (no value available)                     #   0, 0 Unknown Tag & Data
/*
 * FIXME:
 * In order to support : gdcmData/SIEMENS_GBS_III-16-ACR_NEMA_1.acr
 *                       gdcmDataExtra/gdcmSampleData/images_of_interest/XA_GE_JPEG_02_with_Overlays.dcm
 * I cannot simply check for overlay_group,3000 this would not work
 * I would need a strong euristick
 */
      // Store found tag in overlay:
      overlay = de.GetTag();
      // heuristic based on either the Overlay Data or the Col/Row info
      Tag toverlaydata(overlay.GetGroup(),0x3000 );
      Tag toverlayrows(overlay.GetGroup(),0x0010 );
      Tag toverlaycols(overlay.GetGroup(),0x0011 );
      Tag toverlaybitpos(overlay.GetGroup(),0x0102 );
      if( ds.FindDataElement( toverlaydata ) )
        {
        // ok so far so good...
        const DataElement& overlaydata = ds.GetDataElement( toverlaydata );
        //const DataElement& overlaydata = ds.GetDataElement(Tag(overlay.GetGroup(),0x0010));
        if( !overlaydata.IsEmpty() )
          {
          ++numoverlays;
          overlaylist.push_back( overlay.GetGroup() );
          }
        }
      else if( ds.FindDataElement( toverlayrows ) && ds.FindDataElement( toverlaycols )
        && ds.FindDataElement( toverlaybitpos ) )
        {
        // Overlay Pixel are in Unused Pixel
        assert( !ds.FindDataElement( toverlaydata ) );
        const DataElement& overlayrows = ds.GetDataElement( toverlayrows );
        const DataElement& overlaycols = ds.GetDataElement( toverlaycols );
        assert( ds.FindDataElement( toverlaybitpos ) );
        const DataElement& overlaybitpos = ds.GetDataElement( toverlaybitpos );
        if( !overlayrows.IsEmpty() && !overlaycols.IsEmpty() && !overlaybitpos.IsEmpty() )
          {
          ++numoverlays;
          overlaylist.push_back( overlay.GetGroup() );
          }
        }
        // Move on to the next possible one:
        overlay.SetGroup( (uint16_t)(overlay.GetGroup() + 2) );
        // reset to element 0x0 just in case...
        overlay.SetElement( 0 );
      }
    }

  // at most one out of two :
  assert( numoverlays < 0x00ff / 2 );
  // PS 3.3 - 2004:
  // C.9.2 Overlay plane module
  // Each Overlay Plane is one bit deep. Sixteen separate Overlay Planes may be associated with an
  // Image or exist as Standalone Overlays in a Series
  assert( numoverlays <= 16 );
  assert( numoverlays == overlaylist.size() );
  return numoverlays;
}

bool DoOverlays(const DataSet& ds, Pixmap& pixeldata)
{
  bool updateoverlayinfo = false;
  unsigned int numoverlays;
  std::vector<uint16_t> overlaylist;
  if( (numoverlays = GetNumberOfOverlaysInternal( ds, overlaylist )) )
    {
    pixeldata.SetNumberOfOverlays( numoverlays );

    for( unsigned int idxoverlays = 0; idxoverlays < numoverlays; ++idxoverlays )
      {
      Overlay &ov = pixeldata.GetOverlay(idxoverlays);
      uint16_t currentoverlay = overlaylist[idxoverlays];
      Tag overlay(0x6000,0x0000);
      overlay.SetGroup( currentoverlay );
      const DataElement &de = ds.FindNextDataElement( overlay );
      assert( !(currentoverlay % 2) ); // 0x6001 is not an overlay...
      // Now loop on all element from this current group:
      DataElement de2 = de;
      while( de2.GetTag().GetGroup() == currentoverlay )
        {
        ov.Update(de2);
        overlay.SetElement( (uint16_t)(de2.GetTag().GetElement() + 1) );
        de2 = ds.FindNextDataElement( overlay );
        }

      // Let's decode it:
      std::ostringstream unpack;
      ov.Decompress( unpack );
      std::string s = unpack.str();
      //size_t l = s.size();
      // The following line will fail with images like XA_GE_JPEG_02_with_Overlays.dcm
      // since the overlays are stored in the unused bit of the PixelData
      if( !ov.IsEmpty() )
        {
        //assert( unpack.str().size() / 8 == ((ov.GetRows() * ov.GetColumns()) + 7 ) / 8 );
        assert( ov.IsInPixelData( ) == false );
        }
      else
        {
        gdcmDebugMacro( "This image does not contains Overlay in the 0x60xx tags. "
          << "Instead the overlay is stored in the unused bit of the Pixel Data. "
          << "This is not supported right now"
          << std::endl );
        ov.IsInPixelData( true );
        // make sure Overlay is valid
        if( ov.GetBitsAllocated() != pixeldata.GetPixelFormat().GetBitsAllocated() )
          {
          gdcmWarningMacro( "Bits Allocated are wrong. Correcting." );
          ov.SetBitsAllocated( pixeldata.GetPixelFormat().GetBitsAllocated() );
          }

        if( !ov.GrabOverlayFromPixelData(ds) )
          {
          gdcmErrorMacro( "Could not extract Overlay from Pixel Data" );
          //throw Exception("TODO: Could not extract Overlay Data");
          }
        updateoverlayinfo = true;
        }
      }
    //std::cout << "Num of Overlays: " << numoverlays << std::endl;
    }

  // Now is good time to do some cleanup (eg. DX_GE_FALCON_SNOWY-VOI.dcm).
  const PixelFormat &pf = pixeldata.GetPixelFormat();
  // Yes I am using a call in the for() loop, because I internally modify the
  // number of overlays:
  for( size_t ov_idx = pixeldata.GetNumberOfOverlays(); ov_idx != 0; --ov_idx )
    {
    size_t ov = ov_idx - 1;
    const Overlay& o = pixeldata.GetOverlay(ov);
    if( o.IsInPixelData() )
      {
      unsigned short obp = o.GetBitPosition();
      if( obp < pf.GetBitsStored() )
        {
        pixeldata.RemoveOverlay( ov );
        gdcmWarningMacro( "Invalid BitPosition: " << obp << " for overlay #" << ov << " removing it." );
        }
      }
    }

  if( updateoverlayinfo )
    {
    for( size_t ov = 0; ov < pixeldata.GetNumberOfOverlays(); ++ov )
      {
      Overlay& o = pixeldata.GetOverlay(ov);
      // We need to update information
      if( o.GetBitsAllocated() == 16 )
        {
        o.SetBitsAllocated( 1 );
        o.SetBitPosition( 0 );
        }
      else
        {
        gdcmErrorMacro( "Overlay is not supported" );
        return false;
        }
      }
    }

  return true;
}

bool PixmapReader::ReadImage(MediaStorage const &ms)
{
  return ReadImageInternal(ms);
}

bool PixmapReader::ReadImageInternal(MediaStorage const &ms, bool handlepixeldata )
{
  const DataSet &ds = F->GetDataSet();
  std::string conversion;

  bool isacrnema = false;
  const Tag trecognitioncode(0x0008,0x0010);
  if( ds.FindDataElement( trecognitioncode ) && !ds.GetDataElement( trecognitioncode ).IsEmpty() )
    {
    // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
    // PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm
    gdcmDebugMacro( "Mixture of ACR NEMA and DICOM file" );
    isacrnema = true;
    const char *str = ds.GetDataElement( trecognitioncode ).GetByteValue()->GetPointer();
    assert( strncmp( str, "ACR-NEMA", strlen( "ACR-NEMA" ) ) == 0 ||
      strncmp( str, "ACRNEMA", strlen( "ACRNEMA" ) ) == 0 );
    (void)str;//warning removal
    }

  std::vector<unsigned int> vdims = ImageHelper::GetDimensionsValue(*F);
  unsigned int numberofframes = vdims[2];
  // What should I do when numberofframes == 0 ?
  if( numberofframes > 1 )
    {
    PixelData->SetNumberOfDimensions(3);
    PixelData->SetDimension(2, numberofframes );
    }
  else
    {
    gdcmDebugMacro( "NumberOfFrames was specified with a value of: "
      << numberofframes );
    PixelData->SetNumberOfDimensions(2);
    }

  // 2. What are the col & rows:
  PixelData->SetDimension(0, vdims[0] );
  PixelData->SetDimension(1, vdims[1] );

  // 3. Pixel Format ?
  PixelFormat pf;
  // D 0028|0002 [US] [Samples per Pixel] [1]
    {
    Attribute<0x0028,0x0002> at = { 1 }; // By default assume 1 Samples Per Pixel
    at.SetFromDataSet( ds );
    pf.SetSamplesPerPixel( at.GetValue() );
    }

  if( ms == MediaStorage::MRSpectroscopyStorage )
    {
    pf.SetScalarType( PixelFormat::FLOAT32 );
    }
  else
    {
    assert( MediaStorage::IsImage( ms ) );
    // D 0028|0100 [US] [Bits Allocated] [16]
    //pf.SetBitsAllocated(
    //  ReadUSFromTag( Tag(0x0028, 0x0100), ss, conversion ) );
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0100) );
    Attribute<0x0028,0x0100> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetBitsAllocated( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0100), ss, conversion ) );
    }

    // D 0028|0101 [US] [Bits Stored] [12]
    //pf.SetBitsStored(
    //  ReadUSFromTag( Tag(0x0028, 0x0101), ss, conversion ) );
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0101) );
    Attribute<0x0028,0x0101> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetBitsStored( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0101), ss, conversion ) );
    }

    // D 0028|0102 [US] [High Bit] [11]
    //pf.SetHighBit(
    //  ReadUSFromTag( Tag(0x0028, 0x0102), ss, conversion ) );
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0102) );
    Attribute<0x0028,0x0102> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetHighBit( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0102), ss, conversion ) );
    }

    // D 0028|0103 [US] [Pixel Representation] [0]
    //Tag tpixelrep(0x0028, 0x0103);
    //if( ds.FindDataElement( tpixelrep ) && !ds.GetDataElement( tpixelrep ).IsEmpty() )
      {
      //pf.SetPixelRepresentation(
      //  ReadUSFromTag( Tag(0x0028, 0x0103), ss, conversion ) );
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0103) );
    Attribute<0x0028,0x0103> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetPixelRepresentation( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0103), ss, conversion ) );

      }
//    else
//      {
//      gdcmWarningMacro( "Pixel Representation was not found. Default to Unsigned Pixel Representation" );
//      pf.SetPixelRepresentation( 0 );
//      }
    }

  // 5. Photometric Interpretation
  // D 0028|0004 [CS] [Photometric Interpretation] [MONOCHROME2 ]
  const Tag tphotometricinterpretation(0x0028, 0x0004);
  const ByteValue *photometricinterpretation
    = ImageHelper::GetPointerFromElement( tphotometricinterpretation, *F );
  PhotometricInterpretation pi = PhotometricInterpretation::UNKNOW;
  if( photometricinterpretation )
    {
    std::string photometricinterpretation_str(
      photometricinterpretation->GetPointer(),
      photometricinterpretation->GetLength() );
    pi = PhotometricInterpretation::GetPIType( photometricinterpretation_str.c_str() );
    // http://www.dominator.com/assets/003/5278.pdf
    // JPEG 2000 lossless YUV_RCT
    if( pi == PhotometricInterpretation::PI_END )
      {
      gdcmWarningMacro( "Discarding suspicious PhotometricInterpretation found: "
        << photometricinterpretation_str );
      }
    }
  // try again harder:
  if( !photometricinterpretation || pi == PhotometricInterpretation::PI_END )
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
      gdcmWarningMacro( "No PhotometricInterpretation found, default to ARGB" );
      pi = PhotometricInterpretation::ARGB;
      }
    else
      {
      gdcmWarningMacro( "Impossible value for Samples Per Pixel: " << pf.GetSamplesPerPixel() );
      return false;
      }
    }
  assert( pi != PhotometricInterpretation::PI_END );
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
      gdcmWarningMacro( "Cannot recognize image type. Does not looks like"
        "ACR-NEMA and is missing both Sample Per Pixel AND PhotometricInterpretation."
        "Please report" );
      return false;
      }
    }
  assert ( pf.GetSamplesPerPixel() != 0 );
  // Very important to set the PixelFormat here before PlanarConfiguration
  PixelData->SetPixelFormat( pf );
  pf = PixelData->GetPixelFormat();
  if( !pf.IsValid() )
    {
    return false;
    }
  if( pi == PhotometricInterpretation::UNKNOW ) return false;
  PixelData->SetPhotometricInterpretation( pi );

  // 4. Planar Configuration
  // D 0028|0006 [US] [Planar Configuration] [1]
  const Tag planarconfiguration = Tag(0x0028, 0x0006);
  // FIXME: Whatif planaconfiguration is send in a grayscale image... it would be empty...
  // well hopefully :(
  if( ds.FindDataElement( planarconfiguration ) && !ds.GetDataElement( planarconfiguration ).IsEmpty() )
    {
    const DataElement& de = ds.GetDataElement( planarconfiguration );
    Attribute<0x0028,0x0006> at = { 0 };
    at.SetFromDataElement( de );

    //unsigned int pc = ReadUSFromTag( planarconfiguration, ss, conversion );
    unsigned int pc = at.GetValue();
    if( pc && PixelData->GetPixelFormat().GetSamplesPerPixel() != 3 )
      {
      gdcmDebugMacro( "Cannot have PlanarConfiguration=1, when Sample Per Pixel != 3" );
      pc = 0;
      }
    PixelData->SetPlanarConfiguration( pc );
    }


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
  // 4. Palette Color Lookup Table Descriptor
  if ( pi == PhotometricInterpretation::PALETTE_COLOR )
    {
    //const DataElement& modlutsq = ds.GetDataElement( Tag(0x0028,0x3000) );
    //const SequenceOfItems* sq = modlutsq.GetSequenceOfItems();
    //SequenceOfItems::ConstIterator it = sq->Begin();
    //const DataSet &ds = it->GetNestedDataSet();

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
    if( ! lut->Initialized() ) return false;
    PixelData->SetLUT(*lut);
    }
  // TODO
  //assert( pi.GetSamplesPerPixel() == pf.GetSamplesPerPixel() );

  // 5.5 Do IconImage if any
  assert( PixelData->GetIconImage().IsEmpty() );
  DoIconImage(ds, *PixelData);

  // 6. Do the Curves if any
  DoCurves(ds, *PixelData);

  // 7. Do the Overlays if any
  if( !DoOverlays(ds, *PixelData) )
    {
    return false;
    }

  // 8. Do the PixelData
  if( handlepixeldata )
    {
    if( ms == MediaStorage::MRSpectroscopyStorage )
      {
      const Tag spectdata = Tag(0x5600, 0x0020);
      if( !ds.FindDataElement( spectdata ) )
        {
        gdcmWarningMacro( "No Spectroscopy Data Found" );
        return false;
        }
      const DataElement& xde = ds.GetDataElement( spectdata );
      //bool need = PixelData->GetTransferSyntax() == TransferSyntax::ImplicitVRBigEndianPrivateGE;
      //PixelData->SetNeedByteSwap( need );
      PixelData->SetDataElement( xde );
      }
    else
      {
      const Tag pixeldata = Tag(0x7fe0, 0x0010);
      if( !ds.FindDataElement( pixeldata ) )
        {
        gdcmWarningMacro( "No Pixel Data Found" );
        return false;
        }
      const DataElement& xde = ds.GetDataElement( pixeldata );
      bool need = PixelData->GetTransferSyntax() == TransferSyntax::ImplicitVRBigEndianPrivateGE;
      PixelData->SetNeedByteSwap( need );
      PixelData->SetDataElement( xde );

      // FIXME:
      // We should check that when PixelData is RAW that Col * Dim == PixelData->GetLength()
      //PixelFormat guesspf = PixelFormat->GuessPixelFormat();

      }

    const unsigned int *dims = PixelData->GetDimensions();
    if( dims[0] == 0 || dims[1] == 0 )
      {
      // Pseudo-declared JPEG SC image storage. Let's fix col/row/pf/pi
      JPEGCodec jpeg;
      if( jpeg.CanDecode( PixelData->GetTransferSyntax() ) )
        {
        std::stringstream ss;
        const DataElement &de = PixelData->GetDataElement();
        //const ByteValue *bv = de.GetByteValue();
        const SequenceOfFragments *sqf = de.GetSequenceOfFragments();
        if( !sqf )
          {
          // TODO: It would be nice to recognize file such as JPEGDefinedLengthSequenceOfFragments.dcm
          gdcmDebugMacro( "File is declared as JPEG compressed but does not contains Fragmens explicitely." );
          return false;
          }
        sqf->WriteBuffer( ss );
        //std::string s( bv->GetPointer(), bv->GetLength() );
        //is.str( s );
        PixelFormat jpegpf ( PixelFormat::UINT8 ); // usual guess...
        jpeg.SetPixelFormat( jpegpf );
        TransferSyntax ts;
        bool b = jpeg.GetHeaderInfo( ss, ts );
        if( b )
          {
          std::vector<unsigned int> v(3);
          v[0] = PixelData->GetDimensions()[0];
          v[1] = PixelData->GetDimensions()[1];
          v[2] = PixelData->GetDimensions()[2];
          assert( jpeg.GetDimensions()[0] );
          assert( jpeg.GetDimensions()[1] );
          v[0] = jpeg.GetDimensions()[0];
          v[1] = jpeg.GetDimensions()[1];
          PixelData->SetDimensions( &v[0] );
          //PixelData->SetPixelFormat( jpeg.GetPixelFormat() );
          //PixelData->SetPhotometricInterpretation( jpeg.GetPhotometricInterpretation() );
          assert( PixelData->IsTransferSyntaxCompatible( ts ) );
          }
        else
          {
          gdcmDebugMacro( "Columns or Row was found to be 0. Cannot compute dimension." );
          return false;
          }
        }
      else
        {
        gdcmDebugMacro( "Columns or Row was found to be 0. Cannot compute dimension." );
        return false;
        }
      }
    }

  // Let's be smart when computing the lossyflag (0028,2110) 
  // LossyImageCompression
  Attribute<0x0028,0x2110> licat;
  bool lossyflag = false;
  bool haslossyflag = false;
  if( ds.FindDataElement( licat.GetTag() ) )
    {
    haslossyflag = true;
    licat.SetFromDataSet( ds ); // could be empty
    const CSComp & v = licat.GetValue();
    lossyflag = atoi( v.c_str() ) == 1;
    PixelData->SetLossyFlag(lossyflag);
    }

  // Two cases:
  // - DataSet did not specify the lossyflag
  // - DataSet specify it to be 0, but there is still a change it could be wrong:
  if( !haslossyflag || !lossyflag )
    {
    PixelData->ComputeLossyFlag();
    if( PixelData->IsLossy() && (!lossyflag && haslossyflag ) )
      {
      // We always prefer the setting from the stream...
      gdcmWarningMacro( "DataSet set LossyFlag to 0, while Codec made the stream lossy" );
      }
    }

  return true;
}

bool PixmapReader::ReadACRNEMAImage()
{
  const DataSet &ds = F->GetDataSet();
  std::stringstream ss;
  std::string conversion;

  // Ok we have the dataset let's feed the Image (PixelData)
  // 1. First find how many dimensions there is:
  // D 0028|0005 [SS] [Image Dimensions (RET)] [2]
  const Tag timagedimensions = Tag(0x0028, 0x0005);
  if( ds.FindDataElement( timagedimensions ) )
    {
    const DataElement& de0 = ds.GetDataElement( timagedimensions );
    Attribute<0x0028,0x0005> at0 = { 0 };
    at0.SetFromDataElement( de0 );
    assert( at0.GetNumberOfValues() == 1 );
    unsigned short imagedimensions = at0.GetValue();
    //assert( imagedimensions == ReadSSFromTag( timagedimensions, ss, conversion ) );
    if ( imagedimensions == 3 )
      {
      PixelData->SetNumberOfDimensions(3);
      // D 0028|0012 [US] [Planes] [262]
      const DataElement& de1 = ds.GetDataElement( Tag(0x0028, 0x0012) );
      Attribute<0x0028,0x0012> at1 = { 0 };
      at1.SetFromDataElement( de1 );
      assert( at1.GetNumberOfValues() == 1 );
      PixelData->SetDimension(2, at1.GetValue() );
      //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0012), ss, conversion ) );
      }
    else if ( imagedimensions == 2 )
      {
      PixelData->SetNumberOfDimensions(2);
      }
    else
      {
      gdcmErrorMacro( "Unhandled Image Dimensions: " << imagedimensions );
      return false;
      }
    }
  else
    {
    gdcmWarningMacro( "Attempting a guess for the number of dimensions" );
    PixelData->SetNumberOfDimensions( 2 );
    }

  // 2. What are the col & rows:
  // D 0028|0011 [US] [Columns] [512]
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0011) );
    Attribute<0x0028,0x0011> at = { 0 };
    at.SetFromDataSet( ds );
    PixelData->SetDimension(0, at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0011), ss, conversion ) );
    }

  // D 0028|0010 [US] [Rows] [512]
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0010) );
    Attribute<0x0028,0x0010> at = { 0 };
    at.SetFromDataSet( ds );
    PixelData->SetDimension(1, at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0010), ss, conversion ) );
    }

  // This is the definition of an ACR NEMA image:
  // D 0008|0010 [LO] [Recognition Code (RET)] [ACR-NEMA 2.0]
  // LIBIDO compatible code:
  // D 0008|0010 [LO] [Recognition Code (RET)] [ACRNEMA_LIBIDO_1.1]
  const Tag trecognitioncode(0x0008,0x0010);
  if( ds.FindDataElement( trecognitioncode ) && !ds.GetDataElement( trecognitioncode ).IsEmpty() )
    {
    const ByteValue *libido = ds.GetDataElement(trecognitioncode).GetByteValue();
    assert( libido );
    std::string libido_str( libido->GetPointer(), libido->GetLength() );
    assert( libido_str != "CANRME_AILIBOD1_1." );
    if( strcmp(libido_str.c_str() , "ACRNEMA_LIBIDO_1.1") == 0 || strcmp(libido_str.c_str() , "ACRNEMA_LIBIDO_1.0") == 0 )
      {
      // Swap Columns & Rows
      // assert( PixelData->GetNumberOfDimensions() == 2 );
      const unsigned int *dims = PixelData->GetDimensions();
      unsigned int tmp = dims[0];
      PixelData->SetDimension(0, dims[1] );
      PixelData->SetDimension(1, tmp );
      }
    else
      {
      assert( libido_str == "ACR-NEMA 2.0"
           || libido_str == "ACR-NEMA 1.0" );
      }
    }
  else
    {
    gdcmWarningMacro(
      "Reading as ACR NEMA an image which does not look likes ACR NEMA" );
    // File: acc-max.dcm is it ACR or DICOM ?
    // assert(0);
    }

  // 3. Pixel Format ?
  PixelFormat pf;
  // D 0028|0100 [US] [Bits Allocated] [16]
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0100) );
    Attribute<0x0028,0x0100> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetBitsAllocated( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0100), ss, conversion ) );
    }

  // D 0028|0101 [US] [Bits Stored] [12]
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0101) );
    Attribute<0x0028,0x0101> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetBitsStored( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0101), ss, conversion ) );
    }

  // D 0028|0102 [US] [High Bit] [11]
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0102) );
    Attribute<0x0028,0x0102> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetHighBit( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0102), ss, conversion ) );
    }

  // D 0028|0103 [US] [Pixel Representation] [0]
    {
    //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0103) );
    Attribute<0x0028,0x0103> at = { 0 };
    at.SetFromDataSet( ds );
    pf.SetPixelRepresentation( at.GetValue() );
    //assert( at.GetValue() == ReadUSFromTag( Tag(0x0028, 0x0103), ss, conversion ) );
    }

  PixelData->SetPixelFormat( pf );

  // 4. Do the Curves/Overlays if any
  DoCurves(ds, *PixelData);
  DoOverlays(ds, *PixelData);

  // 5. Do the PixelData
  const Tag pixeldata = Tag(0x7fe0, 0x0010);
  if( !ds.FindDataElement( pixeldata ) )
    {
    gdcmWarningMacro( "No Pixel Data Found" );
    return false;
    }
  const DataElement& de = ds.GetDataElement( pixeldata );
  if ( de.GetVR() == VR::OW )
    {
    //assert(0);
    //PixelData->SetNeedByteSwap(true);
    }
  PixelData->SetDataElement( de );

  // There is no such thing as Photometric Interpretation and
  // Planar Configuration in ACR NEMA so let's default to something ...
  PixelData->SetPhotometricInterpretation(
    PhotometricInterpretation::MONOCHROME2 );
  PixelData->SetPlanarConfiguration(0);
  const Tag planarconfiguration(0x0028, 0x0006);
  if( ds.FindDataElement( planarconfiguration ) && !ds.GetDataElement( planarconfiguration ).IsEmpty() )
    {
    //const DataElement& de = ds.GetDataElement( planarconfiguration );
    Attribute<0x0028,0x0006> at = { 0 };
    at.SetFromDataSet( ds );

    //unsigned int pc = ReadUSFromTag( planarconfiguration, ss, conversion );
    unsigned int pc = at.GetValue();
    if( pc && PixelData->GetPixelFormat().GetSamplesPerPixel() != 3 )
      {
      gdcmDebugMacro( "Cannot have PlanarConfiguration=1, when Sample Per Pixel != 3" );
      pc = 0;
      }
    PixelData->SetPlanarConfiguration( pc );
    }

  const Tag tphotometricinterpretation(0x0028, 0x0004);
  // Some funny ACR NEMA file have PhotometricInterpretation ...
  if( ds.FindDataElement( tphotometricinterpretation ) && !ds.GetDataElement( tphotometricinterpretation ).IsEmpty() )
    {
    const ByteValue *photometricinterpretation
      = ds.GetDataElement( tphotometricinterpretation ).GetByteValue();
    assert( photometricinterpretation );
    std::string photometricinterpretation_str(
      photometricinterpretation->GetPointer(),
      photometricinterpretation->GetLength() );
    PhotometricInterpretation pi(
      PhotometricInterpretation::GetPIType(
        photometricinterpretation_str.c_str()));
    PixelData->SetPhotometricInterpretation( pi );
    }
  else
    {
    // Wild guess:
    if( PixelData->GetPixelFormat().GetSamplesPerPixel() == 1 )
      {
      assert( PixelData->GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME2 );
      // No need...
      //PixelData->SetPhotometricInterpretation( PhotometricInterpretation::MONOCHROME2 );
      }
    else if( PixelData->GetPixelFormat().GetSamplesPerPixel() == 3 )
      {
      // LIBIDO-24-ACR_NEMA-Rectangle.dcm
      PixelData->SetPhotometricInterpretation( PhotometricInterpretation::RGB );
      }
    else if( PixelData->GetPixelFormat().GetSamplesPerPixel() == 4 )
      {
      PixelData->SetPhotometricInterpretation( PhotometricInterpretation::ARGB );
      }
    else
      {
      gdcmErrorMacro( "Cannot handle Samples Per Pixel=" << PixelData->GetPixelFormat().GetSamplesPerPixel() );
      return false;
      }
    }

  return true;
}


} // end namespace gdcm
