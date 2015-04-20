/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIconImageFilter.h"
#include "gdcmIconImage.h"
#include "gdcmAttribute.h"
#include "gdcmPrivateTag.h"
#include "gdcmJPEGCodec.h"

namespace gdcm
{

class IconImageFilterInternals
{
public:
  std::vector < SmartPointer< IconImage > > icons;
};

IconImageFilter::IconImageFilter():F(new File),Internals(new IconImageFilterInternals)
{
}

IconImageFilter::~IconImageFilter()
{
  delete Internals;
}

void IconImageFilter::ExtractIconImages()
{
  const DataSet &rootds = F->GetDataSet();
  const FileMetaInformation &header = F->GetHeader();
  const TransferSyntax &ts = header.GetDataSetTransferSyntax();

  // PICKER-16-MONO2-Nested_icon.dcm
  const Tag ticonimage(0x0088,0x0200);

  // Public Icon
  if( rootds.FindDataElement( ticonimage ) )
    {
    const DataElement &iconimagesq = rootds.GetDataElement( ticonimage );
    SmartPointer<SequenceOfItems> sq = iconimagesq.GetValueAsSQ();
    // Is SQ empty ?
    if( sq )
      {
      gdcmAssertAlwaysMacro( sq->GetNumberOfItems() == 1 );

      SmartPointer< IconImage > si1 = new IconImage;
      IconImage &pixeldata = *si1;
      SequenceOfItems::ConstIterator it = sq->Begin();
      const DataSet &ds = it->GetNestedDataSet();

        {
        Attribute<0x0028,0x0011> at = { 0 };
        at.SetFromDataSet( ds );
        pixeldata.SetDimension(0, at.GetValue() );
        }

        {
        Attribute<0x0028,0x0010> at = { 0 };
        at.SetFromDataSet( ds );
        pixeldata.SetDimension(1, at.GetValue() );
        }

      PixelFormat pf;
        {
        Attribute<0x0028,0x0100> at = { 0 };
        at.SetFromDataSet( ds );
        pf.SetBitsAllocated( at.GetValue() );
        }
        {
        Attribute<0x0028,0x0101> at = { 0 };
        at.SetFromDataSet( ds );
        pf.SetBitsStored( at.GetValue() );
        }
        {
        Attribute<0x0028,0x0102> at = { 0 };
        at.SetFromDataSet( ds );
        pf.SetHighBit( at.GetValue() );
        }
        {
        Attribute<0x0028,0x0103> at = { 0 };
        at.SetFromDataSet( ds );
        pf.SetPixelRepresentation( at.GetValue() );
        }
        {
        Attribute<0x0028,0x0002> at = { 1 };
        at.SetFromDataSet( ds );
        pf.SetSamplesPerPixel( at.GetValue() );
        }
      pixeldata.SetPixelFormat( pf );
      // D 0028|0004 [CS] [Photometric Interpretation] [MONOCHROME2 ]
      const Tag tphotometricinterpretation(0x0028, 0x0004);
      assert( ds.FindDataElement( tphotometricinterpretation ) );
      const ByteValue *photometricinterpretation =
        ds.GetDataElement( tphotometricinterpretation ).GetByteValue();
      std::string photometricinterpretation_str(
        photometricinterpretation->GetPointer(),
        photometricinterpretation->GetLength() );
      PhotometricInterpretation pi(
        PhotometricInterpretation::GetPIType(
          photometricinterpretation_str.c_str()));
      assert( pi != PhotometricInterpretation::UNKNOW);
      pixeldata.SetPhotometricInterpretation( pi );

      //
      if ( pi == PhotometricInterpretation::PALETTE_COLOR )
        {
        SmartPointer<LookupTable> lut = new LookupTable;
        const Tag testseglut(0x0028, (0x1221 + 0));
        if( ds.FindDataElement( testseglut ) )
          {
          gdcmAssertAlwaysMacro(0 && "Please report this image");
          }
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
            assert( check == lut_raw->GetLength() ); (void)check;
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
      pixeldata.SetTransferSyntax( ts );
      Internals->icons.push_back( pixeldata );
      }
    }

  // AMIInvalidPrivateDefinedLengthSQasUN.dcm
  // GE_CT_With_Private_compressed-icon.dcm
  // MR_GE_with_Private_Compressed_Icon_0009_1110.dcm
  // FIXME:
  // Not all tags from the private sequence can be handled
  // For instance an icon has a window center/width ...
  const PrivateTag tgeiconimage(0x0009,0x0010,"GEIIS");

  // Private Icon
  if( rootds.FindDataElement( tgeiconimage ) )
    {
    const DataElement &iconimagesq = rootds.GetDataElement( tgeiconimage );
    //const SequenceOfItems* sq = iconimagesq.GetSequenceOfItems();
    SmartPointer<SequenceOfItems> sq = iconimagesq.GetValueAsSQ();
    // Is SQ empty ?
    assert( sq );
    if( !sq ) return;
    SmartPointer< IconImage > si1 = new IconImage;
    IconImage &pixeldata = *si1;

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

    PixelFormat pf1;
    // D 0028|0100 [US] [Bits Allocated] [16]
      {
      //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0100) );
      Attribute<0x0028,0x0100> at = { 0 };
      at.SetFromDataSet( ds );
      pf1.SetBitsAllocated( at.GetValue() );
      }
    // D 0028|0101 [US] [Bits Stored] [12]
      {
      //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0101) );
      Attribute<0x0028,0x0101> at = { 0 };
      at.SetFromDataSet( ds );
      pf1.SetBitsStored( at.GetValue() );
      }
    // D 0028|0102 [US] [High Bit] [11]
      {
      //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0102) );
      Attribute<0x0028,0x0102> at = { 0 };
      at.SetFromDataSet( ds );
      pf1.SetHighBit( at.GetValue() );
      }
    // D 0028|0103 [US] [Pixel Representation] [0]
      {
      //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0103) );
      Attribute<0x0028,0x0103> at = { 0 };
      at.SetFromDataSet( ds );
      pf1.SetPixelRepresentation( at.GetValue() );
      }
    // (0028,0002) US 1                                        #   2, 1 SamplesPerPixel
      {
      //const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0002) );
      Attribute<0x0028,0x0002> at = { 1 };
      at.SetFromDataSet( ds );
      pf1.SetSamplesPerPixel( at.GetValue() );
      }
    pixeldata.SetPixelFormat( pf1 );
    // D 0028|0004 [CS] [Photometric Interpretation] [MONOCHROME2 ]
    const Tag tphotometricinterpretation(0x0028, 0x0004);
    assert( ds.FindDataElement( tphotometricinterpretation ) );
    const ByteValue *photometricinterpretation = ds.GetDataElement( tphotometricinterpretation ).GetByteValue();
    std::string photometricinterpretation_str(
      photometricinterpretation->GetPointer(),
      photometricinterpretation->GetLength() );
    PhotometricInterpretation pi(
      PhotometricInterpretation::GetPIType(
        photometricinterpretation_str.c_str()));
    assert( pi != PhotometricInterpretation::UNKNOW);
    pixeldata.SetPhotometricInterpretation( pi );
    const Tag tpixeldata = Tag(0x7fe0, 0x0010);
    assert( ds.FindDataElement( tpixeldata ) );
      {
      const DataElement& de = ds.GetDataElement( tpixeldata );
#if 0
      JPEGCodec jpeg;
      jpeg.SetPhotometricInterpretation( pixeldata.GetPhotometricInterpretation() );
      jpeg.SetPlanarConfiguration( 0 );
      PixelFormat pf = pixeldata.GetPixelFormat();
      // Apparently bits stored can only be 8 or 12:
      if( pf.GetBitsStored() == 16 )
        {
        pf.SetBitsStored( 12 );
        }
      jpeg.SetPixelFormat( pf );
      DataElement de2;
      jpeg.Decode( de, de2);
      pixeldata.SetDataElement( de2 );
#endif
#if 0
      JPEGCodec jpeg;
      jpeg.SetPhotometricInterpretation( pixeldata.GetPhotometricInterpretation() );
      jpeg.SetPixelFormat( pixeldata.GetPixelFormat() );
      DataElement de2;
      jpeg.Decode( de, de2);
      PixelFormat &pf2 = jpeg.GetPixelFormat();
#endif
#if 1
      std::istringstream is;
      const ByteValue *bv = de.GetByteValue();
      assert( bv );
      is.str( std::string( bv->GetPointer(), bv->GetLength() ) );
      TransferSyntax jpegts;
      JPEGCodec jpeg;
      jpeg.SetPixelFormat( pf1 ); // important to initialize
      bool b = jpeg.GetHeaderInfo( is, jpegts );
      if( !b )
        {
        assert( 0 );
        }
      //jpeg.GetPixelFormat().Print (std::cout);
      pixeldata.SetPixelFormat( jpeg.GetPixelFormat() );
      // Set appropriate transfer Syntax
      pixeldata.SetTransferSyntax( jpegts );
#endif
      pixeldata.SetDataElement( de );
      }
    Internals->icons.push_back( pixeldata );
    }

  // AFAIK this icon SQ is undocumented , but I found it in:
  // gdcmDataExtra/gdcmBreakers/2929J888_8b_YBR_RLE_PlanConf0_breaker.dcm
  // aka 'SmallPreview'
  // The SQ contains a DataElement:
  // (0002,0010) UI [1.2.840.10008.1.2.1]                          # 20,1 Transfer Syntax UID
  // sigh...
  const PrivateTag tgeiconimage2(0x6003,0x0010,"GEMS_Ultrasound_ImageGroup_001");

  if( rootds.FindDataElement( tgeiconimage2 ) )
    {
    const DataElement &iconimagesq = rootds.GetDataElement( tgeiconimage2 );
    //const SequenceOfItems* sq = iconimagesq.GetSequenceOfItems();
    SmartPointer<SequenceOfItems> sq = iconimagesq.GetValueAsSQ();
    // Is SQ empty ?
    assert( sq );
    if( !sq ) return;

    SmartPointer< IconImage > si1 = new IconImage;
    IconImage &pixeldata = *si1;

    SequenceOfItems::ConstIterator it = sq->Begin();
    const DataSet &ds = it->GetNestedDataSet();

    // D 0028|0011 [US] [Columns] [512]
      {
      const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0011) );
      Attribute<0x0028,0x0011> at;
      at.SetFromDataElement( de );
      pixeldata.SetDimension(0, at.GetValue() );
      }

    // D 0028|0010 [US] [Rows] [512]
      {
      const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0010) );
      Attribute<0x0028,0x0010> at;
      at.SetFromDataElement( de );
      pixeldata.SetDimension(1, at.GetValue() );
      }

    PixelFormat pf;
    // D 0028|0100 [US] [Bits Allocated] [16]
      {
      const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0100) );
      Attribute<0x0028,0x0100> at;
      at.SetFromDataElement( de );
      pf.SetBitsAllocated( at.GetValue() );
      }
    // D 0028|0101 [US] [Bits Stored] [12]
      {
      const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0101) );
      Attribute<0x0028,0x0101> at;
      at.SetFromDataElement( de );
      pf.SetBitsStored( at.GetValue() );
      }
    // D 0028|0102 [US] [High Bit] [11]
      {
      const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0102) );
      Attribute<0x0028,0x0102> at;
      at.SetFromDataElement( de );
      pf.SetHighBit( at.GetValue() );
      }
    // D 0028|0103 [US] [Pixel Representation] [0]
      {
      const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0103) );
      Attribute<0x0028,0x0103> at;
      at.SetFromDataElement( de );
      pf.SetPixelRepresentation( at.GetValue() );
      }
    // (0028,0002) US 1                                        #   2, 1 SamplesPerPixel
      {
      const DataElement& de = ds.GetDataElement( Tag(0x0028, 0x0002) );
      Attribute<0x0028,0x0002> at;
      at.SetFromDataElement( de );
      pf.SetSamplesPerPixel( at.GetValue() );
      }
    pixeldata.SetPixelFormat( pf );
    // D 0028|0004 [CS] [Photometric Interpretation] [MONOCHROME2 ]
    const Tag tphotometricinterpretation(0x0028, 0x0004);
    assert( ds.FindDataElement( tphotometricinterpretation ) );
    const ByteValue *photometricinterpretation = ds.GetDataElement( tphotometricinterpretation ).GetByteValue();
    std::string photometricinterpretation_str(
      photometricinterpretation->GetPointer(),
      photometricinterpretation->GetLength() );
    PhotometricInterpretation pi(
      PhotometricInterpretation::GetPIType(
        photometricinterpretation_str.c_str()));
    assert( pi != PhotometricInterpretation::UNKNOW);
    pixeldata.SetPhotometricInterpretation( pi );
    //const Tag tpixeldata = Tag(0x7fe0, 0x0010);
    const PrivateTag tpixeldata(0x6003,0x0011,"GEMS_Ultrasound_ImageGroup_001");
    assert( ds.FindDataElement( tpixeldata ) );
      {
      const DataElement& de = ds.GetDataElement( tpixeldata );
      pixeldata.SetDataElement( de );
      /*
      JPEGCodec jpeg;
      jpeg.SetPhotometricInterpretation( pixeldata.GetPhotometricInterpretation() );
      jpeg.SetPlanarConfiguration( 0 );
      PixelFormat pf = pixeldata.GetPixelFormat();
      // Apparently bits stored can only be 8 or 12:
      if( pf.GetBitsStored() == 16 )
      {
      pf.SetBitsStored( 12 );
      }
      jpeg.SetPixelFormat( pf );
      DataElement de2;
      jpeg.Decode( de, de2);
      pixeldata.SetDataElement( de2 );
       */
      }
      {
      Attribute<0x002,0x0010> at;
      at.SetFromDataSet( ds );
      TransferSyntax tstype = TransferSyntax::GetTSType( at.GetValue() );
      pixeldata.SetTransferSyntax( tstype );
      }
    Internals->icons.push_back( pixeldata );
    }
}

/*
[ICONDATA2]
PrivateCreator = VEPRO VIM 5.0 DATA
Group = 0x0055
Element = 0x0030
Data.ID     = C|0|3
Data.Type    = C|3|1
Data.Width    = I|4|2
Data.Height    = I|6|2
*/
namespace {
struct VeproData
{
  char ID[3];
  char Type;
  uint16_t Width;
  uint16_t Height;
};
}

// documentation was found in : VIM/VIMSYS/dcmviewpriv.dat
void IconImageFilter::ExtractVeproIconImages()
{
  const DataSet &rootds = F->GetDataSet();

  const PrivateTag ticon1(0x55,0x0030,"VEPRO VIF 3.0 DATA");
  const PrivateTag ticon2(0x55,0x0030,"VEPRO VIM 5.0 DATA");

  const ByteValue * bv = NULL;
  // Prefer VIF over VIM ?
  if( rootds.FindDataElement( ticon1 ) )
    {
    const DataElement &de = rootds.GetDataElement( ticon1 );
    bv = de.GetByteValue();
    }
  else if( rootds.FindDataElement( ticon2 ) )
    {
    const DataElement &de = rootds.GetDataElement( ticon2 );
    bv = de.GetByteValue();
    }

  if( bv )
    {
    const char *buf = bv->GetPointer();
    size_t len = bv->GetLength();
    VeproData data;
    memcpy(&data, buf, sizeof(data));

    const char *raw = buf + sizeof(data);

    size_t offset = 4;
    // All header starts with the letter 'RAW\0', it looks like we need
    // to skip them (all 4 of them)
    int magic = memcmp( raw, "RAW\0", 4 );
    gdcmAssertAlwaysMacro( magic == 0 );

    unsigned int dims[3] = {};
    dims[0] = data.Width;
    dims[1] = data.Height;

    assert( dims[0] * dims[1] == len - sizeof(data) - offset );

    DataElement pd;
    pd.SetByteValue( raw + offset, (uint32_t)(len - sizeof(data) - offset) );

    SmartPointer< IconImage > si1 = new IconImage;
    IconImage &pixeldata = *si1;
    pixeldata.SetDataElement( pd );

    pixeldata.SetDimension(0, dims[0] );
    pixeldata.SetDimension(1, dims[1] );

    PixelFormat pf = PixelFormat::UINT8;
    pixeldata.SetPixelFormat( pf );
    pixeldata.SetPhotometricInterpretation( PhotometricInterpretation::MONOCHROME2 );

    Internals->icons.push_back( pixeldata );
    }
}

bool IconImageFilter::Extract()
{
  Internals->icons.clear();
  ExtractIconImages();
  ExtractVeproIconImages();
  return GetNumberOfIconImages() != 0;
}

unsigned int IconImageFilter::GetNumberOfIconImages() const
{
  // what is icons are in undefined length sequence ?
  return (unsigned int)Internals->icons.size();
}

IconImage& IconImageFilter::GetIconImage( unsigned int i ) const
{
  assert( i < Internals->icons.size() );
  return *Internals->icons[i];
}

} // end namespace gdcm
