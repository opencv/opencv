/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageWriter.h"
#include "gdcmTrace.h"
#include "gdcmDataSet.h"
#include "gdcmDataElement.h"
#include "gdcmAttribute.h"
#include "gdcmUIDGenerator.h"
#include "gdcmSystem.h"
#include "gdcmImageHelper.h"
#include "gdcmLookupTable.h"
#include "gdcmItem.h"
#include "gdcmSequenceOfItems.h"

namespace gdcm
{

ImageWriter::ImageWriter()
{
  PixelData = new Image;
}

ImageWriter::~ImageWriter()
{
}

//void ImageWriter::SetImage(Image const &img)
//{
//  PixelData = img;
//}

bool ImageWriter::Write()
{
  if( !PrepareWrite() ) return false;

  //assert( Stream.is_open() );
  File& file = GetFile();
  DataSet& ds = file.GetDataSet();

  // Some Type 2 Element:
  // PatientName
  if( !ds.FindDataElement( Tag(0x0010,0x0010) ) )
    {
    DataElement de( Tag(0x0010,0x0010) );
    de.SetVR( Attribute<0x0010,0x0010>::GetVR() );
    ds.Insert( de );
    }
  // PatientID
  if( !ds.FindDataElement( Tag(0x0010,0x0020) ) )
    {
    DataElement de( Tag(0x0010,0x0020) );
    de.SetVR( Attribute<0x0010,0x0020>::GetVR() );
    ds.Insert( de );
    }
  // PatientBirthDate
  if( !ds.FindDataElement( Tag(0x0010,0x0030) ) )
    {
    DataElement de( Tag(0x0010,0x0030) );
    de.SetVR( Attribute<0x0010,0x0030>::GetVR() );
    ds.Insert( de );
    }
  // PatientSex
  if( !ds.FindDataElement( Tag(0x0010,0x0040) ) )
    {
    DataElement de( Tag(0x0010,0x0040) );
    de.SetVR( Attribute<0x0010,0x0040>::GetVR() );
    ds.Insert( de );
    }
  // Laterality
  if( false && !ds.FindDataElement( Tag(0x0020,0x0060) ) )
    {
    DataElement de( Tag(0x0020,0x0060) );
    de.SetVR( Attribute<0x0020,0x0060>::GetVR() );
    ds.Insert( de );
    }
  // StudyDate
  char date[22];
  const size_t datelen = 8;
  int res = System::GetCurrentDateTime(date);
  assert( res );
  (void)res;//warning removal
  if( !ds.FindDataElement( Tag(0x0008,0x0020) ) )
    {
    DataElement de( Tag(0x0008,0x0020) );
    // Do not copy the whole cstring:
    de.SetByteValue( date, datelen );
    de.SetVR( Attribute<0x0008,0x0020>::GetVR() );
    ds.Insert( de );
    }
  // StudyTime
  const size_t timelen = 6 + 1 + 6; // time + milliseconds
  Attribute<0x0008, 0x0030> studytime;
  if( !ds.FindDataElement( studytime.GetTag() ) )
    {
    // Do not copy the whole cstring:
    studytime.SetValue( CSComp(date+datelen, timelen) );
    ds.Insert( studytime.GetAsDataElement() );
    }
  // ReferringPhysicianName
  if( !ds.FindDataElement( Tag(0x0008,0x0090) ) )
    {
    DataElement de( Tag(0x0008,0x0090) );
    de.SetVR( Attribute<0x0008,0x0090>::GetVR() );
    ds.Insert( de );
    }
  // StudyID
  if( !ds.FindDataElement( Tag(0x0020,0x0010) ) )
    {
    // FIXME: this one is actually bad since the value is needed for DICOMDIR construction
    DataElement de( Tag(0x0020,0x0010) );
    de.SetVR( Attribute<0x0020,0x0010>::GetVR() );
    ds.Insert( de );
    }
  // AccessionNumber
  if( !ds.FindDataElement( Tag(0x0008,0x0050) ) )
    {
    DataElement de( Tag(0x0008,0x0050) );
    de.SetVR( Attribute<0x0008,0x0050>::GetVR() );
    ds.Insert( de );
    }
  // SeriesNumber
  if( !ds.FindDataElement( Tag(0x0020,0x0011) ) )
    {
    DataElement de( Tag(0x0020,0x0011) );
    de.SetVR( Attribute<0x0020,0x0011>::GetVR() );
    ds.Insert( de );
    }
  // InstanceNumber
  if( !ds.FindDataElement( Tag(0x0020,0x0013) ) )
    {
    DataElement de( Tag(0x0020,0x0013) );
    de.SetVR( Attribute<0x0020,0x0013>::GetVR() );
    ds.Insert( de );
    }

  MediaStorage ms;
  ms.SetFromFile( GetFile() );
  assert( ms != MediaStorage::MS_END );

  // Patient Orientation
  if( ms == MediaStorage::SecondaryCaptureImageStorage && !ds.FindDataElement( Tag(0x0020,0x0020) ) )
    {
    DataElement de( Tag(0x0020,0x0020) );
    de.SetVR( Attribute<0x0020,0x0020>::GetVR() );
    ds.Insert( de );
    }

  // (re)Compute MediaStorage:
  if( !ds.FindDataElement( Tag(0x0008, 0x0060) ) )
    {
    const char *modality = ms.GetModality();
    DataElement de( Tag(0x0008, 0x0060 ) );
    VL::Type strlenModality = (VL::Type)strlen(modality);
    de.SetByteValue( modality, strlenModality );
    de.SetVR( Attribute<0x0008, 0x0060>::GetVR() );
    ds.Insert( de );
    }
  else
    {
    const ByteValue *bv = ds.GetDataElement( Tag(0x0008, 0x0060 ) ).GetByteValue();
    std::string modality2;
    if( bv )
      {
      modality2 = std::string( bv->GetPointer(), bv->GetLength() );
      //assert( modality2.find( ' ' ) == std::string::npos ); // no space ...
      }
    else
      {
      // remove empty Modality, and set a new one...
      ds.Remove( Tag(0x0008, 0x0060 ) ); // Modality is Type 1 !
      assert( ms != MediaStorage::MS_END );
      }
/*
    if( modality2 != ms.GetModality() )
      {
      assert( std::string(ms.GetModality()).find( ' ' ) == std::string::npos ); // no space ...
      DataElement de( Tag(0x0008, 0x0060 ) );
      de.SetByteValue( ms.GetModality(), strlen(ms.GetModality()) );
      de.SetVR( Attribute<0x0008, 0x0060>::GetVR() );
      ds.Insert( de ); // FIXME: should we always replace ?
      // Well technically you could have a SecondaryCaptureImageStorage with a modality of NM...
      }
*/
    }
  if( !ds.FindDataElement( Tag(0x0008, 0x0064) ) )
    {
    if( ms == MediaStorage::SecondaryCaptureImageStorage )
      {
      // (0008,0064) CS [SI]                                     #   2, 1 ConversionType
      const char conversion[] = "WSD "; // FIXME
      DataElement de( Tag(0x0008, 0x0064 ) );
      VL::Type strlenConversion = (VL::Type)strlen(conversion);
      de.SetByteValue( conversion, strlenConversion );
      de.SetVR( Attribute<0x0008, 0x0064>::GetVR() );
      ds.Insert( de );
      }
    }


  Image & pixeldata = GetImage();
  PixelFormat pf = pixeldata.GetPixelFormat();
  PhotometricInterpretation pi = pixeldata.GetPhotometricInterpretation();

  // Do the Rescale Intercept & Slope
  if( pi == PhotometricInterpretation::MONOCHROME1 || pi == PhotometricInterpretation::MONOCHROME2 )
    {
    assert( pf.GetSamplesPerPixel() == 1 );
    ImageHelper::SetRescaleInterceptSlopeValue(GetFile(), pixeldata);
    if( ms == MediaStorage::RTDoseStorage && pixeldata.GetIntercept() != 0 )
      {
      return false;
      }
    }
  else
    {
    assert( pixeldata.GetIntercept() == 0 && pixeldata.GetSlope() == 1 );
    }

//      Attribute<0x0028, 0x0006> planarconfiguration;
//      planarconfiguration.SetValue( PixelData->GetPlanarConfiguration() );
//      ds.Replace( planarconfiguration.GetAsDataElement() );

  // PhotometricInterpretation
  // const Tag tphotometricinterpretation(0x0028, 0x0004);
  //if( !ds.FindDataElement( Tag(0x0028, 0x0004) ) )
    {
    //if( pi == PhotometricInterpretation::RGB
    //  || pi == PhotometricInterpretation::YBR_FULL ) // FIXME
    //  {
    //  Attribute<0x0028, 0x0006> planarconfiguration;
    //  planarconfiguration.SetValue( PixelData->GetPlanarConfiguration() );
    //  ds.Replace( planarconfiguration.GetAsDataElement() );
    //  }
    //else
    if ( pi == PhotometricInterpretation::PALETTE_COLOR )
      {
      const LookupTable &lut = PixelData->GetLUT();
      assert( lut.Initialized() );
//      assert( (pf.GetBitsAllocated() == 8  && pf.GetPixelRepresentation() == 0)
//           || (pf.GetBitsAllocated() == 16 && pf.GetPixelRepresentation() == 0) );
      // lut descriptor:
      // (0028,1101) US 256\0\16                                 #   6, 3 RedPaletteColorLookupTableDescriptor
      // (0028,1102) US 256\0\16                                 #   6, 3 GreenPaletteColorLookupTableDescriptor
      // (0028,1103) US 256\0\16                                 #   6, 3 BluePaletteColorLookupTableDescriptor
      // lut data:
      unsigned short length, subscript, bitsize;
      unsigned short rawlut8[256];
      unsigned short rawlut16[65536];
      unsigned short *rawlut = rawlut8;
      unsigned int lutlen = 256;
      if( pf.GetBitsAllocated() == 16 )
        {
        rawlut = rawlut16;
        lutlen = 65536;
        }
      unsigned int l;

      // FIXME: should I really clear rawlut each time ?
      // RED
      memset(rawlut,0,lutlen*2);
      lut.GetLUT(LookupTable::RED, (unsigned char*)rawlut, l);
      DataElement redde( Tag(0x0028, 0x1201) );
      redde.SetVR( VR::OW );
      redde.SetByteValue( (char*)rawlut, l);
      ds.Replace( redde );
      // descriptor:
      Attribute<0x0028, 0x1101, VR::US, VM::VM3> reddesc;
      lut.GetLUTDescriptor(LookupTable::RED, length, subscript, bitsize);
      reddesc.SetValue(length,0); reddesc.SetValue(subscript,1); reddesc.SetValue(bitsize,2);
      ds.Replace( reddesc.GetAsDataElement() );

      // GREEN
      memset(rawlut,0,lutlen*2);
      lut.GetLUT(LookupTable::GREEN, (unsigned char*)rawlut, l);
      DataElement greende( Tag(0x0028, 0x1202) );
      greende.SetVR( VR::OW );
      greende.SetByteValue( (char*)rawlut, l);
      ds.Replace( greende );
      // descriptor:
      Attribute<0x0028, 0x1102, VR::US, VM::VM3> greendesc;
      lut.GetLUTDescriptor(LookupTable::GREEN, length, subscript, bitsize);
      greendesc.SetValue(length,0); greendesc.SetValue(subscript,1); greendesc.SetValue(bitsize,2);
      ds.Replace( greendesc.GetAsDataElement() );

      // BLUE
      memset(rawlut,0,lutlen*2);
      lut.GetLUT(LookupTable::BLUE, (unsigned char*)rawlut, l);
      DataElement bluede( Tag(0x0028, 0x1203) );
      bluede.SetVR( VR::OW );
      bluede.SetByteValue( (char*)rawlut, l);
      ds.Replace( bluede );
      // descriptor:
      Attribute<0x0028, 0x1103, VR::US, VM::VM3> bluedesc;
      lut.GetLUTDescriptor(LookupTable::BLUE, length, subscript, bitsize);
      bluedesc.SetValue(length,0); bluedesc.SetValue(subscript,1); bluedesc.SetValue(bitsize,2);
      ds.Replace( bluedesc.GetAsDataElement() );
      }

    ds.Remove( Tag(0x0028, 0x1221) );
    ds.Remove( Tag(0x0028, 0x1222) );
    ds.Remove( Tag(0x0028, 0x1223) );

    }

  // FIXME shouldn't this be done by the ImageApplyLookupTable filter ?
  if( pi == PhotometricInterpretation::RGB )
    {
    // usual tags:
    ds.Remove( Tag(0x0028, 0x1101) );
    ds.Remove( Tag(0x0028, 0x1102) );
    ds.Remove( Tag(0x0028, 0x1103) );

    ds.Remove( Tag(0x0028, 0x1201) );
    ds.Remove( Tag(0x0028, 0x1202) );
    ds.Remove( Tag(0x0028, 0x1203) );

    // Dont' forget the segmented one:
    ds.Remove( Tag(0x0028, 0x1221) );
    ds.Remove( Tag(0x0028, 0x1222) );
    ds.Remove( Tag(0x0028, 0x1223) );

    // PaletteColorLookupTableUID ??
    ds.Remove( Tag(0x0028, 0x1199) );
    }
//  Attribute<0x0028, 0x0004> photometricinterpretation;
//  photometricinterpretation.SetValue( pi );
//  ds.Replace( photometricinterpretation.GetAsDataElement() );
Attribute<0x0028,0x0004> piat;
//const DataElement &pide = ds.GetDataElement( piat.GetTag() );
//const char *str1 = pide.GetByteValue()->GetPointer();
{
    const char *pistr = PhotometricInterpretation::GetPIString(pi);
    DataElement de( Tag(0x0028, 0x0004 ) );
    VL::Type strlenPistr = (VL::Type)strlen(pistr);
    de.SetByteValue( pistr, strlenPistr );
    de.SetVR( piat.GetVR() );
    ds.Replace( de );
}


  // Spacing:
  std::vector<double> sp;
  sp.resize(3); // important !
  sp[0] = pixeldata.GetSpacing(0);
  sp[1] = pixeldata.GetSpacing(1);
  sp[2] = pixeldata.GetSpacing(2); // might be a dummy value...
  ImageHelper::SetSpacingValue(ds, sp);

  // Direction Cosines:
  const double *dircos = pixeldata.GetDirectionCosines();
  if( dircos )
    {
    std::vector<double> iop;
    iop.resize(6);
    iop[0] = dircos[0];
    iop[1] = dircos[1];
    iop[2] = dircos[2];
    iop[3] = dircos[3];
    iop[4] = dircos[4];
    iop[5] = dircos[5];
    ImageHelper::SetDirectionCosinesValue(ds, iop);
    }

  // Origin:
  const double *origin = pixeldata.GetOrigin();
  if( origin )
    {
    ImageHelper::SetOriginValue(ds, pixeldata);
    }

  assert( Stream );
  if( !Writer::Write() )
    {
    return false;
    }
  return true;
}

} // end namespace gdcm
