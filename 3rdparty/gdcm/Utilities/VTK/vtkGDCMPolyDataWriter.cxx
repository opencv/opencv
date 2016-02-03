/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMPolyDataWriter.h"

#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkInformationVector.h"
#include "vtkPolyData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkFloatArray.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkErrorCode.h"
#include "vtkMedicalImageProperties.h"
#include "vtkRTStructSetProperties.h"
#include "gdcmSystem.h"

#include "gdcmWriter.h"
#include "gdcmUIDs.h"
#include "gdcmUIDGenerator.h"
#include "gdcmSmartPointer.h"
#include "gdcmAttribute.h"
#include "gdcmSmartPointer.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmAnonymizer.h"
#include "gdcmIPPSorter.h"
#include "gdcmAttribute.h"
#include "gdcmDirectoryHelper.h"

vtkCxxRevisionMacro(vtkGDCMPolyDataWriter, "$Revision: 1.74 $")
vtkStandardNewMacro(vtkGDCMPolyDataWriter)
vtkCxxSetObjectMacro(vtkGDCMPolyDataWriter,MedicalImageProperties,vtkMedicalImageProperties)
vtkCxxSetObjectMacro(vtkGDCMPolyDataWriter,RTStructSetProperties,vtkRTStructSetProperties)

//----------------------------------------------------------------------------
vtkGDCMPolyDataWriter::vtkGDCMPolyDataWriter()
{
  this->SetNumberOfInputPorts(1);
  this->MedicalImageProperties = vtkMedicalImageProperties::New();
  this->RTStructSetProperties = vtkRTStructSetProperties::New();
}

//----------------------------------------------------------------------------
vtkGDCMPolyDataWriter::~vtkGDCMPolyDataWriter()
{
  this->MedicalImageProperties->Delete();
  this->RTStructSetProperties->Delete();
}

static void SetStringValueFromTag(const char *s, const gdcm::Tag& t, gdcm::Anonymizer & ano)
{
  if( s && *s )
    {
#if 0
    gdcm::DataElement de( t );
    de.SetByteValue( s, strlen( s ) );
    const gdcm::Global& g = gdcm::Global::GetInstance();
    const gdcm::Dicts &dicts = g.GetDicts();
    // FIXME: we know the tag at compile time we could save some time
    // Using the static dict instead of the run-time one:
    const gdcm::DictEntry &dictentry = dicts.GetDictEntry( t );
    de.SetVR( dictentry.GetVR() );
    ds.Insert( de );
#else
    ano.Replace(t, s);
#endif
    }
}

using namespace gdcm;

//----------------------------------------------------------------------------
void vtkGDCMPolyDataWriter::WriteData()
{
  if ( this->FileName == NULL)
    {
    vtkErrorMacro(<< "Please specify FileName to write");
    this->SetErrorCode(vtkErrorCode::NoFileNameError);
    return;
    }
  Writer writer;
  writer.SetFileName( this->FileName );
  File &file = writer.GetFile();

  this->WriteRTSTRUCTInfo(file);

  int numInputs = this->GetNumberOfInputPorts();
  for(int input = 0; input < numInputs; ++input )
    {
    this->WriteRTSTRUCTData(file, input);
    }

  if( !writer.Write() )
    {
    vtkErrorMacro(<< "Could not write");
    this->SetErrorCode(vtkErrorCode::FileFormatError);
    return;
    }

}

//----------------------------------------------------------------------------
void vtkGDCMPolyDataWriter::WriteRTSTRUCTInfo(gdcm::File &file)
{
  DataSet& ds = file.GetDataSet();
{
  const Tag sisq(0x3006,0x0039);
  DataElement de( sisq );
  de.SetVR( VR::SQ );
  SmartPointer<SequenceOfItems> sqi1 = 0;
  sqi1 = new SequenceOfItems;
  de.SetValue( *sqi1 );
  de.SetVLToUndefined();
  ds.Insert( de );
}

  UIDGenerator uid;

    {
    const char *sop = uid.Generate();
    DataElement de( Tag(0x0008,0x0018) );
    VL::Type strlenSOP = (VL::Type) strlen(sop);
    de.SetByteValue( sop, strlenSOP );
    de.SetVR( Attribute<0x0008, 0x0018>::GetVR() );
    ds.ReplaceEmpty( de );
    }
    {
    //this is incorrect.
    //the study MUST be the same as the image from which this object is derived.
//    const char *study = uid.Generate();
//    DataElement de( Tag(0x0020,0x000d) );
//    VL::Type strlenStudy= (VL::Type)strlen(study);
//    de.SetByteValue( study, strlenStudy );
//    de.SetVR( Attribute<0x0020, 0x000d>::GetVR() );
//    ds.ReplaceEmpty( de );
    }
    {
    const char *series = uid.Generate();
    DataElement de( Tag(0x0020,0x000e) );
    VL::Type strlenSeries= (VL::Type)strlen(series);
    de.SetByteValue( series, strlenSeries );
    de.SetVR( Attribute<0x0020, 0x000e>::GetVR() );
    ds.ReplaceEmpty( de );
    }

  FileMetaInformation &fmi = file.GetHeader();
  TransferSyntax ts = TransferSyntax::ImplicitVRLittleEndian;
    {
    const char *tsuid = TransferSyntax::GetTSString( ts );
    DataElement de( Tag(0x0002,0x0010) );
    VL::Type strlenTSUID = (VL::Type)strlen(tsuid);
    de.SetByteValue( tsuid, strlenTSUID );
    de.SetVR( Attribute<0x0002, 0x0010>::GetVR() );
    fmi.Replace( de );
    fmi.SetDataSetTransferSyntax(ts);
    }
  MediaStorage ms = MediaStorage::RTStructureSetStorage ;
  const char* msstr = MediaStorage::GetMSString(ms);
    {
    DataElement de( Tag(0x0008, 0x0016 ) );
    VL::Type strlenMsstr = (VL::Type)strlen(msstr);
    de.SetByteValue( msstr, strlenMsstr);
    de.SetVR( Attribute<0x0008, 0x0016>::GetVR() );
    ds.Insert( de );
    }

  int year, month, day;
  gdcm::Anonymizer ano;
    ano.SetFile( file );

    SetStringValueFromTag(this->RTStructSetProperties->GetStructureSetLabel(), gdcm::Tag(0x3006,0x0002), ano);
    SetStringValueFromTag(this->RTStructSetProperties->GetStructureSetName(), gdcm::Tag(0x3006,0x0004), ano);
    SetStringValueFromTag(this->RTStructSetProperties->GetStructureSetDate(), gdcm::Tag(0x3006,0x0008), ano);
    SetStringValueFromTag(this->RTStructSetProperties->GetStructureSetTime(), gdcm::Tag(0x3006,0x0009), ano);
    SetStringValueFromTag(this->RTStructSetProperties->GetSOPInstanceUID(), gdcm::Tag(0x0008,0x0018), ano);
    SetStringValueFromTag(this->RTStructSetProperties->GetStudyInstanceUID(), gdcm::Tag(0x0020,0x000d), ano);
    SetStringValueFromTag(this->RTStructSetProperties->GetSeriesInstanceUID(), gdcm::Tag(0x0020,0x000e), ano);

{
  SmartPointer<SequenceOfItems> sqi;
  sqi = new SequenceOfItems;
  vtkIdType n = this->RTStructSetProperties->GetNumberOfReferencedFrameOfReferences();
  for( vtkIdType id = 0; id < n; ++id )
    {
    const char *sopclass = this->RTStructSetProperties->GetReferencedFrameOfReferenceClassUID(id);
    const char *instanceuid = this->RTStructSetProperties->GetReferencedFrameOfReferenceInstanceUID(id);
    Item item;
    item.SetVLToUndefined();
    DataSet &subds = item.GetNestedDataSet();
{
    Attribute<0x0008,0x1150> at;
    at.SetValue( sopclass );
    subds.Insert( at.GetAsDataElement() );
}
{
    Attribute<0x0008,0x1155> at;
    at.SetValue( instanceuid );
    subds.Insert( at.GetAsDataElement() );
}

    sqi->AddItem( item );
    }

  DataElement de1( Tag(0x3006,0x0010) );
  de1.SetVR( VR::SQ );
  SmartPointer<SequenceOfItems> sqi1 = new SequenceOfItems;
  de1.SetValue( *sqi1 );
  de1.SetVLToUndefined();
  ds.Insert( de1 );

  Item item1;
  item1.SetVLToUndefined();
  DataSet &ds2 = item1.GetNestedDataSet();

  gdcm::Attribute<0x0020,0x052> frameofreferenceuid;
  if( this->RTStructSetProperties->GetReferenceFrameOfReferenceUID() )
    frameofreferenceuid.SetValue(
      this->RTStructSetProperties->GetReferenceFrameOfReferenceUID() );
  ds2.Insert( frameofreferenceuid.GetAsDataElement() );

  DataElement de2( Tag(0x3006,0x0012) );
  de2.SetVR( VR::SQ );
  SmartPointer<SequenceOfItems> sqi2 = new SequenceOfItems;
  de2.SetValue( *sqi2 );
  de2.SetVLToUndefined();
  ds2.Insert( de2 );

  Item item2;
  item2.SetVLToUndefined();
  DataSet &ds3 = item2.GetNestedDataSet();

  Attribute<0x0008,0x1150> refsopclassuid;
  const char *rtuid = gdcm::UIDs::GetUIDString(
    gdcm::UIDs::RTStructureSetStorage);
  refsopclassuid.SetValue ( rtuid );
  ds3.Insert( refsopclassuid.GetAsDataElement() );
  Attribute<0x0008,0x1155> refsopinstuid;
  if( this->RTStructSetProperties->GetStudyInstanceUID() )
    refsopinstuid.SetValue ( this->RTStructSetProperties->GetStudyInstanceUID() );
  ds3.Insert( refsopinstuid.GetAsDataElement() );

  DataElement de3( Tag(0x3006,0x0014) );
  de3.SetVR( VR::SQ );
  SmartPointer<SequenceOfItems> sqi3 = new SequenceOfItems;
  de3.SetValue( *sqi3 );
  de3.SetVLToUndefined();
  ds3.Insert( de3 );

  Item item3;
  item3.SetVLToUndefined();
  DataSet &ds4 = item3.GetNestedDataSet();

  gdcm::Attribute<0x0020,0x000e> seriesinstanceuid;
  if ( this->RTStructSetProperties->GetReferenceSeriesInstanceUID() )
    seriesinstanceuid.SetValue(
      this->RTStructSetProperties->GetReferenceSeriesInstanceUID() );
  ds4.Insert( seriesinstanceuid.GetAsDataElement() );

  DataElement de4( Tag(0x3006,0x0016) );
  de4.SetVR( VR::SQ );
  //SmartPointer<SequenceOfItems> sqi4 = new SequenceOfItems;
  de4.SetValue( *sqi );
  de4.SetVLToUndefined();
  ds4.Insert( de4 );

  //Item item4;
  //item4.SetVLToUndefined();
  //DataSet &ds5 = item4.GetNestedDataSet();

  //sqi4->AddItem( item4 );

  sqi3->AddItem( item3 );

  sqi2->AddItem( item2 );

  sqi1->AddItem( item1 );
}
{
  SmartPointer<SequenceOfItems> sqi;
  sqi = new SequenceOfItems;
  SmartPointer<SequenceOfItems> sqiobs;
  sqiobs = new SequenceOfItems;
  vtkIdType n = this->RTStructSetProperties->GetNumberOfStructureSetROIs();
  for( vtkIdType id = 0; id < n; ++id )
    {
    int roinumber              = this->RTStructSetProperties->GetStructureSetROINumber(id);
    const char *refframerefuid = this->RTStructSetProperties->GetStructureSetROIRefFrameRefUID(id);
    const char *roiname        = this->RTStructSetProperties->GetStructureSetROIName(id);
    const char *roigenalgo     = this->RTStructSetProperties->GetStructureSetROIGenerationAlgorithm(id);
    const char *roidesc        = this->RTStructSetProperties->GetStructureSetROIDescription(id);
    Item item;
    item.SetVLToUndefined();
    DataSet &subds = item.GetNestedDataSet();

    gdcm::Attribute<0x3006,0x0022> atroinumber;
    atroinumber.SetValue( roinumber );
    subds.Insert( atroinumber.GetAsDataElement() );

    gdcm::Attribute<0x3006,0x0024> atrefframeuid;
    atrefframeuid.SetValue( refframerefuid );
    subds.Insert( atrefframeuid.GetAsDataElement() );

    gdcm::Attribute<0x3006,0x0026> atroiname;
    atroiname.SetValue( roiname );
    subds.Insert( atroiname.GetAsDataElement() );

    if( roidesc && *roidesc )
      {
      gdcm::Attribute<0x3006,0x0028> atroidesc;
      atroidesc.SetValue( roidesc );
      subds.Insert( atroidesc.GetAsDataElement() );
      }

    gdcm::Attribute<0x3006,0x0036> atroigenalg;
    atroigenalg.SetValue( roigenalgo );
    subds.Insert( atroigenalg.GetAsDataElement() );

    // do the obs stuff
    Item itemobs;
    itemobs.SetVLToUndefined();
    DataSet &subdsobs = itemobs.GetNestedDataSet();

    int observationnumber = this->RTStructSetProperties->GetStructureSetObservationNumber(id);
    gdcm::Attribute<0x3006,0x0082> atobservationnumber;
    atobservationnumber.SetValue( observationnumber );
    subdsobs.Insert( atobservationnumber.GetAsDataElement() );

    gdcm::Attribute<0x3006,0x0084> atreferencedroinumber;
    atreferencedroinumber.SetValue( roinumber );
    subdsobs.Insert( atreferencedroinumber.GetAsDataElement() );

    const char *roiobservationlabel = this->RTStructSetProperties->GetStructureSetROIObservationLabel(id);
    if( roiobservationlabel && *roiobservationlabel )
      {
      gdcm::Attribute<0x3006,0x0085> atroiobservationlabel;
      atroiobservationlabel.SetValue( roiobservationlabel );
      subdsobs.Insert( atroiobservationlabel.GetAsDataElement() );
      }

    const char *rtroiinterpretedtype = this->RTStructSetProperties->GetStructureSetRTROIInterpretedType(id);
    gdcm::Attribute<0x3006,0x00a4> atrtroiinterpretedtype;
    atrtroiinterpretedtype.SetValue( rtroiinterpretedtype );
    subdsobs.Insert( atrtroiinterpretedtype.GetAsDataElement() );

    gdcm::Attribute<0x3006,0x00a6> atroiinterpreter;
    //atroiinterpreter.SetValue( rtroiinterpretedtype );
    subdsobs.Insert( atroiinterpreter.GetAsDataElement() );

    sqiobs->AddItem( itemobs );
    sqi->AddItem( item );
    }
  DataElement de1( Tag(0x3006,0x0020) );
  de1.SetVR( VR::SQ );
  de1.SetValue( *sqi );
  de1.SetVLToUndefined();
  ds.Insert( de1 );

  DataElement de2( Tag(0x3006,0x0080) );
  de2.SetVR( VR::SQ );
  de2.SetValue( *sqiobs );
  de2.SetVLToUndefined();
  ds.Insert( de2 );
}

    // For ex: DICOM (0010,0010) = DOE,JOHN
    SetStringValueFromTag(this->MedicalImageProperties->GetPatientName(), gdcm::Tag(0x0010,0x0010), ano);
    // For ex: DICOM (0010,0020) = 1933197
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientID(), gdcm::Tag(0x0010,0x0020), ano);
    // For ex: DICOM (0010,1010) = 031Y
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientAge(), gdcm::Tag(0x0010,0x1010), ano);
    // For ex: DICOM (0010,0040) = M
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientSex(), gdcm::Tag(0x0010,0x0040), ano);
    // For ex: DICOM (0010,0030) = 19680427
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientBirthDate(), gdcm::Tag(0x0010,0x0030), ano);
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
    // For ex: DICOM (0008,0020) = 20030617
    if( vtkMedicalImageProperties::GetDateAsFields( this->MedicalImageProperties->GetStudyDate(), year, month, day ) )
      SetStringValueFromTag( this->MedicalImageProperties->GetStudyDate(), gdcm::Tag(0x0008,0x0020), ano);
#endif
    // For ex: DICOM (0008,0022) = 20030617
    SetStringValueFromTag( this->MedicalImageProperties->GetAcquisitionDate(), gdcm::Tag(0x0008,0x0022), ano);
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
    // For ex: DICOM (0008,0030) = 162552.0705 or 230012, or 0012
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 4 )
    int hour, minute, second;
    if( vtkMedicalImageProperties::GetTimeAsFields( this->MedicalImageProperties->GetStudyTime(), hour, minute, second ) )
#endif
      SetStringValueFromTag( this->MedicalImageProperties->GetStudyTime(), gdcm::Tag(0x0008,0x0030), ano);
#endif
    // For ex: DICOM (0008,0032) = 162552.0705 or 230012, or 0012
    SetStringValueFromTag( this->MedicalImageProperties->GetAcquisitionTime(), gdcm::Tag(0x0008,0x0032), ano);
    // For ex: DICOM (0008,0023) = 20030617
    SetStringValueFromTag( this->MedicalImageProperties->GetImageDate(), gdcm::Tag(0x0008,0x0023), ano);
    // For ex: DICOM (0008,0033) = 162552.0705 or 230012, or 0012
    SetStringValueFromTag( this->MedicalImageProperties->GetImageTime(), gdcm::Tag(0x0008,0x0033), ano);
    // For ex: DICOM (0020,0013) = 1
    SetStringValueFromTag( this->MedicalImageProperties->GetImageNumber(), gdcm::Tag(0x0020,0x0013), ano);
    // For ex: DICOM (0020,0011) = 902
    SetStringValueFromTag( this->MedicalImageProperties->GetSeriesNumber(), gdcm::Tag(0x0020,0x0011), ano);
    // For ex: DICOM (0008,103e) = SCOUT
    SetStringValueFromTag( this->MedicalImageProperties->GetSeriesDescription(), gdcm::Tag(0x0008,0x103e), ano);
    // For ex: DICOM (0020,0010) = 37481
    SetStringValueFromTag( this->MedicalImageProperties->GetStudyID(), gdcm::Tag(0x0020,0x0010), ano);
    // For ex: DICOM (0008,1030) = BRAIN/C-SP/FACIAL
    SetStringValueFromTag( this->MedicalImageProperties->GetStudyDescription(), gdcm::Tag(0x0008,0x1030), ano);
    // For ex: DICOM (0008,0060)= CT
    SetStringValueFromTag( this->MedicalImageProperties->GetModality(), gdcm::Tag(0x0008,0x0060), ano);
    // For ex: DICOM (0008,0070) = Siemens
    SetStringValueFromTag( this->MedicalImageProperties->GetManufacturer(), gdcm::Tag(0x0008,0x0070), ano);
    // For ex: DICOM (0008,1090) = LightSpeed QX/i
    SetStringValueFromTag( this->MedicalImageProperties->GetManufacturerModelName(), gdcm::Tag(0x0008,0x1090), ano);
    // For ex: DICOM (0008,1010) = LSPD_OC8
    SetStringValueFromTag( this->MedicalImageProperties->GetStationName(), gdcm::Tag(0x0008,0x1010), ano);
    // For ex: DICOM (0008,0080) = FooCity Medical Center
    SetStringValueFromTag( this->MedicalImageProperties->GetInstitutionName(), gdcm::Tag(0x0008,0x0080), ano);
    // For ex: DICOM (0018,1210) = Bone
    SetStringValueFromTag( this->MedicalImageProperties->GetConvolutionKernel(), gdcm::Tag(0x0018,0x1210), ano);
    // For ex: DICOM (0018,0050) = 0.273438
    SetStringValueFromTag( this->MedicalImageProperties->GetSliceThickness(), gdcm::Tag(0x0018,0x0050), ano);
    // For ex: DICOM (0018,0060) = 120
    SetStringValueFromTag( this->MedicalImageProperties->GetKVP(), gdcm::Tag(0x0018,0x0060), ano);
    // For ex: DICOM (0018,1120) = 15
    SetStringValueFromTag( this->MedicalImageProperties->GetGantryTilt(), gdcm::Tag(0x0018,0x1120), ano);
    // For ex: DICOM (0018,0081) = 105
    SetStringValueFromTag( this->MedicalImageProperties->GetEchoTime(), gdcm::Tag(0x0018,0x0081), ano);
    // For ex: DICOM (0018,0091) = 35
    SetStringValueFromTag( this->MedicalImageProperties->GetEchoTrainLength(), gdcm::Tag(0x0018,0x0091), ano);
    // For ex: DICOM (0018,0080) = 2040
    SetStringValueFromTag( this->MedicalImageProperties->GetRepetitionTime(), gdcm::Tag(0x0018,0x0080), ano);
    // For ex: DICOM (0018,1150) = 5
    SetStringValueFromTag( this->MedicalImageProperties->GetExposureTime(), gdcm::Tag(0x0018,0x1150), ano);
    // For ex: DICOM (0018,1151) = 400
    SetStringValueFromTag( this->MedicalImageProperties->GetXRayTubeCurrent(), gdcm::Tag(0x0018,0x1151), ano);
    // For ex: DICOM (0018,1152) = 114
    SetStringValueFromTag( this->MedicalImageProperties->GetExposure(), gdcm::Tag(0x0018,0x1152), ano);

}

//----------------------------------------------------------------------------
void vtkGDCMPolyDataWriter::WriteRTSTRUCTData(gdcm::File &file, int pdidx )
{
    vtkPolyData *input = this->GetInput(pdidx);
    assert( input );
    vtkPoints *pts;
    vtkCellArray *polys;

    polys = input->GetPolys();
    vtkCellArray* lines = input->GetLines();
    pts = input->GetPoints();
    vtkDataArray *scalars = input->GetCellData()->GetScalars();
    vtkDoubleArray *darray = vtkDoubleArray::SafeDownCast( scalars );
    vtkFloatArray *farray = vtkFloatArray::SafeDownCast( scalars );

    if (pts == NULL || polys == NULL || lines == NULL)
      {
      vtkWarningMacro(<<"No data to write!");//should be a warning, not an error, because
      //it's entirely possible to have a blank ROI
      //return;//ok, you have to put the observation here, even if it's blank
      //if it's blank, the color and so forth are still defined.  Otherwise,
      //the observation will be incomplete.
      }

/*
(3006,0039) ?? (SQ)                                               # u/l,1 ROI Contour Sequence
  (fffe,e000) na (Item with undefined length)
    (3006,002a) ?? (IS) [220\160\120 ]                            # 12,3 ROI Display Color
    (3006,0040) ?? (SQ)                                           # u/l,1 Contour Sequence
      (fffe,e000) na (Item with undefined length)
        (3006,0016) ?? (SQ)                                       # u/l,1 Contour Image Sequence
          (fffe,e000) na (Item with undefined length)
            (0008,1150) ?? (UI) [1.2.840.10008.5.1.4.1.1.2]       # 26,1 Referenced SOP Class UID
            (0008,1155) ?? (UI) [1.3.6.1.4.1.22213.1.1396.148]    # 28,1 Referenced SOP Instance UID
          (fffe,e00d)
        (fffe,e0dd)
        (3006,0042) ?? (CS) [CLOSED_PLANAR ]                      # 14,1 Contour Geometric Type
        (3006,0046) ?? (IS) [139 ]                                # 4,1 Number of Contour Points
        (3006,0050) ?? (DS) [-209.81171875\-392.41171875\...]     # 5004,3-3n Contour Data
      (fffe,e00d)

*/
  SmartPointer<SequenceOfItems> sqi;
  sqi = new SequenceOfItems;

  vtkIdType npts = 0;
  vtkIdType *indx = 0;
  double v[3];
  unsigned int cellnum = 0;

  //choose to use either polys or lines
  //the result of vtk marching cubes->stripper->appendpolydata is a set of lines,
  //not polys, so favor that one for now.
  //choose by the number of polys/lines available.
  vtkCellArray* theCells = lines;
  if (!lines || lines->GetNumberOfCells() == 0){
    theCells = polys;
  }

  std::vector<double> cellpoints;
  for (theCells->InitTraversal(); theCells->GetNextCell(npts,indx); cellnum++ ){
    cellpoints.resize(0);
    for(vtkIdType index = 0; index < npts; ++index){
      pts->GetPoint(indx[index],v);
      //precision problems are _definitely_ here by this point
      //this a crude hack to the get the 9999's under control,
      //or pollution by switching from doubles to floats and back again
      //cellpoints.push_back( (double)((int)(v[0]*10000.0))/10000.0 );
      //cellpoints.push_back( (double)((int)(v[1]*10000.0))/10000.0 );
      //cellpoints.push_back( (double)((int)(v[2]*10000.0))/10000.0 );
      cellpoints.push_back( v[0] );
      cellpoints.push_back( v[1] );
      cellpoints.push_back( v[2 ]);
    }
    Item item0;
    item0.SetVLToUndefined();
    DataSet &subds0 = item0.GetNestedDataSet();
    Attribute<0x3006,0x0050> at;
    at.SetValues( &cellpoints[0], (unsigned int)cellpoints.size(), false );
    subds0.Insert( at.GetAsDataElement() );

    Attribute<0x3006,0x0046> numcontpoints;
    numcontpoints.SetValue( (int)npts );
    subds0.Insert( numcontpoints.GetAsDataElement() );
    Attribute<0x3006,0x0042> contgeotype;
    contgeotype.SetValue( "CLOSED_PLANAR " );
    subds0.Insert( contgeotype.GetAsDataElement() );

    SmartPointer<SequenceOfItems> thesqi = new SequenceOfItems;
    {
      Item item;
      item.SetVLToUndefined();
      DataSet &subds = item.GetNestedDataSet();

      gdcm::Attribute<0x0008,0x1150> classat; 
      classat.SetValue ( this->RTStructSetProperties->
        GetContourReferencedFrameOfReferenceClassUID( pdidx, cellnum ));
      subds.Insert( classat.GetAsDataElement() );
      gdcm::Attribute<0x0008,0x1155> instat;
      instat.SetValue ( this->RTStructSetProperties->
        GetContourReferencedFrameOfReferenceInstanceUID( pdidx, cellnum ));
      subds.Insert( instat.GetAsDataElement() );
      thesqi->AddItem( item );
    }

    DataElement contimsq = DataElement( Tag(0x3006,0x0016) );
    contimsq.SetVR( VR::SQ );
    contimsq.SetValue( *thesqi );
    contimsq.SetVLToUndefined();
    subds0.Insert( contimsq );


    sqi->AddItem( item0 );
  }
  DataSet& ds = file.GetDataSet();
{
  const Tag sisq(0x3006,0x0039);
  SmartPointer<SequenceOfItems> sqi1 = 0;
  sqi1 = ds.GetDataElement( sisq ).GetValueAsSQ();
  assert( sqi1 );

  Item item;
  item.SetVLToUndefined();
  DataSet &subds = item.GetNestedDataSet();

  gdcm::Attribute<0x3006,0x0084> referencedroinumber;
  //referencedroinumber.SetValue ( pdidx );
  referencedroinumber.SetValue( this->RTStructSetProperties->GetStructureSetROINumber(pdidx) );
  subds.Insert( referencedroinumber.GetAsDataElement() );

  //(3006,002a) IS [220\160\120]  #  12, 3 ROIDisplayColor
  gdcm::Attribute<0x3006,0x002a> roidispcolor;
  int32_t intcolor[3] = {0,0,0};
  //assert( darray || farray );
  if( darray )
    {
    double tuple[3];
    darray->GetTupleValue( 0, tuple );
    intcolor[0] = (int32_t)(tuple[0] * 255.);
    intcolor[1] = (int32_t)(tuple[1] * 255.);
    intcolor[2] = (int32_t)(tuple[2] * 255.);
    }
  else if( farray )
    {
    float ftuple[3];
    farray->GetTupleValue( 0, ftuple );
    intcolor[0] = (int32_t)(ftuple[0] * 255.);
    intcolor[1] = (int32_t)(ftuple[1] * 255.);
    intcolor[2] = (int32_t)(ftuple[2] * 255.);
    }
  else
    {
    vtkDebugMacro( "No color" );
    }
  roidispcolor.SetValues( intcolor, 3 );
  subds.Insert( roidispcolor.GetAsDataElement() );

  if(  sqi->GetNumberOfItems() )
    {
    const Tag sisq2(0x3006,0x0040);
    DataElement de2( sisq2 );
    de2.SetVR( VR::SQ );
    de2.SetValue( *sqi );
    de2.SetVLToUndefined();
    subds.Insert( de2 );
    }

    sqi1->AddItem( item );
}

}

//----------------------------------------------------------------------------
void vtkGDCMPolyDataWriter::SetNumberOfInputPorts(int n)
{
  Superclass::SetNumberOfInputPorts(n);
}

//----------------------------------------------------------------------------
void vtkGDCMPolyDataWriter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}


//this function will initialize the contained rtstructset with
//the inputs of the writer and the various extra information
//necessary for writing a complete rtstructset.
//NOTE: inputs must be set BEFORE calling this function!
//NOTE: the number of outputs for the appendpolydata MUST MATCH the organ vectors!
void vtkGDCMPolyDataWriter::InitializeRTStructSet(vtkStdString inDirectory,
                                                  vtkStdString inStructLabel,
                                                  vtkStdString inStructName,
                                                  vtkStringArray* inROINames,
                                                  vtkStringArray* inROIAlgorithmName,
                                                  vtkStringArray* inROIType)
{
  gdcm::Directory::FilenamesType theCTSeries =
    gdcm::DirectoryHelper::GetCTImageSeriesUIDs(inDirectory);
  if (theCTSeries.size() > 1)
    {
    gdcmWarningMacro("More than one CT series detected, only reading series UID: "
      << theCTSeries[0]);
    }
  if (theCTSeries.empty())
    {
    gdcmWarningMacro("No CT Series found, trying MR.");
    theCTSeries = gdcm::DirectoryHelper::GetMRImageSeriesUIDs(inDirectory);
    if (theCTSeries.size() > 1)
      {
      gdcmWarningMacro("More than one MR series detected, only reading series UID: "
        << theCTSeries[0]);
      }
    if (theCTSeries.empty())
      {
      gdcmWarningMacro("No CT or MR series found, throwing.");
      return;// false;
      }
    }
  //load the images in the CT series
  std::vector<DataSet> theCTDataSets =
    gdcm::DirectoryHelper::LoadImageFromFiles(inDirectory, theCTSeries[0]);
  if (theCTDataSets.empty())
    {
    gdcmWarningMacro("No CT or MR Images loaded, throwing.");
    return;// false;
    }

  //now, armed with this set of images, we can begin to properly construct the RTStructureSet
  vtkRTStructSetProperties* theRTStruct = RTStructSetProperties;//initially, this function was a static construction
  //but that doesn't jive with swig wrapping easily
  theRTStruct->SetStructureSetLabel(inStructLabel.c_str());
  theRTStruct->SetStructureSetName(inStructName.c_str());
  //theRTStruct->SetSOPInstanceUID(<#const char *_arg#>);//should be autogenerated by the object itself
  {
  const ByteValue* theValue = theCTDataSets[0].FindNextDataElement(Tag(0x0020,0x000d)).GetByteValue();
  std::string theStringValue(theValue->GetPointer(), theValue->GetLength());
  theRTStruct->SetStudyInstanceUID(theStringValue.c_str());
  }
  {
  const ByteValue* theValue = theCTDataSets[0].FindNextDataElement(Tag(0x0020,0x000e)).GetByteValue();
  std::string theStringValue(theValue->GetPointer(), theValue->GetLength());
  theRTStruct->SetReferenceSeriesInstanceUID(theStringValue.c_str());
  }
  {
  const ByteValue* theValue = theCTDataSets[0].FindNextDataElement(Tag(0x0020,0x0052)).GetByteValue();
  std::string theStringValue(theValue->GetPointer(), theValue->GetLength());
  theRTStruct->SetReferenceFrameOfReferenceUID(theStringValue.c_str());
  }
  //the series UID should be set automatically, and happen during creation
  //set the date and time to be now

  char date[22];
  const size_t datelen = 8;
  int res = System::GetCurrentDateTime(date);
  assert( res );
  (void)res;//warning removal//causes java wrapping to fail
  //the date is the first 8 chars
  std::string dateString;
  dateString.insert(dateString.begin(), &(date[0]), &(date[datelen]));
  theRTStruct->SetStructureSetDate(dateString.c_str());
  std::string timeString;
  const size_t timelen = 6; //for now, only need hhmmss
  timeString.insert(timeString.begin(), &(date[datelen]), &(date[datelen+timelen]));
  theRTStruct->SetStructureSetTime(timeString.c_str());

  //for each image, we need to fill in the sop class and instance UIDs for the frame of reference
  std::string theSOPClassID = DirectoryHelper::GetSOPClassUID(theCTDataSets).c_str();
  for (unsigned long i = 0; i < theCTDataSets.size(); i++)
    {
    theRTStruct->AddReferencedFrameOfReference(theSOPClassID.c_str(),
      DirectoryHelper::RetrieveSOPInstanceUIDFromIndex((int)i,theCTDataSets).c_str());
    }

  //now, we have go to through each vtkPolyData, assign the ROI names per polydata, and then also assign the
  //reference SOP instance UIDs on a per-plane basis.
  int theNumPorts = GetNumberOfInputPorts();
  for (int j = 0; j < theNumPorts; j++)
    {
    int contour = j;
    theRTStruct->AddStructureSetROI(contour,
        theRTStruct->GetReferenceFrameOfReferenceUID(),
      inROINames->GetValue(j).c_str(),
      inROIAlgorithmName->GetValue(j).c_str());
    
    theRTStruct->AddStructureSetROIObservation(contour,
      contour, inROIType->GetValue(j).c_str(), "");
     //for each organ, gotta go through and add in the right planes in the
     //order that the tuples appear, as well as the colors
     //right now, each cell in the vtkpolydata is a contour in an xy plane
     //that's what MUST be passed in
    vtkPolyData* theData = dynamic_cast<vtkPolyData*>(GetInput(j));
    if (theData == NULL)
      {
      gdcmWarningMacro("theData for input " << j << " is NULL, continuing");
      continue;
      }
    unsigned int cellnum = 0;
    vtkPoints *pts;
    vtkCellArray *polys;
    vtkIdType npts = 0;
    vtkIdType *indx = 0;
    pts = theData->GetPoints();
    polys = theData->GetPolys();
    vtkCellArray* lines = theData->GetLines();

    //choose to use either polys or lines
    //the result of vtk marching cubes->stripper->appendpolydata is a set of lines,
    //not polys, so favor that one for now.
    //choose by the number of polys/lines available.
    vtkCellArray* theCells = lines;
    if (!lines || lines->GetNumberOfCells() == 0)
      {
      theCells = polys;
      }
    double v[3];
    vtkIdType theNumCells = theCells->GetNumberOfCells();
    gdcmDebugMacro("The number of cells:" << theNumCells);
    if (theNumCells == 0) continue;// no observation of blank organs

    for (theCells->InitTraversal(); theCells->GetNextCell(npts,indx); cellnum++ )
      {
      if (npts < 1)
        {
        gdcmWarningMacro("theCells for input " << j << " is less than 1, continuing");
        continue;
        }
      pts->GetPoint(indx[0],v);
      double theZ = v[2];
      std::string theSOPInstance =
        DirectoryHelper::RetrieveSOPInstanceUIDFromZPosition(theZ, theCTDataSets);
      //j is correct here, because it's adding, as in there's an internal vector
      //that's growing.
      gdcmDebugMacro("SOP Instance for plane " << theZ << " is " << theSOPInstance);

      theRTStruct->AddContourReferencedFrameOfReference(contour,
        theSOPClassID.c_str(), theSOPInstance.c_str());
      }
  }
}
