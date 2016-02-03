/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCurve.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmAttribute.h"

#include <vector>

namespace gdcm
{
/*

C.10.2.1 Curve Attribute Descriptions
C.10.2.1.1 Type of data
A description of the Type of Data (50xx,0020) in this curve.
Defined Terms:
TAC = time activity curve PROF = image profile
HIST = histogram ROI = polygraphic region of interest
TABL = table of values FILT = filter kernel
POLY = poly line ECG = ecg data
PRESSURE = pressure data FLOW = flow data
PHYSIO = physio data RESP = Respiration trace
*/

class CurveInternal
{
public:
  CurveInternal():
  Group(0),
  Dimensions(0),
  NumberOfPoints(0),
  TypeOfData(),
  CurveDescription(),
  DataValueRepresentation(0),
  Data() {}

/*
(5004,0000) UL 2316                                     #   4, 1 CurveGroupLength
(5004,0005) US 1                                        #   2, 1 CurveDimensions
(5004,0010) US 1126                                     #   2, 1 NumberOfPoints
(5004,0020) CS [PHYSIO]                                 #   6, 1 TypeOfData
(5004,0022) LO (no value available)                     #   0, 0 CurveDescription
(5000,0030) SH [DPPS\NONE]                              #  10, 2 AxisUnits
(5004,0103) US 0                                        #   2, 1 DataValueRepresentation
(5000,0110) US 0\1                                      #   4, 2 CurveDataDescriptor
(5000,0112) US 0                                        #   2, 1 CoordinateStartValue
(5000,0114) US 300                                      #   2, 1 CoordinateStepValue
(5000,2500) LO [Physio_1]                               #   8, 1 CurveLabel
(5004,3000) OW 0020\0020\0020\0020\0020\0020\0020\0020\0020\0020\0020\0020\0020... # 2252, 1 CurveData
*/
// Identifier need to be in the [5000,50FF] range (no odd number):
  unsigned short Group;
  unsigned short Dimensions;
  unsigned short NumberOfPoints;
  std::string TypeOfData;
  std::string CurveDescription;
  unsigned short DataValueRepresentation;
  std::vector<char> Data;
  std::vector<unsigned short> CurveDataDescriptor;
  unsigned short CoordinateStartValue;
  unsigned short CoordinateStepValue;
  void Print(std::ostream &os) const {
    os << "Group           0x" <<  std::hex << Group << std::dec << std::endl;
    os << "Dimensions                         :" << Dimensions << std::endl;
    os << "NumberOfPoints                     :" << NumberOfPoints << std::endl;
    os << "TypeOfData                         :" << TypeOfData << std::endl;
    os << "CurveDescription                   :" << CurveDescription << std::endl;
    os << "DataValueRepresentation            :" << DataValueRepresentation << std::endl;
    unsigned short * p = (unsigned short*)&Data[0];
    for(int i = 0; i < NumberOfPoints; i+=2)
      {
      os << p[i] << "," << p[i+1] << std::endl;
      }
  }
};

Curve::Curve()
{
  Internal = new CurveInternal;
}

Curve::~Curve()
{
  delete Internal;
}

Curve::Curve(Curve const &ov):Object(ov)
{
  //delete Internal;
  Internal = new CurveInternal;
  // TODO: copy CurveInternal into other...
  *Internal = *ov.Internal;
}

void Curve::Print(std::ostream &os) const
{
  Internal->Print( os );
}

unsigned int Curve::GetNumberOfCurves(DataSet const & ds)
{
  Tag overlay(0x5000,0x0000); // First possible overlay
  bool finished = false;
  unsigned int numoverlays = 0;
  while( !finished )
    {
    const DataElement &de = ds.FindNextDataElement( overlay );
    if( de.GetTag().GetGroup() > 0x50FF ) // last possible curve
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
      // Yeah this is an overlay element
      //if( ds.FindDataElement( Tag(overlay.GetGroup(),0x3000 ) ) )
      if( ds.FindDataElement( Tag(de.GetTag().GetGroup(),0x3000 ) ) )
        {
        // ok so far so good...
        //const DataElement& overlaydata = ds.GetDataElement(Tag(overlay.GetGroup(),0x3000));
        const DataElement& overlaydata = ds.GetDataElement(Tag(de.GetTag().GetGroup(),0x3000));
        if( !overlaydata.IsEmpty() )
          {
          ++numoverlays;
          }
        }
      // Store found tag in overlay:
      overlay = de.GetTag();
      // Move on to the next possible one:
      overlay.SetGroup( (uint16_t)(overlay.GetGroup() + 2) );
      // reset to element 0x0 just in case...
      overlay.SetElement( 0 );
      }
    }

  return numoverlays;
}

void Curve::Update(const DataElement & de)
{
  assert( de.GetTag().IsPublic() );
  const ByteValue* bv = de.GetByteValue();
  if( !bv ) return; // Discard any empty element (will default to another value)
  assert( bv->GetPointer() && bv->GetLength() );
  std::string s( bv->GetPointer(), bv->GetLength() );
  // What if a \0 can be found before the end of string...
  //assert( strlen( s.c_str() ) == s.size() );

  // First thing check consistency:
  if( !GetGroup() )
    {
    SetGroup( de.GetTag().GetGroup() );
    }
  else // check consistency
    {
    assert( GetGroup() == de.GetTag().GetGroup() ); // programmer error
    }

  //std::cerr << "Tag: " << de.GetTag() << std::endl;
  if( de.GetTag().GetElement() == 0x0000 ) // CurveGroupLength
    {
    // ??
    }
  else if( de.GetTag().GetElement() == 0x0005 ) // CurveDimensions
    {
    Attribute<0x5000,0x0005> at;
    at.SetFromDataElement( de );
    SetDimensions( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0010 ) // NumberOfPoints
    {
    Attribute<0x5000,0x0010> at;
    at.SetFromDataElement( de );
    SetNumberOfPoints( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0020 ) // TypeOfData
    {
    SetTypeOfData( s.c_str() );
    }
  else if( de.GetTag().GetElement() == 0x0022 ) // CurveDescription
    {
    SetCurveDescription( s.c_str() );
    }
  else if( de.GetTag().GetElement() == 0x0030 ) // AxisUnits
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x0040 ) // Axis Labels
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x0103 ) // DataValueRepresentation
    {
    Attribute<0x5000,0x0103> at;
    at.SetFromDataElement( de );
    SetDataValueRepresentation( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0104 ) // Minimum Coordinate Value
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x0105 ) // Maximum Coordinate Value
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x0106 ) // Curve Range
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x0110 ) // CurveDataDescriptor
    {
    Attribute<0x5000,0x0110> at;
    at.SetFromDataElement( de );
    SetCurveDataDescriptor( at.GetValues(), at.GetNumberOfValues() );
    }
  else if( de.GetTag().GetElement() == 0x0112 ) // CoordinateStartValue
    {
    Attribute<0x5000,0x0112> at;
    at.SetFromDataElement( de );
    SetCoordinateStartValue( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0114 ) // CoordinateStepValue
    {
    Attribute<0x5000,0x0114> at;
    at.SetFromDataElement( de );
    SetCoordinateStepValue( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x2500 ) // CurveLabel
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x2600 ) // Referenced Overlay Sequence
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x2610 ) // Referenced Overlay Group
    {
    gdcmWarningMacro( "TODO" );
    }
  else if( de.GetTag().GetElement() == 0x3000 ) // CurveData
    {
    SetCurve(bv->GetPointer(), bv->GetLength());
    }
  else
    {
    assert( 0 && "should not happen: Unknown curve tag" );
    }

}

void Curve::SetGroup(unsigned short group) { Internal->Group = group; }
unsigned short Curve::GetGroup() const { return Internal->Group; }
void Curve::SetDimensions(unsigned short dimensions) { Internal->Dimensions = dimensions; }
unsigned short Curve::GetDimensions() const { return Internal->Dimensions; }
void Curve::SetNumberOfPoints(unsigned short numberofpoints) { Internal->NumberOfPoints = numberofpoints; }
unsigned short Curve::GetNumberOfPoints() const { return Internal->NumberOfPoints; }
void Curve::SetTypeOfData(const char *typeofdata)
{
  if( typeofdata )
    Internal->TypeOfData = typeofdata;
}
const char *Curve::GetTypeOfData() const { return Internal->TypeOfData.c_str(); }

static const char * const TypeOfDataDescription[][2] = {
{ "TAC" , "time activity curve" },
{ "PROF" , "image profile" },
{ "HIST" , "histogram" },
{ "ROI" , "polygraphic region of interest" },
{ "TABL" , "table of values" },
{ "FILT" , "filter kernel" },
{ "POLY" , "poly line" },
{ "ECG" , "ecg data" },
{ "PRESSURE" , "pressure data" },
{ "FLOW" , "flow data" },
{ "PHYSIO" , "physio data" },
{ "RESP" , "Respiration trace" },
{ 0 , 0 }
};
const char *Curve::GetTypeOfDataDescription() const
{
  typedef const char* const (*TypeOfDataDescriptionType)[2];
  TypeOfDataDescriptionType t = TypeOfDataDescription;
  int i = 0;
  const char *p = t[i][0];
  while( p )
    {
    if( Internal->TypeOfData == p )
      {
      break;
      }
    ++i;
    p = t[i][0];
    }
  return t[i][1];
}

void Curve::SetCurveDescription(const char *curvedescription)
{
  if( curvedescription )
    Internal->CurveDescription = curvedescription;
}
void Curve::SetDataValueRepresentation(unsigned short datavaluerepresentation) { Internal->DataValueRepresentation = datavaluerepresentation; }
unsigned short Curve::GetDataValueRepresentation() const { return Internal->DataValueRepresentation; }

void Curve::SetCurveDataDescriptor(const uint16_t * values, size_t num)
{
  Internal->CurveDataDescriptor = std::vector<uint16_t>(values, values+num);
}
std::vector<unsigned short> const &Curve::GetCurveDataDescriptor() const
{
  return Internal->CurveDataDescriptor;
}

void Curve::SetCoordinateStartValue( unsigned short v )
{
  Internal->CoordinateStartValue = v;
}
void Curve::SetCoordinateStepValue( unsigned short v )
{
  Internal->CoordinateStepValue = v;
}

bool Curve::IsEmpty() const
{
  return Internal->Data.empty();
}

void Curve::SetCurve(const char *array, unsigned int length)
{
  if( !array || length == 0 ) return;
  Internal->Data.resize( length );
  std::copy(array, array+length, Internal->Data.begin());
  //assert( 8 * length == (unsigned int)Internal->Rows * Internal->Columns );
  //assert( Internal->Data.size() == length );
}

void Curve::Decode(std::istream &is, std::ostream &os)
{
  (void)is;
  (void)os;
  assert(0);
}

/*
PS 3.3 - 2004
C.10.2.1.2 Data value representation
0000H = unsigned short (US)
0001H = signed short (SS)
0002H = floating point single (FL)
0003H = floating point double (FD)
0004H = signed long (SL)
*/
inline size_t getsizeofrep( unsigned short dr )
{
  size_t val = 0;
  switch( dr )
    {
  case 0:
    val = sizeof( uint16_t );
    break;
  case 1:
    val = sizeof( int16_t );
    break;
  case 2:
    val = sizeof( float );
    break;
  case 3:
    val = sizeof( double );
    break;
  case 4:
    val = sizeof( int32_t );
    break;
    }
  return val;
}

/*
C.10.2.1.5 Curve data descriptor, coordinate start value, coordinate step value
The Curve Data for dimension(s) containing evenly distributed data can be eliminated by using a
method that defines the Coordinate Start Value and Coordinate Step Value (interval). The one
dimensional data list is then calculated rather than being enumerated.
For the Curve Data Descriptor (50xx,0110) an Enumerated Value describing how each
component of the N-tuple curve is described, either by points or interval spacing. One value for
each dimension. Where:
0000H = Dimension component described using interval spacing
0001H = Dimension component described using values
Using interval spacing:
Dimension component(s) described by interval spacing use Attributes of Coordinate Start Value
(50xx,0112), Coordinate Step Value (50xx,0114) and Number of Points (50xx,0010). The 1-
dimensional data list is calculated by using a start point of Coordinate Start Value and adding the
interval (Coordinate Step Value) to obtain each data point until the Number of Points is satisfied.
The data points of this dimension will be absent from Curve Data (50xx,3000).
*/
double Curve::ComputeValueFromStartAndStep(unsigned int idx) const
{
  assert( !Internal->CurveDataDescriptor.empty() );
  const double res = Internal->CoordinateStartValue +
    Internal->CoordinateStepValue * idx;
  return res;
}

void Curve::GetAsPoints(float *array) const
{
  assert( getsizeofrep(Internal->DataValueRepresentation) );
  if( Internal->CurveDataDescriptor.empty() )
    {
    assert( Internal->Data.size() == (uint32_t)Internal->NumberOfPoints *
      Internal->Dimensions * getsizeofrep( Internal->DataValueRepresentation) );
    }
  else
    {
    assert( Internal->Data.size() == (uint32_t)Internal->NumberOfPoints *
      1 * getsizeofrep( Internal->DataValueRepresentation) );
    }
  assert( Internal->Dimensions == 1 || Internal->Dimensions == 2 );

  const int mult = Internal->Dimensions;
  int genidx = -1;
  if( !Internal->CurveDataDescriptor.empty() )
    {
    assert( Internal->CurveDataDescriptor.size() == Internal->Dimensions );
    assert( Internal->CurveDataDescriptor.size() == 2 ); // FIXME
    if( Internal->CurveDataDescriptor[0] == 0 )
      {
      assert( Internal->CurveDataDescriptor[1] == 1 );
      genidx = 0;
      }
    else if( Internal->CurveDataDescriptor[1] == 0 )
      {
      assert( Internal->CurveDataDescriptor[0] == 1 );
      genidx = 1;
      }
    else
      {
      assert( 0 && "TODO" );
      }
    }
  const char * beg = &Internal->Data[0];
  const char * end = beg + Internal->Data.size();
  if( genidx == -1 )
    {
    assert( end == beg + 2 * Internal->NumberOfPoints ); (void)beg;(void)end;
    }
  else
    {
    assert( end == beg + mult * Internal->NumberOfPoints ); (void)beg;(void)end;
    }


  if( Internal->DataValueRepresentation == 0 )
    {
    // PS 3.3 - 2004
    // C.10.2.1.5 Curve data descriptor, coordinate start value, coordinate step value
    uint16_t * p = (uint16_t*)&Internal->Data[0];
    // X
    if( genidx == 0 )
      for(int i = 0; i < Internal->NumberOfPoints; i++ )
        array[3*i+0] = ComputeValueFromStartAndStep( i );
    else
      for(int i = 0; i < Internal->NumberOfPoints; i++ )
        array[3*i+0] = p[i + 0];
    // Y
    if( genidx == 1 )
      for(int i = 0; i < Internal->NumberOfPoints; i++ )
        array[3*i+1] = ComputeValueFromStartAndStep( i );
    else
      {
      if( mult == 2 && genidx == -1 )
        {
        for(int i = 0; i < Internal->NumberOfPoints; i++ )
          array[3*i+1] = p[i + 1];
        }
      else if( mult == 2 && genidx == 0 )
        {
        for(int i = 0; i < Internal->NumberOfPoints; i++ )
          array[3*i+1] = p[i + 0];
        }
      else
        {
        for(int i = 0; i < Internal->NumberOfPoints; i++ )
          array[3*i+1] = 0;
        }
      }
    // Z
    for(int i = 0; i < Internal->NumberOfPoints; i++ )
      array[3*i+2] = 0;
    }
  else if( Internal->DataValueRepresentation == 1 )
    {
    int16_t * p = (int16_t*)&Internal->Data[0];
    for(int i = 0; i < Internal->NumberOfPoints; i++ )
      {
      array[3*i+0] = p[mult*i + 0];
      if( mult > 1 )
        array[3*i+1] = p[mult*i + 1];
      else
        array[3*i+1] = 0;
      array[3*i+2] = 0;
      }
    }
  else if( Internal->DataValueRepresentation == 2 )
    {
    float * p = (float*)&Internal->Data[0];
    for(int i = 0; i < Internal->NumberOfPoints; i++ )
      {
      array[3*i+0] = p[mult*i + 0];
      if( mult > 1 )
        array[3*i+1] = p[mult*i + 1];
      else
        array[3*i+1] = 0;
      array[3*i+2] = 0;
      }
    }
  else if( Internal->DataValueRepresentation == 3 )
    {
    double * p = (double*)&Internal->Data[0];
    for(int i = 0; i < Internal->NumberOfPoints; i++ )
      {
      array[3*i+0] = (float)p[mult*i + 0];
      if( mult > 1 )
        array[3*i+1] = (float)p[mult*i + 1];
      else
        array[3*i+1] = 0;
      array[3*i+2] = 0;
      }
    }
  else if( Internal->DataValueRepresentation == 4 )
    {
    int32_t * p = (int32_t*)&Internal->Data[0];
    for(int i = 0; i < Internal->NumberOfPoints; i++ )
      {
      array[3*i+0] = (float)p[mult*i + 0];
      if( mult > 1 )
        array[3*i+1] = (float)p[mult*i + 1];
      else
        array[3*i+1] = 0;
      array[3*i+2] = 0;
      }
    }
  else
    {
    assert( 0 );
    }
}

}
