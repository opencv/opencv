/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAttribute.h"

#include <limits>
#include <math.h> // fabs

int TestAttributeAE() { return 0; }
int TestAttributeAS() { return 0; }
int TestAttributeAT() { return 0; }

/*
int TestAttributeCS()
{
  // (0008,9007) CS [ORIGINAL\PRIMARY\T1\NONE]               #  24, 4 FrameType
  static const char* values[] = {"ORIGINAL","PRIMARY","T1","NONE"};
  static const char* newvalues[] = {"DERIVED","SECONDARY","T2","ALL"};
  const unsigned int numvalues = sizeof(values) / sizeof(values[0]);
  if( numvalues != 4 ) return 1;

  gdcm::Attribute<0x0008,0x9007> it = {"ORIGINAL","PRIMARY","T1","NONE"};
  // FIXME HARDCODED:
  if( it.GetVM() != gdcm::VM::VM4 ) return 1;
  if( it.GetVR() != gdcm::VR::CS ) return 1;
  // END FIXME

  if( it.GetNumberOfValues() != numvalues ) return 1;

  for(unsigned int i = 0; i < numvalues; ++i)
    if( it.GetValue(i) != values[i] ) return 1;

  it.Print( std::cout );
  std::cout << std::endl;

  gdcm::DataElement de = it.GetAsDataElement();
  std::cout << de << std::endl;

  // new values:
  // Using implicit cstor of gdcm::String from const char *
  for(unsigned int i = 0; i < numvalues; ++i)
    it.SetValue( newvalues[i], i );
  if( it.GetNumberOfValues() != numvalues ) return 1;

  for(unsigned int i = 0; i < numvalues; ++i)
    if( it.GetValue(i) != newvalues[i] ) return 1;

  // const char * is not a gdcm::String, need an array of gdcm::String
  static const gdcm::String<> newvalues2[] = {"DERIVED","SECONDARY","T2","ALL"};
  const unsigned int numnewvalues2 = sizeof(newvalues2) / sizeof(newvalues2[0]);
  it.SetValues( newvalues2 );

  it.Print( std::cout );
  std::cout << std::endl;

  de = it.GetAsDataElement();
  std::cout << de << std::endl;

  // (0008,0008) CS [DERIVED\PRIMARY\AXIAL]                  #  22, 3 ImageType
  gdcm::Attribute<0x0008,0x0008> it1;
  if( it1.GetVM() != gdcm::VM::VM2_n )
    {
    std::cerr << "Wrong VM:" << it1.GetVM() << std::endl;
    return 1;
    }
  it1.SetValues( newvalues2, numnewvalues2 );

  it1.Print( std::cout );
  std::cout << std::endl;

  de = it1.GetAsDataElement();
  std::cout << de << std::endl;

  // redo the same but this time copy the values:
  it1.SetValues( newvalues2, numnewvalues2, true );

  it1.Print( std::cout );
  std::cout << std::endl;

  de = it1.GetAsDataElement();
  std::cout << de << std::endl;

  return 0;
}
*/

int TestAttributeDA() { return 0; }

int TestAttributeDS()
{
  // (0020,0032) DS [-158.135803\-179.035797\-75.699997]     #  34, 3 ImagePositionPatient
  const double values[] = {-158.135803,-179.035797,-75.699997};
  const double newvalues[] = {12.34,56.78,90.0};
  const unsigned int numvalues = sizeof(values) / sizeof(values[0]);

  gdcm::Attribute<0x0020,0x0032> ipp = {{-158.135803,-179.035797,-75.699997}};
  // FIXME HARDCODED:
  if( ipp.GetVM() != gdcm::VM::VM3 ) return 1;
  if( ipp.GetVR() != gdcm::VR::DS ) return 1;
  // END FIXME
  if( ipp.GetNumberOfValues() != numvalues ) return 1;

  for(unsigned int i = 0; i < numvalues; ++i)
    if( fabs(ipp.GetValue(i) - values[i]) > std::numeric_limits<float>::epsilon() ) return 1;

  ipp.Print( std::cout );
  std::cout << std::endl;

  gdcm::DataElement de = ipp.GetAsDataElement();
  std::cout << de << std::endl;

  // new values:
  ipp.SetValues( newvalues );
  if( ipp.GetNumberOfValues() != numvalues ) return 1;

  for(unsigned int i = 0; i < numvalues; ++i)
    if( fabs(ipp.GetValue(i) - newvalues[i]) > std::numeric_limits<float>::epsilon() ) return 1;

  ipp.Print( std::cout );
  std::cout << std::endl;

  de = ipp.GetAsDataElement();
  std::cout << de << std::endl;

{
  //const char v[] = "0.960000000000662 "; // not working
  const char v[] = "1.960000000000662 ";
  gdcm::DataElement invalid( gdcm::Tag(0x10,0x1030) ); // Patient's Weight
  invalid.SetVR( gdcm::VR::DS );
  invalid.SetByteValue( v, (uint32_t)strlen(v) );

  gdcm::Attribute<0x0010,0x1030> pw;
  pw.SetFromDataElement( invalid );

  gdcm::DataElement valid = pw.GetAsDataElement();
  std::ostringstream os;
  os << valid.GetValue();
  size_t l = os.str().size();
  if( l > 16 )
    {
    return 1;
    }


}

  return 0;
}

int TestAttributeDT() { return 0; }
int TestAttributeFL() { return 0; }
int TestAttributeFD() { return 0; }
int TestAttributeIS()
{
  // <entry group="0018" element="1182" vr="IS" vm="1-2" name="Focal Distance"/>
  // This case is slightly more complex it is up to the user to say what is the VM:
  gdcm::Attribute<0x0018,0x1182, gdcm::VR::IS, gdcm::VM::VM1> fd1 = {0};
  if( fd1.GetVM() != gdcm::VM::VM1 ) return 1;

  gdcm::Attribute<0x0018,0x1182, gdcm::VR::IS, gdcm::VM::VM2> fd2 = {{0,1}};
  if( fd2.GetVM() != gdcm::VM::VM2 ) return 1;

  // this one should not be allowed, I need a special CTest macro...
  //gdcm::Attribute<0x0018,0x1182, gdcm::VR::IS, gdcm::VM::VM3> fd3 = {0,1};
  //return 1;

  return 0;
}

int TestAttributeLO() { return 0; }
int TestAttributeLT() { return 0; }
int TestAttributeOB() { return 0; }
int TestAttributeOF()
{
  gdcm::DataSet ds;
  const float array[] = { 0, 1, 2, 3, 4 };
  gdcm::Attribute<0x5600,0x0020, gdcm::VR::OF, gdcm::VM::VM1_n> at;
  at.SetValues( array, sizeof( array ) / sizeof( *array ) );
  ds.Insert( at.GetAsDataElement() );

  if( at.GetNumberOfValues() != 5 ) return 1;

  // Sup 132
  // Tag : (0x0066,0x0016), VR : OF, VM : 1, Type : 1
  gdcm::Attribute<0x0066,0x0016> at1;
  float value = 1.f;
  at1.SetValue( value );
  ds.Insert( at1.GetAsDataElement() );

  return 0;
}
int TestAttributeOW() { return 0; }
int TestAttributePN() { return 0; }
int TestAttributeSH() { return 0; }
int TestAttributeSL() { return 0; }
int TestAttributeSQ() { return 0; }
int TestAttributeSS() { return 0; }
int TestAttributeST() { return 0; }
int TestAttributeTM() { return 0; }
int TestAttributeUI() { return 0; }
int TestAttributeUL() { return 0; }
int TestAttributeUN() { return 0; }
int TestAttributeUS() { return 0; }
int TestAttributeUT() { return 0; }


int TestAttribute1(int , char *[])
{
  int numerrors = 0;
  numerrors += TestAttributeAE();
  numerrors += TestAttributeAS();
  numerrors += TestAttributeAT();
  //numerrors += TestAttributeCS();
  numerrors += TestAttributeDA();
  numerrors += TestAttributeDS();
  numerrors += TestAttributeDT();
  numerrors += TestAttributeFL();
  numerrors += TestAttributeFD();
  numerrors += TestAttributeIS();
  numerrors += TestAttributeLO();
  numerrors += TestAttributeLT();
  numerrors += TestAttributeOB();
  numerrors += TestAttributeOF();
  numerrors += TestAttributeOW();
  numerrors += TestAttributePN();
  numerrors += TestAttributeSH();
  numerrors += TestAttributeSL();
  numerrors += TestAttributeSQ();
  numerrors += TestAttributeSS();
  numerrors += TestAttributeST();
  numerrors += TestAttributeTM();
  numerrors += TestAttributeUI();
  numerrors += TestAttributeUL();
  numerrors += TestAttributeUN();
  numerrors += TestAttributeUS();
  numerrors += TestAttributeUT();

  return numerrors;
}
