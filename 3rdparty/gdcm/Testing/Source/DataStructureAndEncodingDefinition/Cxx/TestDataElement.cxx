/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDataElement.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmTag.h"

#include "gdcmSwapper.h"

#include <fstream>

inline void DebugElement(std::stringstream const &os)
{
  std::ofstream of("/tmp/bla.bin", std::ios::binary);
  std::string str = os.str();
  of.write(str.c_str(), str.size());
  of.close();
}


inline void WriteRead(gdcm::DataElement const &w, gdcm::DataElement &r)
{
  // w will be written
  // r will be read back
  std::stringstream ss;
  w.Write<gdcm::ExplicitDataElement,gdcm::SwapperNoOp>(ss);
  r.Read<gdcm::ExplicitDataElement,gdcm::SwapperNoOp>(ss);
}

int TestDataElement1(const uint16_t group, const uint16_t element,
  const uint16_t vl)
{
  const char *str;
  std::stringstream ss;
  // SimpleData Element, just group,element and length
  str = reinterpret_cast<const char*>(&group);
  ss.write(str, sizeof(group));
  str = reinterpret_cast<const char*>(&element);
  ss.write(str, sizeof(element));
  str = reinterpret_cast<const char*>(&vl);
  ss.write(str, sizeof(vl));

//  gdcm::DataElement de;
//  de.Read( ss );
//  if( de.GetTag().GetGroup()   != group ||
//      de.GetTag().GetElement() != element ||
//      de.GetVL()               != vl )
//    {
//    std::cerr << de << std::endl;
//    return 1;
//    }
//
//  gdcm::DataElement de2;
//  WriteRead(de, de2);
//  if( !(de == de2) )
//    {
//    std::cerr << de << std::endl;
//    std::cerr << de2 << std::endl;
//    return 1;
//    }

  return 0;
}

// Explicit
int TestDataElement2(const uint16_t group, const uint16_t element,
  const char* vr, const char* value)
{
  const char *str;
  const uint32_t vl = (uint32_t)strlen( value );
  std::stringstream ss;
  // SimpleData Element, just group,element and length
  str = reinterpret_cast<const char*>(&group);
  ss.write(str, sizeof(group));
  str = reinterpret_cast<const char*>(&element);
  ss.write(str, sizeof(element));
  if( gdcm::VR::GetVRType(vr) == gdcm::VR::INVALID )
    {
    std::cerr << "Test buggy" << std::endl;
    return 1;
    }
  ss.write(vr, strlen(vr) );
  str = reinterpret_cast<const char*>(&vl);
  ss.write(str, sizeof(vl));
  assert( !(strlen(value) % 2) );
  ss.write(value, strlen(value) );
  //DebugElement(ss);

  gdcm::ExplicitDataElement de;
  de.Read<gdcm::SwapperNoOp>( ss );
  if( de.GetTag().GetGroup()   != group ||
      de.GetTag().GetElement() != element ||
      de.GetVL()               != vl )
    {
    std::cerr << de << std::endl;
    return 1;
    }

  gdcm::ExplicitDataElement de2;
  WriteRead(de, de2);
  if( !(de == de2) )
    {
    std::cerr << de << std::endl;
    std::cerr << de2 << std::endl;
    return 1;
    }

  return 0;
}

namespace
{

  // Tests operator== and operator!= for various gdcm::DataElement objects.
  // Also tests comparing to itself, symmetry of the comparison operators
  // with respect to their operands, and consistency between 
  // operator== and operator!=.
  // Note: This function recursively calls itself, in order to get a pointer
  // to an equivalent array of gdcm::DataElement objects.
  int TestDataElementEqualityComparison(
    const gdcm::DataElement* const equivalentDataElements = NULL)
  {
    const unsigned int numberOfDataElements = 6;

    gdcm::DataElement dataElements[numberOfDataElements] =
    {
      gdcm::DataElement(gdcm::Tag(0, 0), 0, gdcm::VR::INVALID),
      gdcm::DataElement(gdcm::Tag(1, 1), 0, gdcm::VR::INVALID),
      gdcm::DataElement(gdcm::Tag(1, 1), 1, gdcm::VR::INVALID),
      gdcm::DataElement(gdcm::Tag(1, 1), 1, gdcm::VR::AE)
    };

    dataElements[4].SetByteValue("\0", 2);
    dataElements[5].SetByteValue("123", 4);

    // Now all data elements of the array dataElements are different. 

    if ( equivalentDataElements == NULL )
    {
      return TestDataElementEqualityComparison(dataElements);
    }

    // equivalentDataElements != NULL, and because this function 
    // is called recursively, equivalentDataElements[i] is equivalent 
    // to dataElements[i].

    for (unsigned int i = 0; i < numberOfDataElements; ++i)
    {
      const gdcm::DataElement& dataElement = dataElements[i]; 

      if ( ! (dataElement == dataElement) )
      {
        std::cerr <<
          "Error: A data element should compare equal to itself!\n";
        return 1;
      }
      if ( dataElement != dataElement )
      {
        std::cerr <<
          "Error: A data element should not compare unequal to itself!\n";
        return 1;
      }
      const gdcm::DataElement& equivalentDataElement = equivalentDataElements[i]; 

      if (  ! (dataElement == equivalentDataElement) ||
        ! (equivalentDataElement == dataElement ) )
      {
        std::cerr <<
          "Error: A data element should compare equal to an equivalent one!\n";
        return 1;
      }

      if ( (dataElement != equivalentDataElement) ||
        (equivalentDataElement != dataElement) )
      {
        std::cerr <<
          "Error: A data element should not compare unequal to an equivalent one!\n";
        return 1;
      }

      for (unsigned int j = i + 1; j < numberOfDataElements; ++j)
      {
        // dataElements[j] is different from dataElements[i].

        const gdcm::DataElement& differentDataElement = dataElements[j];

        if ( (dataElement == differentDataElement) ||
          (differentDataElement == dataElement) )
        {
          std::cerr <<
            "Error: A data element should not compare equal to a different one!\n";
          return 1;
        }

        if ( !(dataElement != differentDataElement) ||
          !(differentDataElement != dataElement) )
        {
          std::cerr <<
            "Error: A data element should compare unequal to a different one!\n";
          return 1;
        }
      }
    }
    return 0;
  }

} // End of unnamed namespace.

// Test Data Element
int TestDataElement(int , char *[])
{
  const uint16_t group = 0x0010;
  const uint16_t element = 0x0012;
  const uint16_t vl = 0x0;
  int r = 0;
  r += TestDataElement1(group, element, vl);
  r += TestDataElementEqualityComparison();

  // Full DataElement
  //const char vr[] = "UN";
  //const char value[] = "ABCDEFGHIJKLMNOP";
  //r += TestDataElement2(group, element, vr, value);

  return r;
}
