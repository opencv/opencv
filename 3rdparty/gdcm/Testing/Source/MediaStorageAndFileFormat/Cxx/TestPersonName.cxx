/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPersonName.h"
#include "gdcmByteValue.h"

#include <iostream>

int TestPersonName(int, char *[])
{
  typedef gdcm::PersonName PN;

  PN pn0;
  pn0.SetComponents();
  std::cout << "NumComp:" << pn0.GetNumberOfComponents() << std::endl;
  pn0.Print( std::cout );
  std::cout << std::endl;


  PN pn00 = {{ "" }};
  std::cout << "NumComp:" << pn00.GetNumberOfComponents() << std::endl;

  PN pn1 = {{ "abc123", "def", "ghi", "klm", "opq" }};
  pn1.Print( std::cout );
  std::cout << std::endl;
  std::cout << "NumComp:" << pn1.GetNumberOfComponents() << std::endl;

  PN pn2 = {{ "malaterre", "mathieu olivier patrick"}};
  pn2.Print( std::cout );
  std::cout << std::endl;
  std::cout << "NumComp:" << pn2.GetNumberOfComponents() << std::endl;

// Rev. John Robert Quincy Adams, B.A. M.Div. Adams^John Robert Quincy^^Rev.^B.A. M.Div. [One family name; three given names; no middle name; one prefix; two suffixes.]
        PN pn3 = {{ "Adams", "John Robert Quincy", "", "Rev.", "B.A. M.Div." }};
  pn3.Print( std::cout );
  std::cout << std::endl;
  std::cout << "NumComp:" << pn3.GetNumberOfComponents() << std::endl;
// Susan Morrison-Jones, Ph.D., Chief Executive Officer Morrison-Jones^Susan^^^Ph.D., Chief Executive Officer [Two family names; one given name; no middle name; no prefix; two suffixes.]
        PN pn4 = {{ "Morrison-Jones", "Susan", "", "", "Ph.D., Chief Executive Officer" }};
  pn4.Print( std::cout );
  std::cout << std::endl;
  std::cout << "NumComp:" << pn4.GetNumberOfComponents() << std::endl;

// John Doe Doe^John [One family name; one given name; no middle name, prefix, or suffix. Delimiters have been omitted for the three trailing null components.]
        PN pn5 = {{ "Doe", "John" }};
  pn5.Print( std::cout );
  std::cout << std::endl;
  std::cout << "NumComp:" << pn5.GetNumberOfComponents() << std::endl;


// (for examples of the encoding of Person Names using multi-byte character sets see Annex H)
// Smith^Fluffy [A cat, rather than a
//human, whose responsible party family name is Smith, and whose own name is Fluffy]
        PN pn6 = {{ "Smith", "Fluffy" }};
  pn6.Print( std::cout );
  std::cout << std::endl;
  std::cout << "NumComp:" << pn6.GetNumberOfComponents() << std::endl;
//ABC Farms^Running on Water [A horse whose responsible organization is named ABC Farms, and whose name is Running On Water]
        PN pn7 = {{ "ABC Farms", "Running on Water" }};
  pn7.Print( std::cout );
  std::cout << std::endl;
  std::cout << "NumComp:" << pn7.GetNumberOfComponents() << std::endl;

  gdcm::ByteValue bv;
  PN pn8;
  pn8.SetBlob( bv );

  return 0;
}
