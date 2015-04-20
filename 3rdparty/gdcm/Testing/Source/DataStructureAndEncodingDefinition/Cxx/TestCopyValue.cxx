/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmValue.h"
#include "gdcmByteValue.h"

#include <tr1/memory>

//namespace std { namespace tr1 = Boost; }

void RawPtr(const char *array, const size_t len)
{
  std::cout << "RawPtr" << std::endl;
  gdcm::ByteValue *bv1 = new gdcm::ByteValue(array, len);
  std::cout << *bv1 << std::endl;
  gdcm::ByteValue *bv2 = new gdcm::ByteValue(array, len);
  std::cout << *bv2 << std::endl;

  const gdcm::Value &ref1 = *bv1;
  std::cout << ref1 << std::endl;

  const gdcm::Value &ref2 = *bv2;
  std::cout << ref2 << std::endl;

  // Need to delete:
  delete bv1;
  delete bv2;
}

typedef std::tr1::shared_ptr<gdcm::Value> ValuePtr;
void SharedPtr(const char *array, const size_t len)
{
  std::cout << "SharedPtr" << std::endl;

  //ValuePtr w;
  //std::cout << w << std::endl;
  ValuePtr bv1 ( new gdcm::ByteValue(array, len) );
  std::cout << *bv1 << std::endl;
  ValuePtr bv2 ( new gdcm::ByteValue(array, len) );
  std::cout << *bv2 << std::endl;

  const gdcm::Value &ref1 = *bv1;
  std::cout << ref1 << std::endl;

  const gdcm::Value &ref2 = *bv2;
  std::cout << ref2 << std::endl;

  // No need to delete :)
}

int TestCopyValue(int , char *[])
{

  const char array[] = "GDCM is yet another DICOM library.";
  const size_t len = strlen( array );
  RawPtr(array, len);
  SharedPtr(array, len);

  std::vector<ValuePtr> v;
  v.push_back( ValuePtr(new gdcm::ByteValue(array, len) ));
  v.push_back( ValuePtr(new gdcm::ByteValue(array, len) ));
  v.push_back( ValuePtr(new gdcm::ByteValue(array, len) ));
  v.push_back( ValuePtr(new gdcm::ByteValue(array, len) ));
  v.push_back( ValuePtr(new gdcm::ByteValue(array, len) ));
  // no delete :)

  return 0;
}
