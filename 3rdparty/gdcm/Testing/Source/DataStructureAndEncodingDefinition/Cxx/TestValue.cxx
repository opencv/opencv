/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmByteValue.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmStringStream.h"

#include "gdcmSwapper.h"

void PrintStream(IStream &is)
{
  char c;
  while(is.get(c))
    {
    std::cout << (int)c << std::endl;
    }
}

int CheckStream(IStream &is, int size)
{
  char c;
  int t = 0;
  while(is.get(c) && (int)c == t)
    {
    std::cerr << (int)c << std::endl;
    ++t;
    }
  return t != size;
}

int TestValue(int , char *[])
{
  int r = 0;
  gdcm::Value *v;
  gdcm::SequenceOfItems si;
  gdcm::ByteValue bv;
  v = &si;
  v = &bv;

  const int size = 128;
  char buffer[size];
  for(int i=0; i<size;++i)
    {
    buffer[i] = static_cast<char>(i);
    }
  std::stringstream ss;
  ss.write(buffer, size);
  //PrintStream(ss);

  v->SetLength( size );
  v->Read<gdcm::SwapperNoOp>(ss);
  std::stringstream ss2;
  v->Write<gdcm::SwapperNoOp>(ss2);
  //PrintStream(ss2);
  r += CheckStream(ss2, size);

  return r;
}
