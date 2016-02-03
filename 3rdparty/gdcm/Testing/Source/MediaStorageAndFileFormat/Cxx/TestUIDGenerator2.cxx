/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUIDGenerator.h"

#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <iterator>

#include <pthread.h>

const unsigned int nuids = 100;

void* func (void* argc)
{
  gdcm::UIDGenerator g;
  std::set<std::string> *uids= reinterpret_cast< std::set<std::string>* >(argc);
  for(unsigned int i = 0; i < nuids; i++)
    {
    const char *s = g.Generate();
    //std::cout << s << std::endl;
    if ( uids->count(s) == 1 )
      {
      std::cerr << "Already found: " << s << std::endl;
      //pthread_exit(); // How do I say this is an error...
      }
    uids->insert( s );
   }
  return NULL;
}

int TestUIDGenerator2(int , char *[])
{
  const unsigned int nthreads = 10; // multiple of 2 please
  pthread_t th[nthreads];
  std::set<std::string> uids[nthreads];
  unsigned int i;
  for (i = 0; i < nthreads; i++)
    {
    const int ret = pthread_create (&th[i], NULL, func, (void*)(uids+i));
    if( ret ) return 1;
    }
  for (i = 0; i < nthreads; i++)
    pthread_join (th[i], NULL);

  std::vector<std::string> v_one(nuids*nthreads);
  std::vector<std::string>::iterator it = v_one.begin();
  for(i = 0; i < nthreads; i+=2)
    {
    std::set_union(uids[i].begin(), uids[i].end(),
      uids[i+1].begin(), uids[i+1].end(), it);
    it += nuids*2;
    }
  std::cout << v_one.size() << std::endl;
  assert( v_one.size() == nuids * nthreads ); // programmer error

  std::copy(v_one.begin(), v_one.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

  std::set<std::string> global;
  for(it = v_one.begin(); it != v_one.end(); ++it)
    {
    global.insert( *it );
    }
  std::cout << "set:" << global.size() << std::endl;
  if( global.size() != nuids * nthreads )
    {
    return 1;
    }

  return 0;
}
