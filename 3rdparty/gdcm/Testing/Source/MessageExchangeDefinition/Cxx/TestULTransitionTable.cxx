/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmULTransitionTable.h"
#include "gdcmULActionAA.h"

int TestULTransitionTable(int , char *[])
{
  gdcm::network::Transition t1;
  gdcm::network::Transition *t2 =
    gdcm::network::Transition::MakeNew(
      gdcm::network::eSta1Idle,
      new gdcm::network::ULActionAA2()
    );
  gdcm::network::TableRow tr1;
  //tr1.transitions[0] = &t1; // no stack please
  tr1.transitions[1] = t2;
  gdcm::network::ULTransitionTable o;
  return 0;
}
