/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCommand.h"
#include "gdcmEvent.h"
#include "gdcmSmartPointer.h"

using gdcm::SmartPointer;

struct Watcher {};

void foo(gdcm::Command *c)
{
  c->Execute((gdcm::Subject*)0, gdcm::AnyEvent() );
}

int TestCommand(int , char *[])
{
  SmartPointer<gdcm::MemberCommand<Watcher> > mc = gdcm::MemberCommand<Watcher>::New();
  foo(mc);
  SmartPointer<gdcm::SimpleMemberCommand<Watcher> > smc = gdcm::SimpleMemberCommand<Watcher>::New();
  foo(smc);
  return 0;
}
