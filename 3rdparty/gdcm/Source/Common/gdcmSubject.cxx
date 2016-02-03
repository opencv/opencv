/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSubject.h"

#include "gdcmSmartPointer.h"
#include "gdcmCommand.h"
#include "gdcmEvent.h"

#include <list>

namespace gdcm
{
class Observer
{
public:
  Observer(Command* c,
           const Event * event,
           unsigned long tag) :m_Command(c),
                               m_Event(event),
                               m_Tag(tag)
  { }
  virtual ~Observer()
  { delete m_Event; }
  SmartPointer<Command> m_Command;
  const Event * m_Event;
  unsigned long m_Tag;
};

class SubjectInternals
{
public:
  SubjectInternals() {m_Count = 0;}
  ~SubjectInternals();
  unsigned long AddObserver(const Event & event, Command* cmd);
  unsigned long AddObserver(const Event & event, Command* cmd) const;
  void RemoveObserver(unsigned long tag);
  void RemoveAllObservers();
  void InvokeEvent( const Event & event, Subject* self);
  void InvokeEvent( const Event & event, const Subject* self);
  Command *GetCommand(unsigned long tag);
  bool HasObserver(const Event & event) const;
  bool PrintObservers(std::ostream& os, std::string indent) const;
private:
  std::list<Observer* > m_Observers;
  unsigned long m_Count;
};

SubjectInternals::
~SubjectInternals()
{
  for(std::list<Observer* >::iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    delete (*i);
    }
}


unsigned long
SubjectInternals::
AddObserver(const Event & event,
            Command* cmd)
{
  Observer* ptr = new Observer(cmd, event.MakeObject(), m_Count);
  m_Observers.push_back(ptr);
  m_Count++;
  return ptr->m_Tag;
}


unsigned long
SubjectInternals::
AddObserver(const Event & event,
            Command* cmd) const
{
  Observer* ptr = new Observer(cmd, event.MakeObject(), m_Count);
  SubjectInternals * me = const_cast<SubjectInternals *>( this );
  me->m_Observers.push_back(ptr);
  me->m_Count++;
  return ptr->m_Tag;
}


void
SubjectInternals::
RemoveObserver(unsigned long tag)
{
  for(std::list<Observer* >::iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    if((*i)->m_Tag == tag)
      {
      delete (*i);
      m_Observers.erase(i);
      return;
      }
    }
}


void
SubjectInternals::
RemoveAllObservers()
{
  for(std::list<Observer* >::iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    delete (*i);
    }
  m_Observers.clear();
}


void
SubjectInternals::
InvokeEvent( const Event & event,
             Subject* self)
{
  for(std::list<Observer* >::iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    const Event * e =  (*i)->m_Event;
    if(e->CheckEvent(&event))
      {
      (*i)->m_Command->Execute(self, event);
      }
    }
}

void
SubjectInternals::
InvokeEvent( const Event & event,
             const Subject* self)
{
  for(std::list<Observer* >::iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    const Event * e =  (*i)->m_Event;
    if(e->CheckEvent(&event))
      {
      (*i)->m_Command->Execute(self, event);
      }
    }
}


Command*
SubjectInternals::
GetCommand(unsigned long tag)
{
  for(std::list<Observer* >::iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    if ( (*i)->m_Tag == tag)
      {
      return (*i)->m_Command;
      }
    }
  return 0;
}

bool
SubjectInternals::
HasObserver(const Event & event) const
{
  for(std::list<Observer* >::const_iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    const Event * e =  (*i)->m_Event;
    if(e->CheckEvent(&event))
      {
      return true;
      }
    }
  return false;
}

bool
SubjectInternals::
PrintObservers(std::ostream& os, std::string indent) const
{
  if(m_Observers.empty())
    {
    return false;
    }

  for(std::list<Observer* >::const_iterator i = m_Observers.begin();
      i != m_Observers.end(); ++i)
    {
    const Event * e =  (*i)->m_Event;
    const Command* c = (*i)->m_Command; (void)c;
    os << indent << e->GetEventName() << "(" /*<< c->GetNameOfClass()*/ << ")\n";
    }
  return true;
}

Subject::Subject():Internals(new SubjectInternals)
{
}

Subject::~Subject()
{
  delete Internals;
}

unsigned long Subject::AddObserver(const Event & event, Command *cmd)
{
  return this->Internals->AddObserver(event,cmd);
}
unsigned long Subject::AddObserver(const Event & event, Command *cmd) const
{
  return this->Internals->AddObserver(event,cmd);
}

Command* Subject::GetCommand(unsigned long tag)
{
  return this->Internals->GetCommand(tag);
}

void Subject::InvokeEvent( const Event & event)
{
  this->Internals->InvokeEvent(event,this);
}

void Subject::InvokeEvent( const Event & event) const
{
  this->Internals->InvokeEvent(event,this);
}

void Subject::RemoveObserver(unsigned long tag)
{
  this->Internals->RemoveObserver(tag);
}

void Subject::RemoveAllObservers()
{
  this->Internals->RemoveAllObservers();
}

bool Subject::HasObserver( const Event & event ) const
{
  return this->Internals->HasObserver(event);
}

}
