/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMANONYMIZEEVENT_H
#define GDCMANONYMIZEEVENT_H

#include "gdcmEvent.h"
#include "gdcmTag.h"

namespace gdcm
{

/**
 * \brief AnonymizeEvent
 * Special type of event triggered during the Anonymization process
 *
 * \see Anonymizer
 */
class AnonymizeEvent : public AnyEvent
{
public:
  typedef AnonymizeEvent Self;
  typedef AnyEvent Superclass;
  AnonymizeEvent(Tag const &tag = 0):m_Tag(tag) {}
  virtual ~AnonymizeEvent() {}
  virtual const char * GetEventName() const { return "AnonymizeEvent"; }
  virtual bool CheckEvent(const ::gdcm::Event* e) const
  { return (dynamic_cast<const Self*>(e) == NULL ? false : true) ; }
  virtual ::gdcm::Event* MakeObject() const
    { return new Self; }
  AnonymizeEvent(const Self&s) : AnyEvent(s){};

  void SetTag(const Tag& t ) { m_Tag = t; }
  Tag const & GetTag() const { return m_Tag; }
private:
  void operator=(const Self&);
  Tag m_Tag;
};


} // end namespace gdcm

#endif //GDCMANONYMIZEEVENT_H
