/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDATAEVENT_H
#define GDCMDATAEVENT_H

#include "gdcmEvent.h"

namespace gdcm
{

/**
 * \brief DataEvent
 */
class DataEvent : public AnyEvent
{
public:
  typedef DataEvent Self;
  typedef AnyEvent Superclass;
  DataEvent(const char *bytes = 0, size_t len = 0):Bytes(bytes),Length(len) {}
  virtual ~DataEvent() {}
  virtual const char * GetEventName() const { return "DataEvent"; }
  virtual bool CheckEvent(const ::gdcm::Event* e) const
  { return (dynamic_cast<const Self*>(e) == NULL ? false : true) ; }
  virtual ::gdcm::Event* MakeObject() const
    { return new Self; }
  DataEvent(const Self&s) : AnyEvent(s){};

  void SetData(const char *bytes, size_t len) {
    Bytes = bytes;
    Length = len;
  }
  size_t GetDataLength() const { return Length; }
  const char *GetData() const { return Bytes; }

  //std::string GetValueAsString() const { return; }

private:
  void operator=(const Self&);
  const char *Bytes;
  size_t Length;
};


} // end namespace gdcm

#endif //GDCMDATAEVENT_H
