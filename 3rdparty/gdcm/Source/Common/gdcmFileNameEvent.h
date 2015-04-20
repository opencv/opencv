/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILENAMEEVENT_H
#define GDCMFILENAMEEVENT_H

#include "gdcmEvent.h"
#include "gdcmTag.h"

namespace gdcm
{

/**
 * \brief FileNameEvent
 * Special type of event triggered during processing of FileSet
 *
 * \see AnyEvent
 */
class FileNameEvent : public AnyEvent
{
public:
  typedef FileNameEvent Self;
  typedef AnyEvent Superclass;
  FileNameEvent(const char *s = ""):m_FileName(s) {}
  virtual ~FileNameEvent() {}
  virtual const char * GetEventName() const { return "FileNameEvent"; }
  virtual bool CheckEvent(const ::gdcm::Event* e) const
    { return dynamic_cast<const Self*>(e) ? true : false; }
  virtual ::gdcm::Event* MakeObject() const
    { return new Self; }
  FileNameEvent(const Self&s) : AnyEvent(s){};

  void SetFileName(const char *f) { m_FileName = f; }
  const char *GetFileName() const { return m_FileName.c_str(); }
private:
  void operator=(const Self&);
  std::string m_FileName;
};


} // end namespace gdcm

#endif //GDCMFILENAMEEVENT_H
