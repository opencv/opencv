/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDATASETEVENT_H
#define GDCMDATASETEVENT_H

#include "gdcmEvent.h"
#include "gdcmDataSet.h"

namespace gdcm
{

/**
 * \brief DataSetEvent
 * Special type of event triggered during the DataSet store/move process
 *
 * \see
 */
class DataSetEvent : public AnyEvent
{
public:
  typedef DataSetEvent Self;
  typedef AnyEvent Superclass;
  DataSetEvent(DataSet const *ds = NULL):m_DataSet(ds) {}
  virtual ~DataSetEvent() {}
  virtual const char * GetEventName() const { return "DataSetEvent"; }
  virtual bool CheckEvent(const ::gdcm::Event* e) const
  { return (dynamic_cast<const Self*>(e) == NULL ? false : true) ; }
  virtual ::gdcm::Event* MakeObject() const
    { return new Self; }
  DataSetEvent(const Self&s) : AnyEvent(s){};

  DataSet const & GetDataSet() const { return *m_DataSet; }
private:
  void operator=(const Self&);
  const DataSet *m_DataSet;
};


} // end namespace gdcm

#endif //GDCMANONYMIZEEVENT_H
