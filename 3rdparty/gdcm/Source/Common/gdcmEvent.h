/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMEVENT_H
#define GDCMEVENT_H

#include "gdcmTypes.h"

namespace gdcm
{
//-----------------------------------------------------------------------------
/**
 * \brief superclass for callback/observer methods
 * \see Command Subject
 */
class GDCM_EXPORT Event
{
public :
  Event();
  Event(const Event&);
  virtual ~Event();

  /**  Create an Event of this type This method work as a Factory for
   *  creating events of each particular type. */
  virtual Event* MakeObject() const = 0;

  /** Print Event information.  This method can be overridden by
   * specific Event subtypes.  The default is to print out the type of
   * the event. */
  virtual void Print(std::ostream& os) const;

  /** Return the StringName associated with the event. */
  virtual const char * GetEventName(void) const = 0;

  /** Check if given event matches or derives from this event. */
  virtual bool CheckEvent(const Event*) const = 0;

protected:
private:
  void operator=(const Event&);  // Not implemented.
};

/// Generic inserter operator for Event and its subclasses.
inline std::ostream& operator<<(std::ostream& os, Event &e)
{
  e.Print(os);
  return os;
}

/*
 *  Macro for creating new Events
 */
#define gdcmEventMacro( classname , super ) \
 /** \brief classname */  \
 class  classname : public super { \
   public: \
     typedef classname Self; \
     typedef super Superclass; \
     classname() {} \
     virtual ~classname() {} \
     virtual const char * GetEventName() const { return #classname; } \
     virtual bool CheckEvent(const ::gdcm::Event* e) const \
       { return dynamic_cast<const Self*>(e) ? true : false; } \
     virtual ::gdcm::Event* MakeObject() const \
       { return new Self; } \
     classname(const Self&s) : super(s){}; \
   private: \
     void operator=(const Self&); \
 }

/**
 *      Define some common GDCM events
 */
gdcmEventMacro( NoEvent            , Event );
gdcmEventMacro( AnyEvent           , Event );
gdcmEventMacro( StartEvent         , AnyEvent );
gdcmEventMacro( EndEvent           , AnyEvent );
//gdcmEventMacro( ProgressEvent      , AnyEvent );
gdcmEventMacro( ExitEvent          , AnyEvent );
gdcmEventMacro( AbortEvent         , AnyEvent );
gdcmEventMacro( ModifiedEvent      , AnyEvent );
gdcmEventMacro( InitializeEvent    , AnyEvent );
gdcmEventMacro( IterationEvent     , AnyEvent );
//gdcmEventMacro( AnonymizeEvent     , AnyEvent );
gdcmEventMacro( UserEvent          , AnyEvent );


} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMEVENT_H
