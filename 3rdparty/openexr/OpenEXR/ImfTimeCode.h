//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_TIME_CODE_H
#define INCLUDED_IMF_TIME_CODE_H

#include "ImfExport.h"
#include "ImfNamespace.h"

//-----------------------------------------------------------------------------
//
//	class TimeCode
// 	
// 	A TimeCode object stores time and control codes as described
// 	in SMPTE standard 12M-1999.  A TimeCode object contains the
// 	following fields:
//
// 	    Time Address:
//
//		hours			integer, range 0 - 23
//		minutes			integer, range 0 - 59
//		seconds			integer, range 0 - 59
//		frame 			integer, range 0 - 29
//
// 	    Flags:
//
// 		drop frame flag		boolean
//		color frame flag	boolean
//		field/phase flag	boolean
//		bgf0			boolean
//		bgf1			boolean
//		bgf2			boolean
//
// 	    Binary groups for user-defined data and control codes:
//
//		binary group 1		integer, range 0 - 15
//		binary group 2		integer, range 0 - 15
//		...
//		binary group 8		integer, range 0 - 15
//
//	Class TimeCode contains methods to convert between the fields
//	listed above and a more compact representation where the fields
//	are packed into two unsigned 32-bit integers.  In the packed
//	integer representations, bit 0 is the least significant bit,
//	and bit 31 is the most significant bit of the integer value.
//
//	The time address and flags fields can be packed in three
//	different ways:
//
//	      bits	packing for	  packing for	    packing for
//	    		24-frame 	  60-field 	    50-field
//	    		film		  television	    television
//
//	     0 -  3	frame units	  frame units	    frame units
//	     4 -  5	frame tens	  frame tens	    frame tens
//	     6		unused, set to 0  drop frame flag   unused, set to 0
//	     7		unused, set to 0  color frame flag  color frame flag
//	     8 - 11	seconds units	  seconds units	    seconds units
//	    12 - 14	seconds tens	  seconds tens	    seconds tens
//	    15		phase flag	  field/phase flag  bgf0
//	    16 - 19	minutes units	  minutes units	    minutes units
//	    20 - 22	minutes tens	  minutes tens	    minutes tens
//	    23		bgf0		  bgf0		    bgf2
//	    24 - 27	hours units	  hours units	    hours units
//	    28 - 29	hours tens	  hours tens	    hours tens
//	    30		bgf1		  bgf1		    bgf1
//	    31		bgf2		  bgf2		    field/phase flag
//
//	User-defined data and control codes are packed as follows:
//
//	      bits	field
//
//	     0 -  3	binary group 1
//	     4 -  7	binary group 2
//	     8 - 11	binary group 3
//	    12 - 15	binary group 4
//	    16 - 19	binary group 5
//	    20 - 23	binary group 6
//	    24 - 27	binary group 7
//	    28 - 31	binary group 8
//
//-----------------------------------------------------------------------------

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

   
class IMF_EXPORT_TYPE TimeCode
{
  public:

    //---------------------
    // Bit packing variants
    //---------------------

    enum IMF_EXPORT_ENUM Packing
    {
	TV60_PACKING,		// packing for 60-field television
	TV50_PACKING,		// packing for 50-field television
	FILM24_PACKING		// packing for 24-frame film
    };


    //-------------------------------------
    // Constructors and assignment operator
    //-------------------------------------

    IMF_EXPORT
    TimeCode ();  // all fields set to 0 or false

    IMF_EXPORT
    TimeCode (int hours,
	      int minutes,
	      int seconds,
	      int frame,
	      bool dropFrame = false,
	      bool colorFrame = false,
	      bool fieldPhase = false,
	      bool bgf0 = false,
	      bool bgf1 = false,
	      bool bgf2 = false,
	      int binaryGroup1 = 0,
	      int binaryGroup2 = 0,
	      int binaryGroup3 = 0,
	      int binaryGroup4 = 0,
	      int binaryGroup5 = 0,
	      int binaryGroup6 = 0,
	      int binaryGroup7 = 0,
	      int binaryGroup8 = 0);

    IMF_EXPORT
    TimeCode (unsigned int timeAndFlags,
	      unsigned int userData = 0,
	      Packing packing = TV60_PACKING);

    IMF_EXPORT
    TimeCode (const TimeCode &other);

    ~TimeCode () = default;

    IMF_EXPORT
    TimeCode & operator = (const TimeCode &other);


    //----------------------------
    // Access to individual fields
    //----------------------------

    IMF_EXPORT
    int		hours () const;
    IMF_EXPORT
    void	setHours (int value);

    IMF_EXPORT
    int		minutes () const;
    IMF_EXPORT
    void	setMinutes (int value);

    IMF_EXPORT
    int		seconds () const;
    IMF_EXPORT
    void	setSeconds (int value);

    IMF_EXPORT
    int		frame () const;
    IMF_EXPORT
    void	setFrame (int value);

    IMF_EXPORT
    bool	dropFrame () const;
    IMF_EXPORT
    void	setDropFrame (bool value);

    IMF_EXPORT
    bool	colorFrame () const;
    IMF_EXPORT
    void	setColorFrame (bool value);

    IMF_EXPORT
    bool	fieldPhase () const;
    IMF_EXPORT
    void	setFieldPhase (bool value);

    IMF_EXPORT
    bool	bgf0 () const;
    IMF_EXPORT
    void	setBgf0 (bool value);

    IMF_EXPORT
    bool	bgf1 () const;
    IMF_EXPORT
    void	setBgf1 (bool value);

    IMF_EXPORT
    bool	bgf2 () const;
    IMF_EXPORT
    void	setBgf2 (bool value);

    IMF_EXPORT
    int		binaryGroup (int group) const; // group must be between 1 and 8
    IMF_EXPORT
    void	setBinaryGroup (int group, int value);

    
    //---------------------------------
    // Access to packed representations
    //---------------------------------

    IMF_EXPORT
    unsigned int	timeAndFlags (Packing packing = TV60_PACKING) const;

    IMF_EXPORT
    void		setTimeAndFlags (unsigned int value,
					 Packing packing = TV60_PACKING);

    IMF_EXPORT
    unsigned int	userData () const;

    IMF_EXPORT
    void		setUserData (unsigned int value);
    
    
    //---------
    // Equality
    //---------
    
    IMF_EXPORT
    bool		operator == (const TimeCode &v) const;    
    IMF_EXPORT
    bool		operator != (const TimeCode &v) const;
    
  private:

    unsigned int	_time;
    unsigned int	_user;
};



OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
