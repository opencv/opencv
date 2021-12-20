//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_KEY_CODE_H
#define INCLUDED_IMF_KEY_CODE_H

//-----------------------------------------------------------------------------
//
//	class KeyCode
// 	
// 	A KeyCode object uniquely identifies a motion picture film frame.
// 	The following fields specifiy film manufacturer, film type, film
// 	roll and the frame's position within the roll:
//
//	    filmMfcCode		film manufacturer code
//				range: 0 - 99
//
//	    filmType		film type code
// 				range: 0 - 99
//
//	    prefix		prefix to identify film roll
// 				range: 0 - 999999
//
//	    count		count, increments once every perfsPerCount
// 				perforations (see below)
// 				range: 0 - 9999
//
//	    perfOffset		offset of frame, in perforations from
// 				zero-frame reference mark
// 				range: 0 - 119
//
//	    perfsPerFrame	number of perforations per frame 
// 				range: 1 - 15
//
//				typical values:
//
//				    1 for 16mm film
//				    3, 4, or 8 for 35mm film
//				    5, 8 or 15 for 65mm film
//
//	    perfsPerCount	number of perforations per count 
// 				range: 20 - 120
//
//				typical values:
//
//				    20 for 16mm film
//				    64 for 35mm film
//				    80 or 120 for 65mm film
//
// 	For more information about the interpretation of those fields see
// 	the following standards and recommended practice publications:
//
// 	    SMPTE 254	Motion-Picture Film (35-mm) - Manufacturer-Printed
// 			Latent Image Identification Information
//
// 	    SMPTE 268M 	File Format for Digital Moving-Picture Exchange (DPX)
// 			(section 6.1)
//
// 	    SMPTE 270	Motion-Picture Film (65-mm) - Manufacturer- Printed
// 			Latent Image Identification Information
//
// 	    SMPTE 271	Motion-Picture Film (16-mm) - Manufacturer- Printed
// 			Latent Image Identification Information
//
//-----------------------------------------------------------------------------
#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

   
class IMF_EXPORT_TYPE KeyCode
{
  public:

    //-------------------------------------
    // Constructors and assignment operator
    //-------------------------------------

    IMF_EXPORT
    KeyCode (int filmMfcCode = 0,
	     int filmType = 0,
	     int prefix = 0,
	     int count = 0,
	     int perfOffset = 0,
	     int perfsPerFrame = 4,
	     int perfsPerCount = 64);

    IMF_EXPORT
    KeyCode (const KeyCode &other);
    ~KeyCode() = default;
    IMF_EXPORT
    KeyCode & operator = (const KeyCode &other);


    //----------------------------
    // Access to individual fields
    //----------------------------

    IMF_EXPORT
    int		filmMfcCode () const;
    IMF_EXPORT
    void	setFilmMfcCode (int filmMfcCode);

    IMF_EXPORT
    int		filmType () const;
    IMF_EXPORT
    void	setFilmType (int filmType);

    IMF_EXPORT
    int		prefix () const;
    IMF_EXPORT
    void	setPrefix (int prefix);

    IMF_EXPORT
    int		count () const;
    IMF_EXPORT
    void	setCount (int count);

    IMF_EXPORT
    int		perfOffset () const;
    IMF_EXPORT
    void	setPerfOffset (int perfOffset);

    IMF_EXPORT
    int		perfsPerFrame () const;
    IMF_EXPORT
    void	setPerfsPerFrame (int perfsPerFrame);

    IMF_EXPORT
    int		perfsPerCount () const;
    IMF_EXPORT
    void	setPerfsPerCount (int perfsPerCount);

  private:

    int		_filmMfcCode;
    int		_filmType;
    int		_prefix;
    int		_count;
    int		_perfOffset;
    int		_perfsPerFrame;
    int		_perfsPerCount;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
