/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDIMSE_H
#define GDCMDIMSE_H

#include "gdcmTypes.h"

namespace gdcm
{

namespace network
{

/**
 * \brief DIMSE
 * PS 3.7 - 2009
 * Annex E Command Dictionary (Normative)
 * E.1 REGISTRY OF DICOM COMMAND ELEMENTS
 * Table E.1-1
 * COMMAND FIELDS (PART 1)
 */
class DIMSE {
public:
  typedef enum {
C_STORE_RQ         = 0x0001,
C_STORE_RSP        = 0x8001,
C_GET_RQ           = 0x0010,
C_GET_RSP          = 0x8010,
C_FIND_RQ          = 0x0020,
C_FIND_RSP         = 0x8020,
C_MOVE_RQ          = 0x0021,
C_MOVE_RSP         = 0x8021,
C_ECHO_RQ          = 0x0030,
C_ECHO_RSP         = 0x8030,
N_EVENT_REPORT_RQ  = 0x0100,
N_EVENT_REPORT_RSP = 0x8100,
N_GET_RQ           = 0x0110,
N_GET_RSP          = 0x8110,
N_SET_RQ           = 0x0120,
N_SET_RSP          = 0x8120,
N_ACTION_RQ        = 0x0130,
N_ACTION_RSP       = 0x8130,
N_CREATE_RQ        = 0x0140,
N_CREATE_RSP       = 0x8140,
N_DELETE_RQ        = 0x0150,
N_DELETE_RSP       = 0x8150,
C_CANCEL_RQ        = 0x0FFF
  } CommandTypes;
};

/*
9.1.5.1 C-ECHO parameters
Table 9.1-5
C-ECHO PARAMETERS
*/
class CEchoRQ
{
public:
  uint16_t          MessageID;                              /* M */
  UIComp            AffectedSOPClassUID;                    /* M */
};

class CEchoRSP
{
public:
/*
Message ID M U
Message ID Being Responded To  M
Affected SOP Class UID M U(=)
Status  M
*/
};

/**
PS 3.4 - 2009
Table B.2-1 C-STORE STATUS
 */
class CFind
{
/*
Failure Refused: Out of Resources A700 (0000,0902)
Identifier does not match SOP Class A900 (0000,0901)
(0000,0902)
Unable to process Cxxx (0000,0901)
(0000,0902)
Cancel Matching terminated due to Cancel
request
FE00 None
Success Matching is complete – No final Identifier
is supplied.
0000 None
Pending Matches are continuing – Current Match
is supplied and any Optional Keys were
supported in the same manner as
Required Keys.
FF00 Identifier
Matches are continuing – Warning that
one or more Optional Keys were not
supported for existence and/or matching
for this Identifier.
FF01 Identifier
*/
};


} // end namespace network

} // end namespace gdcm

#endif //GDCMDIMSE_H
