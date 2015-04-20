/*
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Adapted from Herb Marselas
"Profiling, Data Analysis, Scalability, and Magic Numbers: Meeting the Minimum System Requirements for AGE OF EMPIRES 2: THE AGE OF KINGS"
Game Developer magazine
June, 2000 issue.
*/


#ifndef __PROFILE_H
#define __PROFILE_H

#include "openjpeg.h"
//==============================================================================
typedef enum
{
  PGROUP_RATE,
  PGROUP_DC_SHIFT,
  PGROUP_MCT,
  PGROUP_DWT,
  PGROUP_T1,
  PGROUP_T2,
  PGROUP_LASTGROUP
} OPJ_PROFILE_GROUP;

//==============================================================================
typedef struct PROFILELIST
{
  OPJ_UINT32   start;
  OPJ_UINT32   end;
    OPJ_UINT32   total_time;
    OPJ_UINT32   totalCalls;
    OPJ_PROFILE_GROUP   section;
    const OPJ_CHAR     *sectionName; // string name of the profile group
} OPJ_PROFILE_LIST;

//==============================================================================
void _ProfStart(OPJ_PROFILE_GROUP group);
void _ProfStop (OPJ_PROFILE_GROUP group);

//==============================================================================
//==============================================================================
#ifdef _PROFILE
#define PROFINIT() _ProfInit();
#define PROFSTART (group) _ProfStart (group);
#define PROFSTOP (group) _ProfStop (group);
#define PROFSAVE(file) _ProfSave(file);
#define PROFPRINT() _ProfPrint();
#else
#define PROFINIT()
#define PROFSTART(group)
#define PROFSTOP (group)
#define PROFSAVE(file)
#define PROFPRINT()
#endif // !_PROFILE

//==============================================================================
#endif // __PROFILE_H
