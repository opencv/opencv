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

#include "profile.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
//==============================================================================
static OPJ_PROFILE_LIST group_list [PGROUP_LASTGROUP];

//==============================================================================
static void GetTimeStamp(OPJ_UINT32 *pdwtime);

//==============================================================================
#define SetMajorSection(entry, major) \
        { group_list[ entry ].section = entry ; \
          group_list[ entry ].sectionName = #major ; }

//==============================================================================
void _ProfInit(void)
{
   // clear everything out
   memset(group_list, 0, sizeof(group_list));

   // set groups and parents for timing
   SetMajorSection(PGROUP_DWT,PGROUP_DWT);
   SetMajorSection(PGROUP_T1, PGROUP_T1);
   SetMajorSection(PGROUP_T2, PGROUP_T2);
} // ProfInit

//==============================================================================
void _ProfStart (OPJ_PROFILE_GROUP group)
{
   // make sure this hasn't been incorrectly started twice
   if (group_list[group].start)
   {
      return;
   }

   // get the start time
   GetTimeStamp(&(group_list[group].start));

} // _ProfStart

//==============================================================================
void _ProfStop(OPJ_PROFILE_GROUP group)
{
   // make sure we called start first
   if (!group_list[group].start)
   {
      return;
   }

   // get ending time
   GetTimeStamp(&(group_list[group].end));

   // calculate this latest elapsed interval
   group_list[group].total_time += group_list[group].end - group_list[group].start;

   // reset starting time
   group_list[group].start = 0;

   // incr the number of calls made
   ++group_list[group].totalCalls;

} // _ProfStop

//==============================================================================
#define proftracef(id,totalTime) \
        fprintf(p, #id "\t%u\t\t%6.6f\t\t%12.6f\t%2.2f%%\n",  \
                group_list[ id ].totalCalls, \
               (OPJ_FLOAT64) group_list[ id ].total_time / CLOCKS_PER_SEC, \
               ((OPJ_FLOAT64) group_list[ id ].total_time / (group_list[ id ].totalCalls ? group_list[ id ].totalCalls : 1)), \
         ((OPJ_FLOAT64) group_list[ id ].total_time / totalTime * 100))

#define proftracep(id,totalTime) \
        printf(#id "\t%u\t\t%6.6f\t\t%12.6f\t%2.2f%%\n",  \
                group_list[ id ].totalCalls, \
               (OPJ_FLOAT64) group_list[ id ].total_time / CLOCKS_PER_SEC, \
               ((OPJ_FLOAT64) group_list[ id ].total_time  / (group_list[ id ].totalCalls ? group_list[ id ].totalCalls : 1)), \
         ((OPJ_FLOAT64) group_list[ id ].total_time / totalTime * 100))

//==============================================================================
void _ProfSave(const OPJ_CHAR * pFileName)
{
  FILE *p = fopen(pFileName, "wt");
  OPJ_FLOAT64 totalTime = 0.;
  OPJ_UINT32 i;

  if (!p)
  {
    return;
  }

  for
    (i=0;i<PGROUP_LASTGROUP;++i)
  {
    totalTime += group_list[i].total_time;
  }

  fputs("\n\nProfile Data:\n", p);
  fputs("description\tnb calls\ttotal time (sec)\ttime per call\t%% of section\n", p);

  proftracef(PGROUP_DWT,totalTime);
  proftracef(PGROUP_T1,totalTime);
  proftracef(PGROUP_T2,totalTime);

   fputs("=== end of profile list ===\n\n", p);

   fclose(p);

} // _ProfSave

//==============================================================================
void _ProfPrint(void)
{
  OPJ_FLOAT64 totalTime = 0.;
  OPJ_UINT32 i;

  for
    (i=0;i<PGROUP_LASTGROUP;++i)
  {
    totalTime += group_list[i].total_time;
  }

  printf("\n\nProfile Data:\n");
  printf("description\tnb calls\ttotal time (sec)\ttime per call\t%% of section\n");

  proftracep(PGROUP_RATE, totalTime);
  proftracep(PGROUP_DC_SHIFT, totalTime);
  proftracep(PGROUP_MCT, totalTime);
  proftracep(PGROUP_DWT, totalTime);
  proftracep(PGROUP_T1, totalTime);
  proftracep(PGROUP_T2, totalTime);

  printf("\nTotal time: %6.3f second(s)\n", totalTime / CLOCKS_PER_SEC);

  printf("=== end of profile list ===\n\n");

} // _ProfPrint

//==============================================================================
static void GetTimeStamp(unsigned *time)
{
   *time = clock();

} // GetTimeStamp
