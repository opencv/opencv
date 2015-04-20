/*
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
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

#include "event.h"
#include "openjpeg.h"
#include "opj_includes.h"


/* ==========================================================
     Utility functions
   ==========================================================*/

#if !defined(_MSC_VER) && !defined(__MINGW32__)
static OPJ_CHAR*
i2a(OPJ_UINT32 i, OPJ_CHAR *a, OPJ_UINT32 r) {
  if (i/r > 0) a = i2a(i/r,a,r);
  *a = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i%r];
  return a+1;
}
#endif
/* ----------------------------------------------------------------------- */

bool opj_event_msg(opj_event_mgr_t * p_event_mgr, OPJ_INT32 event_type, const OPJ_CHAR *fmt, ...) {
#define MSG_SIZE 512 /* 512 bytes should be more than enough for a short message */
  opj_msg_callback msg_handler = 00;
  void * l_data = 00;


  if(p_event_mgr != 00) {
    switch(event_type) {
      case EVT_ERROR:
        msg_handler = p_event_mgr->error_handler;
        l_data = p_event_mgr->m_error_data;
        break;
      case EVT_WARNING:
        msg_handler = p_event_mgr->warning_handler;
        l_data = p_event_mgr->m_warning_data;
        break;
      case EVT_INFO:
        msg_handler = p_event_mgr->info_handler;
        l_data = p_event_mgr->m_info_data;
        break;
      default:
        break;
    }
    if(msg_handler == 00) {
      return false;
    }
  } else {
    return false;
  }

  if ((fmt != 00) && (p_event_mgr != 00)) {
    va_list arg;
    OPJ_INT32 str_length/*, i, j*/; /* UniPG */
    OPJ_CHAR message[MSG_SIZE];
    memset(message, 0, MSG_SIZE);
    /* initialize the optional parameter list */
    va_start(arg, fmt);
    /* check the length of the format string */
    str_length = (strlen(fmt) > MSG_SIZE) ? MSG_SIZE : strlen(fmt);
    /* parse the format string and put the result in 'message' */
    vsprintf(message, fmt, arg); /* UniPG */
    /* deinitialize the optional parameter list */
    va_end(arg);

    /* output the message to the user program */
    msg_handler(message, l_data);
  }

  return true;
}
