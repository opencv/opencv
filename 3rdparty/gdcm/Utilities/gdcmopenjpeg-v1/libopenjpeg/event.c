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

#include "opj_includes.h"

/* ==========================================================
     Utility functions
   ==========================================================*/

#ifndef _WIN32
static char*
i2a(unsigned i, char *a, unsigned r) {
	if (i/r > 0) a = i2a(i/r,a,r);
	*a = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i%r];
	return a+1;
}

/** 
 Transforms integer i into an ascii string and stores the result in a; 
 string is encoded in the base indicated by r.
 @param i Number to be converted
 @param a String result
 @param r Base of value; must be in the range 2 - 36
 @return Returns a
*/
static char *
_itoa(int i, char *a, int r) {
	r = ((r < 2) || (r > 36)) ? 10 : r;
	if(i < 0) {
		*a = '-';
		*i2a(-i, a+1, r) = 0;
	}
	else *i2a(i, a, r) = 0;
	return a;
}

#endif /* !_WIN32 */

/* ----------------------------------------------------------------------- */

opj_event_mgr_t* OPJ_CALLCONV opj_set_event_mgr(opj_common_ptr cinfo, opj_event_mgr_t *event_mgr, void *context) {
	if(cinfo) {
		opj_event_mgr_t *previous = cinfo->event_mgr;
		cinfo->event_mgr = event_mgr;
		cinfo->client_data = context;
		return previous;
	}

	return NULL;
}

bool opj_event_msg(opj_common_ptr cinfo, int event_type, const char *fmt, ...) {
#define MSG_SIZE 512 /* 512 bytes should be more than enough for a short message */
	opj_msg_callback msg_handler = NULL;

	opj_event_mgr_t *event_mgr = cinfo->event_mgr;
	if(event_mgr != NULL) {
		switch(event_type) {
			case EVT_ERROR:
				msg_handler = event_mgr->error_handler;
				break;
			case EVT_WARNING:
				msg_handler = event_mgr->warning_handler;
				break;
			case EVT_INFO:
				msg_handler = event_mgr->info_handler;
				break;
			default:
				break;
		}
		if(msg_handler == NULL) {
			return false;
		}
	} else {
		return false;
	}

	if ((fmt != NULL) && (event_mgr != NULL)) {
		va_list arg;
		int str_length/*, i, j*/; /* UniPG */
		char message[MSG_SIZE];
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
		msg_handler(message, cinfo->client_data);
	}

	return true;
}

