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
#ifndef __EVENT_H
#define __EVENT_H
/**
@file event.h
@brief Implementation of a event callback system

The functions in EVENT.C have for goal to send output messages (errors, warnings, debug) to the user.
*/

#define EVT_ERROR	1	/**< Error event type */
#define EVT_WARNING	2	/**< Warning event type */
#define EVT_INFO	4	/**< Debug event type */

/** @defgroup EVENT EVENT - Implementation of a event callback system */
/*@{*/

/** @name Exported functions (see also openjpeg.h) */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Write formatted data to a string and send the string to a user callback. 
@param cinfo Codec context info
@param event_type Event type or callback to use to send the message
@param fmt Format-control string (plus optionnal arguments)
@return Returns true if successful, returns false otherwise
*/
bool opj_event_msg(opj_common_ptr cinfo, int event_type, const char *fmt, ...);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __EVENT_H */
