/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2005-2006, Dept. of Electronic and Information Engineering, Universita' degli Studi di Perugia, Italy
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

#ifdef USE_JPWL

/**
@file crc.h
@brief Functions used to compute the 16- and 32-bit CRC of byte arrays

*/

#ifndef __CRC16_HEADER__
#define __CRC16_HEADER__

/** file: CRC16.HPP
 *
 * CRC - Cyclic Redundancy Check (16-bit)
 *
 * A CRC-checksum is used to be sure, the data hasn't changed or is false.
 * To create a CRC-checksum, initialise a check-variable (unsigned short),
 * and set this to zero. Than call for every byte in the file (e.g.) the
 * procedure updateCRC16 with this check-variable as the first parameter,
 * and the byte as the second. At the end, the check-variable contains the
 * CRC-checksum.
 *
 * implemented by Michael Neumann, 14.06.1998
 *
 */
void updateCRC16(unsigned short *, unsigned char);

#endif /* __CRC16_HEADER__ */


#ifndef __CRC32_HEADER__
#define __CRC32_HEADER__

/** file: CRC32.HPP
 *
 * CRC - Cyclic Redundancy Check (32-bit)
 *
 * A CRC-checksum is used to be sure, the data hasn't changed or is false.
 * To create a CRC-checksum, initialise a check-variable (unsigned short),
 * and set this to zero. Than call for every byte in the file (e.g.) the
 * procedure updateCRC32 with this check-variable as the first parameter,
 * and the byte as the second. At the end, the check-variable contains the
 * CRC-checksum.
 *
 * implemented by Michael Neumann, 14.06.1998
 *
 */
void updateCRC32(unsigned long *, unsigned char);

#endif /* __CRC32_HEADER__ */


#endif /* USE_JPWL */
