/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/
/*
************************************************************
param.h

The basic system parameters are kept here.

************************************************************
*/

#ifndef PARAM_DONE
#define PARAM_DONE

/* This is the general definition for the size and width of the
   JPEG blocks. Do not change.
   */
#define BLOCKSIZE 64
#define BLOCKWIDTH 8
#define BLOCKHEIGHT 8

/* Definitions for JPEG and internal compatibility. */

#define MAXIMUM_HORIZONTAL_FREQUENCY 15
#define MAXIMUM_VERTICAL_FREQUENCY 15

#define MAXIMUM_JPEG_HORIZONTAL_FREQUENCY 4
#define MAXIMUM_JPEG_VERTICAL_FREQUENCY 4

#define MINIMUM_BUFFERSIZE 16

#define MAXIMUM_UNSIGNED16 65535
#define MAXIMUM_RESYNC_INTERVAL 65535
#define MAXIMUM_BUFFERSIZE 65535
#define MAXIMUM_IMAGE_HEIGHT 65535
#define MAXIMUM_IMAGE_WIDTH 65535

/* Devices: Number of active devices operating at one time.
   Quantization tables, huffman tables, etc. are all devices.
   */
#define MAXIMUM_DEVICES 16

/* Sources: Number of active sources in stream at one time.
   A source is one interleave possibility.
   */
#define MAXIMUM_SOURCES 16

/* Components: Number of components that can be active per frame.
   A component consists of one complete plane of the image.
*/
#define MAXIMUM_COMPONENTS 256

/* Q value as defined by archaic and now defunct F-Factor:
   Used to rescale quantization matrices.
 */

#define Q_PRECISION 50

/* Scan component threshold is the maximum number of components put
in per scan */

#define SCAN_COMPONENT_THRESHOLD 4

/* Mask to be used for creating files. */
#define UMASK 0666  /* Octal */

/* Buffersize is used as the default I/O buffer. A smaller size ensures
   less storage space. A larger size requires more storage space.
   256 seems like a good number for smaller machines, but for machines
   with greater than 0.5 MB of memory, 1024 would be better because
   it reduces on the number of seeks necessary.  Helpful for macro-sized
   words such as 16 bit or 24 bit to have a proper multiple of such
   word.
   */
#define BUFFERSIZE 256

/* Lossless Buffersize is a variable that is kept for the lossless
   streams.  It can be any positive number, though a larger number
   will speed up the processing of information.  A large number also
   will cause (MAXIMUM_SOURCES)*(LOSSLESSBUFFERSIZE)*sizeof(int)
   storage consumption.  To ensure proper operation, this should
   be equivalent to the BUFFERSIZE variable * dimensions of the
   scan frequencies so that two fetches are not required for filling
   the lossless buffer. (It would make having the upper buffer useless).
   The minimum number is (MAX_HF+1)*(MAX_VF+1) */

#define LOSSLESSBUFFERSIZE 289

/* Number of streams is the number of active read/write streams possible.
   For all jpeg operations, this value is 1.*/

#define NUMBER_OF_STREAMS 1

#define ISO_DCT
#define LEE_DCT

#endif
