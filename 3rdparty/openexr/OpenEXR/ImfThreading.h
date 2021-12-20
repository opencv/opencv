//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_THREADING_H
#define INCLUDED_IMF_THREADING_H

#include "ImfExport.h"
#include "ImfNamespace.h"

//-----------------------------------------------------------------------------
//
//	Threading support for the OpenEXR library
//
//	The OpenEXR library uses threads to perform reading and writing
//	of OpenEXR files in parallel.  The thread that calls the library
//	always performs the actual file IO (this is usually the main
//	application thread) whereas a several worker threads perform
//	data compression and decompression.  The number of worker
//	threads can be any non-negative value (a value of zero reverts
//	to single-threaded operation).  As long as there is at least
//	one worker thread, file IO and compression can potentially be
//	done concurrently through pinelining.  If there are two or more
//	worker threads, then pipelining as well as concurrent compression
//	of multiple blocks can be performed.
// 
//	Threading in the EXR library is controllable at two granularities:
//
//	* The functions in this file query and control the total number
//	  of worker threads, which will be created globally for the whole
//	  library.  Regardless of how many input or output files are
//	  opened simultaneously, the library will use at most this number
//	  of worker threads to perform all work.  The default number of
//	  global worker threads is zero (i.e. single-threaded operation;
//	  everything happens in the thread that calls the library).
//
//	* Furthermore, it is possible to set the number of threads that
//	  each input or output file should keep busy.  This number can
//	  be explicitly set for each file.  The default behavior is for
//	  each file to try to occupy all worker threads in the library's
//	  thread pool.
//
//-----------------------------------------------------------------------------

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


//-----------------------------------------------------------------------------
// Return the number of Imf-global worker threads used for parallel
// compression and decompression of OpenEXR files.
//-----------------------------------------------------------------------------
    
IMF_EXPORT int     globalThreadCount ();


//-----------------------------------------------------------------------------
// Change the number of Imf-global worker threads
//-----------------------------------------------------------------------------

IMF_EXPORT void    setGlobalThreadCount (int count);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
